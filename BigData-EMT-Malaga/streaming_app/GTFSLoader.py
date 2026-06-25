"""
GTFSLoader.py - GTFS Timetable Data Loader

Loads and processes GTFS (General Transit Feed Specification) data for the
EMT Malaga bus network. This module handles the static timetable data that
is used for schedule lookups and route geometry.

Key responsibilities:
    - Load stops, trips, and stop_times from GTFS CSV files
    - Filter trips by target bus line and current date
    - Build route geometry (ordered stops per direction)
    - Build schedule lookup tables (arrival times per stop)
"""

import os
from datetime import datetime
from typing import Dict, List

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

from Config import Config


class GTFSLoader:
    """
    Loads and processes GTFS timetable data using Spark.
    
    This loader reads GTFS CSV files and creates optimized lookup structures
    for use in the streaming application. All heavy processing is done with
    Spark, and only the final lookup dictionaries are collected to the driver.
    
    Attributes:
        stops_by_dir: Dict mapping direction_id to list of stop dictionaries
        schedule_lookup: Dict mapping (stop_id, direction_id) to list of arrival times
        destination_by_dir: Dict mapping direction_id to trip headsign (e.g., "Universidad")
    """
    
    def __init__(self, spark: SparkSession, config: Config):
        """
        Initialize the loader and process GTFS data.
        
        Args:
            spark: Active SparkSession for distributed processing
            config: Application configuration with file paths
        """
        self.spark = spark
        self.config = config
        
        # Output structures
        self.stops_by_dir: Dict[int, List[Dict]] = {}
        self.schedule_lookup: Dict[tuple, List[str]] = {}
        self.destination_by_dir: Dict[int, str] = {}
        
        self._load()
    
    def _load(self) -> None:
        """Load and process all GTFS data."""
        if not os.path.exists(self.config.STOPS_FILE):
            print("ERROR: GTFS files not found.")
            return
        
        today_str = datetime.now().strftime("%Y%m%d")
        print(f"Loading schedules for date: {today_str}")
        
        # Load base DataFrames
        stops_df = self._load_stops()
        trips_df = self._load_trips()
        stop_times_df = self._load_stop_times()
        
        # Filter trips by active services for today
        trips_df = self._filter_by_calendar(trips_df, today_str)
        
        # Filter for target line only
        trips_target = trips_df.filter(F.col("route_id") == self.config.TARGET_LINE)
        
        # Extract destination names per direction
        self._extract_destinations(trips_target)
        
        # Join all data together
        merged_df = stop_times_df \
            .join(F.broadcast(trips_target), "trip_id") \
            .join(F.broadcast(stops_df), "stop_id") \
            .cache()
        
        # Build output structures
        self._build_route_geometry(merged_df)
        self._build_schedule_lookup(merged_df)
        
        merged_df.unpersist()
    
    def _load_stops(self):
        """Load stops.csv with proper type casting."""
        return self.spark.read.csv(
            self.config.STOPS_FILE, 
            header=True, 
            inferSchema=True
        ).withColumn("stop_id", F.col("stop_id").cast("string")) \
         .withColumn("stop_code", F.col("stop_code").cast("string"))
    
    def _load_trips(self):
        """Load trips.csv with proper type casting."""
        return self.spark.read.csv(
            self.config.TRIPS_FILE,
            header=True,
            inferSchema=True
        ).withColumn("route_id", F.col("route_id").cast("string")) \
         .withColumn("trip_id", F.col("trip_id").cast("string")) \
         .withColumn("direction_id", F.col("direction_id").cast("int"))
    
    def _load_stop_times(self):
        """Load stop_times.csv with proper type casting."""
        return self.spark.read.csv(
            self.config.STOP_TIMES_FILE,
            header=True,
            inferSchema=True
        ).withColumn("stop_id", F.col("stop_id").cast("string")) \
         .withColumn("trip_id", F.col("trip_id").cast("string"))
    
    def _filter_by_calendar(self, trips_df, today_str: str):
        """
        Filter trips to only those running today.
        
        Uses calendar_dates.csv to find active services. Falls back to
        using all schedules if no services are found for today.
        """
        calendar_file = os.path.join(self.config.TIMETABLE_DIR, "calendar_dates.csv")
        
        if not os.path.exists(calendar_file):
            return trips_df
        
        cal_dates_df = self.spark.read.csv(
            calendar_file, 
            header=True, 
            inferSchema=True
        ).withColumn("date", F.col("date").cast("string"))
        
        # Get service_ids running today (exception_type=1 means added)
        active_services = cal_dates_df \
            .filter((F.col("date") == today_str) & (F.col("exception_type") == 1)) \
            .select("service_id").distinct()
        
        active_count = active_services.count()
        print(f"Found {active_count} active services for today")
        
        if active_count > 0:
            return trips_df.join(F.broadcast(active_services), "service_id")
        else:
            print("WARNING: No services found for today, using all schedules")
            return trips_df
    
    def _extract_destinations(self, trips_df) -> None:
        """Extract trip headsigns (destinations) for each direction."""
        dest_rows = trips_df \
            .select("direction_id", "trip_headsign") \
            .distinct() \
            .collect()
        
        for row in dest_rows:
            self.destination_by_dir[int(row["direction_id"])] = row["trip_headsign"]
        
        print(f"Direction destinations: {self.destination_by_dir}")
    
    def _build_route_geometry(self, merged_df) -> None:
        """
        Build ordered list of stops for each direction.
        
        This creates the route geometry used for finding the nearest
        stop and determining stop sequences.
        """
        # Window to get one canonical stop per sequence number
        window = Window.partitionBy("direction_id", "stop_sequence").orderBy("trip_id")
        
        canonical_df = merged_df \
            .withColumn("row_num", F.row_number().over(window)) \
            .filter(F.col("row_num") == 1) \
            .select(
                "direction_id", "stop_id", "stop_code", 
                "stop_name", "stop_lat", "stop_lon", "stop_sequence"
            ) \
            .orderBy("direction_id", "stop_sequence")
        
        rows = canonical_df.collect()
        
        for row in rows:
            d_id = int(row["direction_id"])
            if d_id not in self.stops_by_dir:
                self.stops_by_dir[d_id] = []
            self.stops_by_dir[d_id].append(row.asDict())
        
        total_stops = sum(len(stops) for stops in self.stops_by_dir.values())
        print(f"Loaded {total_stops} stops across {len(self.stops_by_dir)} directions")
    
    def _build_schedule_lookup(self, merged_df) -> None:
        """
        Build mapping of (stop_id, direction) -> [arrival_times].
        
        This lookup is used for finding the scheduled arrival time
        closest to the current time during streaming.
        """
        schedule_df = merged_df \
            .select("stop_id", "direction_id", "arrival_time") \
            .orderBy("arrival_time") \
            .groupBy("stop_id", "direction_id") \
            .agg(F.collect_list("arrival_time").alias("times"))
        
        sched_rows = schedule_df.collect()
        
        for row in sched_rows:
            key = (str(row["stop_id"]), int(row["direction_id"]))
            self.schedule_lookup[key] = row["times"]
        
        print(f"Loaded {len(sched_rows)} schedule entries")