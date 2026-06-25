# This script builds a complete Spark-based processing workflow for EMT Málaga Line 11.
# It loads static GTFS data, retrieves real-time bus locations, performs map matching to identify
# the closest stop and next scheduled stop, and generates structured live-status outputs.
# The design combines distributed Spark operations with Python geospatial utilities and a
# continuous polling loop to maintain updated snapshots of bus movements.

import os
import time
import csv
import io
from datetime import datetime, timedelta

import requests
from geopy.distance import geodesic

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, TimestampType
)
from pyspark.sql.window import Window

# This section configures paths for GTFS and output directories.
# GTFS static data is stored locally, while snapshots of real-time data will be saved in the "dati_linea11" directory.
DATA_DIR = "dati_linea11"
os.makedirs(DATA_DIR, exist_ok=True)

GTFS_DIR = "realData"

STOPS = os.path.join(GTFS_DIR, "stops.txt")
TRIPS = os.path.join(GTFS_DIR, "trips.txt")
STOP_TIMES = os.path.join(GTFS_DIR, "stop_times.txt")

# This is the public endpoint used to fetch EMT real-time GPS positions in CSV format.
REALTIME_URL = (
    "https://datosabiertos.malaga.eu/recursos/transporte/EMT/EMTlineasUbicaciones/lineasyubicaciones.csv"
)

# This threshold determines when a bus is considered to have "reached" a stop based on geodesic distance.
STOP_REACHED_THRESHOLD = 150  # meters

# The SparkSession is created for local execution.
# Logging levels are reduced for readability during continuous execution.
def create_spark():
    spark = (
        SparkSession.builder
        .appName("StreamingLinea11Spark")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

# Spark writes distributed CSV output in multiple part files.
# This helper function rewrites a DataFrame as a single CSV file by coalescing to one partition,
# then renaming the resulting part file so it can be consumed by external tools expecting a single-file output.
def write_single_csv(df, path):
    tmp_dir = path + "_tmp"
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_dir)

    import glob
    import shutil
    part_file = glob.glob(os.path.join(tmp_dir, "part-*.csv"))[0]

    if os.path.exists(path):
        os.remove(path)

    shutil.move(part_file, path)
    shutil.rmtree(tmp_dir)

# This function loads GTFS static data required for Line 11.
# It filters trip records to retain only those associated with route_id 11.
# It joins stop times, trips, and stops to build a unified GTFS schedule DataFrame.
# A dictionary of stops grouped by direction is also prepared and broadcast later for map matching.
def load_gtfs_line11(spark):
    df_stops = spark.read.csv(STOPS, header=True, inferSchema=True)
    df_trips = spark.read.csv(TRIPS, header=True, inferSchema=True)
    df_times = spark.read.csv(STOP_TIMES, header=True, inferSchema=True)

    df_trips11 = df_trips.filter(F.col("route_id") == 11)
    df_times11 = df_times.join(df_trips11.select("trip_id"), on="trip_id", how="inner")

    df_sched = (
        df_times11.join(df_stops, on="stop_id", how="left")
        .join(df_trips11.select("trip_id", "direction_id"), on="trip_id", how="left")
    )

    df_sched = df_sched.withColumn("stop_sequence", F.col("stop_sequence").cast("int"))
    df_sched = df_sched.withColumn("arrival_time_str", F.col("arrival_time").cast("string"))

    # This step prepares a lookup structure for map matching.
    # Stops are gathered into a Python dictionary keyed by direction_id to optimize stop matching at runtime.
    stops_dir = (
        df_sched.select(
            "direction_id",
            "stop_id",
            "stop_name",
            "stop_lat",
            "stop_lon",
            "stop_sequence"
        )
        .dropDuplicates()
    )

    stops_by_dir = {}
    for row in stops_dir.collect():
        d = int(row["direction_id"]) if row["direction_id"] is not None else 0
        stops_by_dir.setdefault(d, []).append(
            (
                row["stop_id"],
                row["stop_name"],
                float(row["stop_lat"]),
                float(row["stop_lon"]),
                int(row["stop_sequence"]) if row["stop_sequence"] is not None else None,
            )
        )

    return df_sched, stops_by_dir

# This function creates a Spark UDF that performs map matching.
# Each bus location is compared with GTFS stops to identify the nearest stop and the next stop along the route.
# The stop dictionaries are broadcast to avoid repeatedly sending large objects to executors.
def build_position_udf(spark, stops_by_dir):

    schema = StructType([
        StructField("current_stop_id", StringType(), True),
        StructField("current_stop_name", StringType(), True),
        StructField("stop_sequence", IntegerType(), True),
        StructField("next_stop_id", StringType(), True),
        StructField("next_stop_name", StringType(), True),
        StructField("distance_m", DoubleType(), True),
    ])

    bc_stops = spark.sparkContext.broadcast(stops_by_dir)

    @F.udf(returnType=schema)
    def map_position(lat, lon, sentido):
        if lat is None or lon is None:
            return (None, None, None, None, None, None)

        try:
            sentido_int = int(sentido)
        except:
            sentido_int = 1

        direction_id = 0 if sentido_int == 1 else 1
        stops_list = bc_stops.value.get(direction_id, [])
        if not stops_list:
            all_stops = []
            for lst in bc_stops.value.values():
                all_stops.extend(lst)
            stops_list = all_stops

        if not stops_list:
            return (None, None, None, None, None, None)

        # This loop finds the closest stop to the bus using geodesic distance.
        best = None
        best_dist = 1e12
        for stop_id, stop_name, s_lat, s_lon, seq in stops_list:
            try:
                d = geodesic((lat, lon), (s_lat, s_lon)).meters
            except:
                continue
            if d < best_dist:
                best_dist = d
                best = (stop_id, stop_name, s_lat, s_lon, seq)

        if best is None:
            return (None, None, None, None, None, None)

        stop_id, stop_name, s_lat, s_lon, seq = best

        # If the bus is close enough to a stop, the stop is considered reached.
        # The next stop is identified by ordering stops by their sequence.
        if best_dist <= STOP_REACHED_THRESHOLD:
            current_stop = (stop_id, stop_name, seq)
            sorted_st = sorted(stops_list, key=lambda x: (x[4] if x[4] else 10**9))
            next_stop = None
            for st in sorted_st:
                if st[4] and seq and st[4] > seq:
                    next_stop = st
                    break

            if next_stop is None:
                return (
                    stop_id,
                    stop_name,
                    seq,
                    None,
                    None,
                    float(best_dist)
                )
            return (
                stop_id,
                stop_name,
                seq,
                next_stop[0],
                next_stop[1],
                float(best_dist)
            )

        # If the bus is between stops, the algorithm determines the nearest upstream and downstream stops.
        sorted_st = sorted(stops_list, key=lambda x: (x[4] if x[4] else 10**9))
        current_stop = best
        next_stop = None

        for i in range(len(sorted_st) - 1):
            A = sorted_st[i]
            B = sorted_st[i + 1]
            if A[4] is not None and B[4] is not None and seq is not None and A[4] <= seq <= B[4]:
                current_stop = A
                next_stop = B
                break

        if next_stop is None:
            return (
                current_stop[0],
                current_stop[1],
                current_stop[4],
                None,
                None,
                float(best_dist)
            )

        # This computes the distance to the next stop along the route.
        try:
            dist2 = geodesic((lat, lon), (next_stop[2], next_stop[3])).meters
        except:
            dist2 = None

        return (
            current_stop[0],
            current_stop[1],
            current_stop[4],
            next_stop[0],
            next_stop[1],
            float(dist2) if dist2 else None
        )

    return map_position

# This function downloads real-time GPS data for all lines, filters Line 11,
# normalizes fields, casts types, and adds a snapshot timestamp.
# The result is written as a single CSV file for archival and debugging.
def download_line11_snapshot(spark):
    resp = requests.get(REALTIME_URL, timeout=15)
    resp.raise_for_status()
    text = resp.text

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        schema = StructType([StructField("codLinea", StringType(), True)])
        return spark.createDataFrame([], schema=schema)

    df = spark.createDataFrame(rows)

    df = df.withColumn("codLinea", F.regexp_replace(F.col("codLinea"), "\\.0$", ""))

    df11 = df.filter(F.col("codLinea") == "11")

    df11 = df11.withColumn("lat", F.col("lat").cast("double"))
    df11 = df11.withColumn("lon", F.col("lon").cast("double"))

    df11 = df11.withColumn(
        "last_update",
        F.to_timestamp("last_update", "yyyy-MM-dd HH:mm:ss")
    )

    now = datetime.now()
    df11 = df11.withColumn("snapshot_time", F.lit(now))

    df11 = df11.select(
        "codBus",
        "codLinea",
        "sentido",
        "lon",
        "lat",
        "codParIni",
        "last_update",
        "snapshot_time"
    )

    filename = now.strftime("linea11_%Y%m%d_%H%M%S.csv")
    path = os.path.join(DATA_DIR, filename)
    write_single_csv(df11, path)

    print(f"[{now.strftime('%H:%M:%S')}] Saved snapshot {filename} ({df11.count()} rows)")

    return df11

# This function builds the live merged status for Line 11 by combining
# real-time GPS data with static GTFS scheduling.
# The map-matching UDF is applied to compute proximity and stop sequencing.
# A windowing operation ensures that only the closest upcoming stop is retained.
def build_live_status(spark, df_realtime, df_sched, stops_by_dir):
    map_position = build_position_udf(spark, stops_by_dir)

    df_live = df_realtime.withColumn(
        "pos_info",
        map_position(F.col("lat"), F.col("lon"), F.col("sentido"))
    )

    df_live = df_live.select(
        "codBus",
        F.col("last_update").alias("timestamp"),
        "lat",
        "lon",
        F.col("pos_info.current_stop_id").alias("current_stop_id"),
        F.col("pos_info.current_stop_name").alias("current_stop_name"),
        F.col("pos_info.stop_sequence").alias("stop_sequence"),
        F.col("pos_info.next_stop_id").alias("next_stop_id"),
        F.col("pos_info.next_stop_name").alias("next_stop_name"),
        F.col("pos_info.distance_m").alias("distance_m"),
        "sentido"
    )

    df_live = df_live.withColumn(
        "direction_id",
        F.when(F.col("sentido").cast("int") == 1, F.lit(0)).otherwise(F.lit(1))
    )

    df_sched_dir = df_sched.select("direction_id", "stop_id", "arrival_time_str").dropDuplicates()

    df_live = df_live.join(
        df_sched_dir,
        (df_live["next_stop_id"] == df_sched_dir["stop_id"]) &
        (df_live["direction_id"] == df_sched_dir["direction_id"]),
        "left"
    )

    df_live = df_live.drop(df_sched_dir["direction_id"]) \
                     .drop(df_sched_dir["stop_id"])

    df_live = df_live.withColumn("arrival_ts",
        F.to_timestamp(
            F.concat(
                F.date_format("timestamp", "yyyy-MM-dd "),
                F.col("arrival_time_str")
            ),
            "yyyy-MM-dd HH:mm:ss"
        )
    )

    df_live = df_live.withColumn(
        "next_scheduled_time",
        F.when(F.col("arrival_ts") < F.col("timestamp"),
               F.col("arrival_ts") + F.expr("INTERVAL 1 DAY"))
        .otherwise(F.col("arrival_ts"))
    )

    w = Window.partitionBy("codBus").orderBy(
        F.col("next_scheduled_time").asc_nulls_last()
    )
    df_live = df_live.withColumn("rn", F.row_number().over(w))
    df_live = df_live.filter("rn = 1").drop("rn")

    df_live = df_live.select(
        "codBus",
        "timestamp",
        "lat",
        "lon",
        "direction_id",
        "current_stop_id",
        "current_stop_name",
        "stop_sequence",
        "next_stop_id",
        "next_stop_name",
        "distance_m",
        "next_scheduled_time"
    )

    return df_live

# This loop continuously polls the real-time EMT endpoint,
# processes the new data through Spark, applies map matching, and writes the live status file.
# It runs indefinitely with a 60-second sleep interval between updates.
def live_loop():
    spark = create_spark()
    print("Loading GTFS for Line 11 (Spark)...")
    df_sched, stops_by_dir = load_gtfs_line11(spark)
    print("GTFS loaded.")

    while True:
        print(" New LIVE cycle ")
        realtime = download_line11_snapshot(spark)

        if realtime.rdd.isEmpty():
            print("No active buses for Line 11.")
        else:
            live_df = build_live_status(spark, realtime, df_sched, stops_by_dir)

            try:
                write_single_csv(live_df, "linea11_status_live.csv")
                print(f"Saved live status → linea11_status_live.csv (rows: {live_df.count()})")
            except Exception as e:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                alt = f"linea11_status_live_{ts}.csv"
                write_single_csv(live_df, alt)
                print(f"Error saving main file: saved → {alt}")

        time.sleep(60)

# The main entry point starts the continuous live update loop.
if __name__ == "__main__":
    live_loop()
