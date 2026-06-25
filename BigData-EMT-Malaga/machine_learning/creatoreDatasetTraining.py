# This script constructs the full training dataset for predicting bus arrival delays on EMT Málaga Line 11.
# It integrates static GTFS information with historical GPS snapshots, applies a fast map-matching algorithm
# using Spark transformations, detects actual arrival times, computes true delays, and finally expands each arrival
# into a supervised learning window of observations collected in the minutes preceding the stop arrival.

import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, TimestampType
)
from pyspark.sql.window import Window

# These paths define the location where real-time snapshots were saved and where GTFS static data is stored.
# The configuration also sets thresholds for detecting arrival events and for defining the historical window
# used to generate supervised training samples.
DATA_DIR = r"C:\Users\utente\Desktop\bigDataProject\dati_linea11"
GTFS_DIR = r"C:\Users\utente\Desktop\bigDataFinal\realData"

STOPS = os.path.join(GTFS_DIR, "stops.txt")
TRIPS = os.path.join(GTFS_DIR, "trips.txt")
STOP_TIMES = os.path.join(GTFS_DIR, "stop_times.txt")

ARRIVAL_DISTANCE_THRESHOLD = 150  # meters
WINDOW_MINUTES = 5               # minutes before arrival for supervised learning

# The Spark session is created with local parallelism enabled.
# Logging is reduced to avoid excessive output during large transformations.
def create_spark():
    spark = (
        SparkSession.builder
        .appName("CreatorDatasetTrainingSpark")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

# Spark writes distributed CSV outputs in multiple part files.
# This helper consolidates all partitions into a single CSV to facilitate external use.
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

# This function loads GTFS static data for Line 11.
# It filters trips to retain only route_id = 11 and joins them with stop times and stop metadata.
# Arrival times are normalized and validated to remove corrupted entries.
def load_gtfs_line11(spark):
    df_stops = spark.read.csv(STOPS, header=True, inferSchema=True)
    df_trips = spark.read.csv(TRIPS, header=True, inferSchema=True)
    df_times = spark.read.csv(STOP_TIMES, header=True, inferSchema=True)

    df_trips11 = df_trips.filter(F.col("route_id") == 11)
    df_times11 = df_times.join(df_trips11.select("trip_id"), "trip_id", "inner")

    df_sched = (
        df_times11.join(df_stops, "stop_id", "left")
        .join(df_trips11.select("trip_id", "direction_id"), "trip_id", "left")
    )

    df_sched = df_sched.withColumn("stop_sequence", F.col("stop_sequence").cast("int"))

    df_sched = df_sched.withColumn(
        "arrival_time_str",
        F.col("arrival_time").cast("string")
    )

    # Cleaning malformed time entries by extracting hour and ensuring it falls in the valid interval.
    df_sched = df_sched.withColumn(
        "hour_int",
        F.split("arrival_time_str", ":").getItem(0).cast("int")
    ).filter("hour_int >= 0 AND hour_int < 24")

    return df_sched

# All previously saved GPS snapshots for Line 11 are loaded.
# Timestamps and coordinates are cast to appropriate types and invalid rows are removed.
def load_all_snapshots(spark):
    files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.startswith("linea11_") and f.endswith(".csv")
    ]

    if not files:
        return None

    df = spark.read.csv(files, header=True, inferSchema=True)

    df = df.withColumn("timestamp", F.to_timestamp("last_update"))

    df = df.withColumn("lat", F.col("lat").cast("double"))
    df = df.withColumn("lon", F.col("lon").cast("double"))

    df = df.filter("codBus IS NOT NULL AND timestamp IS NOT NULL")

    return df.select("codBus", "timestamp", "lat", "lon")

# This function performs a fast map-matching step using Spark itself.
# A broadcast join distributes the GTFS stops efficiently, enabling a cross join between GPS points and all stops.
# A Haversine distance computation identifies the closest stop for each GPS record.
def match_snapshot_to_stop(spark, df_gps, df_sched):

    stops = df_sched.select(
        "stop_id",
        "stop_name",
        "stop_lat",
        "stop_lon",
        "stop_sequence"
    ).dropDuplicates()

    stops_b = F.broadcast(stops)

    # Haversine formula implemented using Spark SQL functions.
    def haversine(lat1, lon1, lat2, lon2):
        return (
            6371000 * 2 * F.asin(
                F.sqrt(
                    F.pow(F.sin(F.radians(lat2 - lat1)/2), 2) +
                    F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) *
                    F.pow(F.sin(F.radians(lon2 - lon1)/2), 2)
                )
            )
        )

    # The crossJoin generates candidate stop associations for each GPS point.
    # The following window operation selects the nearest stop per timestamp and vehicle.
    df_join = df_gps.crossJoin(stops_b)

    df_join = df_join.withColumn(
        "distance_m",
        haversine(
            F.col("lat"),
            F.col("lon"),
            F.col("stop_lat"),
            F.col("stop_lon")
        )
    )

    w = Window.partitionBy("codBus", "timestamp").orderBy("distance_m")

    df_ranked = df_join.withColumn("rn", F.row_number().over(w))

    df_best = df_ranked.filter("rn = 1").drop("rn")

    return df_best.select(
        "codBus",
        "timestamp",
        "lat",
        "lon",
        "stop_id",
        "stop_name",
        "stop_sequence",
        "distance_m"
    )

# This step identifies actual arrival events by selecting GPS points whose distance from a stop
# falls below a predefined threshold. The earliest timestamp is used as the actual arrival time.
def detect_actual_arrivals(df_matched):
    df_close = df_matched.filter(F.col("distance_m") < ARRIVAL_DISTANCE_THRESHOLD)

    return df_close.groupBy("codBus", "stop_id").agg(
        F.min("timestamp").alias("actual_time")
    )

# This function computes true arrival delays by comparing actual arrival timestamps
# with the GTFS-scheduled arrival times. Spark windowing is used to select the best temporal match.
def compute_true_delay(df_arrivals, df_sched):
    df_join = df_arrivals.alias("a").join(
        df_sched.select("stop_id", "arrival_time_str").alias("s"),
        "stop_id",
        "inner"
    )

    df_join = df_join.withColumn(
        "scheduled_time",
        F.to_timestamp(
            F.concat(
                F.date_format("a.actual_time", "yyyy-MM-dd "),
                F.col("s.arrival_time_str")
            ),
            "yyyy-MM-dd HH:mm:ss"
        )
    )

    df_join = df_join.withColumn(
        "diff",
        F.abs(F.unix_timestamp("scheduled_time") - F.unix_timestamp("a.actual_time"))
    )

    w = Window.partitionBy("codBus", "stop_id", "actual_time").orderBy("diff")

    df_best = df_join.withColumn("rn", F.row_number().over(w)).filter("rn = 1")

    return df_best.select(
        "codBus",
        "stop_id",
        "scheduled_time",
        "actual_time",
        (F.unix_timestamp("actual_time") - F.unix_timestamp("scheduled_time")).alias("true_delay_seconds")
    )

# This function constructs the supervised learning dataset.
# For each detected arrival event, all GPS points from the preceding time window are paired with the delay label.
# This produces the training structure needed for ML models predicting delays before arrival.
def expand_training_dataset(df_matched, df_delays):

    df_join = df_matched.alias("m").join(
        df_delays.alias("d"),
        F.col("m.codBus") == F.col("d.codBus"),
        "inner"
    )

    df_join = df_join.filter(
        (F.col("m.timestamp") >= F.col("d.actual_time") - F.expr(f"INTERVAL {WINDOW_MINUTES} MINUTES")) &
        (F.col("m.timestamp") < F.col("d.actual_time"))
    )

    df_train = df_join.select(
        F.col("m.timestamp").alias("timestamp"),
        F.col("m.lat").alias("lat"),
        F.col("m.lon").alias("lon"),
        F.col("d.stop_id").alias("stop_id"),
        F.col("m.stop_sequence").alias("stop_sequence"),
        F.col("m.distance_m").alias("distance_m"),
        F.col("d.scheduled_time").alias("scheduled_time"),
        F.col("d.actual_time").alias("actual_time"),
        (
            F.unix_timestamp(F.col("d.scheduled_time")) -
            F.unix_timestamp(F.col("m.timestamp"))
        ).alias("seconds_to_scheduled"),
        F.col("d.true_delay_seconds").alias("true_delay_seconds")
    )

    return df_train

# The main pipeline loads GTFS data, loads historical GPS snapshots,
# performs map matching, detects actual arrivals, computes true delays,
# expands the training dataset, and finally writes the dataset to disk.
if __name__ == "__main__":
    spark = create_spark()

    print("Loading GTFS...")
    df_sched = load_gtfs_line11(spark)

    print("Loading snapshots...")
    df_hist = load_all_snapshots(spark)
    if df_hist is None:
        print("No snapshots found. Exiting.")
        spark.stop()
        exit(0)

    print("Map-matching snapshots…")
    df_matched = match_snapshot_to_stop(spark, df_hist, df_sched)

    print("Detecting arrivals…")
    df_arrivals = detect_actual_arrivals(df_matched)
    print("Arrivals detected:", df_arrivals.count())

    print("Computing TRUE delays…")
    df_delays = compute_true_delay(df_arrivals, df_sched)

    print("Expanding training dataset…")
    df_train = expand_training_dataset(df_matched, df_delays)

    write_single_csv(df_train, "training_dataset_line11.csv")
    print("Saved:", df_train.count(), "rows")

    spark.stop()
