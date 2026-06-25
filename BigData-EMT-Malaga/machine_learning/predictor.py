# This script loads a previously trained Spark ML pipeline for predicting bus arrival delays
# for EMT Málaga Line 11. It reads the latest live-status file, performs feature engineering
# consistent with the training workflow, applies the model to compute delay predictions,
# and exports the results as a singleCSV file for downstream analysis or visualization.

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel


# A SparkSession is initialized for local execution.
# Logging verbosity is reduced to avoid excessive output during inference.
def create_spark():
    spark = (
        SparkSession.builder
        .appName("DelayPredictorSpark")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# Spark writes CSV files as distributed partitioned outputs.
# This helper consolidates all partitions into a single CSV file for portability.
def write_single_csv(df, path):
    import os, glob, shutil
    tmp_dir = path + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(tmp_dir)
    part_file = glob.glob(os.path.join(tmp_dir, "part-*.csv"))[0]
    if os.path.exists(path):
        os.remove(path)
    shutil.move(part_file, path)
    shutil.rmtree(tmp_dir)


# Main inference workflow.
# This pipeline reconstructs all required features so that the model receives inputs
# in the exact same structure as used during the training phase.
if __name__ == "__main__":
    spark = create_spark()

    print("Loading live input file...")

    # The live-status file contains the most recent map-matched positions for active buses.
    # These entries include the projected next stop and scheduled arrival time.
    df = spark.read.csv(
        "linea11_status_live.csv",
        header=True,
        inferSchema=True
    )
    print("Rows loaded:", df.count())

    # Temporal fields are converted to timestamp format to enable correct extraction
    # of temporal components and computation of the seconds-to-scheduled feature.
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
    df = df.withColumn("next_scheduled_time", F.to_timestamp("next_scheduled_time"))

    # This feature measures the time difference between the bus GPS timestamp
    # and the GTFS-scheduled arrival time for the next stop.
    df = df.withColumn(
        "seconds_to_scheduled",
        F.unix_timestamp("next_scheduled_time") - F.unix_timestamp("timestamp")
    )

    # All required numerical fields are cast explicitly to guarantee consistency
    # with the feature vector definition used during model training.
    df = df.withColumn("lat", F.col("lat").cast("double"))
    df = df.withColumn("lon", F.col("lon").cast("double"))
    df = df.withColumn("stop_sequence", F.col("stop_sequence").cast("int"))
    df = df.withColumn("distance_m", F.col("distance_m").cast("double"))
    df = df.withColumn("seconds_to_scheduled", F.col("seconds_to_scheduled").cast("double"))

    # Temporal features are extracted exactly as during training.
    # This ensures strict feature parity between training and prediction pipelines.
    df = df.withColumn("hour", F.hour("timestamp"))
    df = df.withColumn("minute", F.minute("timestamp"))
    df = df.withColumn("weekday", F.dayofweek("timestamp") - 2)
    df = df.withColumn("time_of_day", F.col("hour") * 60 + F.col("minute"))

    print("Loading model delay_predictor_line11_spark ...")

    # The previously trained Spark ML pipeline is loaded from disk.
    # The pipeline internally contains both the VectorAssembler and the Random Forest model.
    model = PipelineModel.load("delay_predictor_line11_spark")

    print("Predicting delays...")

    # Predictions are computed by applying the pipeline’s transform method.
    # Spark automatically applies the same preprocessing steps followed during training.
    preds = model.transform(df)

    # Only relevant fields are selected for output.
    # The prediction column contains the estimated arrival delay in seconds.
    preds = preds.select(
        "codBus",
        "timestamp",
        "lat",
        "lon",
        "stop_sequence",
        "distance_m",
        "seconds_to_scheduled",
        F.col("prediction").alias("predicted_delay_seconds")
    )

    # Results are exported as a single CSV file for presentation or integration
    # into downstream operational systems.
    write_single_csv(preds, "delay_predictions.csv")

    print("Predictions saved → delay_predictions.csv")
    print("Done.")

    spark.stop()
