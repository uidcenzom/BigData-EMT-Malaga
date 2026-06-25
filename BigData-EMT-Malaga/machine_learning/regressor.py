# This script trains a Spark ML regression model for predicting the delay at arrival
# for EMT Málaga Line 11, using a dataset previously generated via map-matching and
# GTFS alignment. The workflow includes: data loading, timestamp normalization,
# feature engineering, cleaning, pipeline construction, model training,
# evaluation with multiple metrics, extraction of feature importance, and model persistence.

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline


# A SparkSession is created with local parallelism enabled.
# This serves as the execution engine for distributed feature transformations
# and for training the Random Forest model.
def create_spark():
    spark = (
        SparkSession.builder
        .appName("DelayRegressorSparkML")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# Spark writes CSV outputs using multiple part files by default.
# This helper utility consolidates the distributed output into a single CSV file,
# improving interoperability with external tools that expect a single-file dataset.
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


# The main training workflow begins here.
if __name__ == "__main__":
    spark = create_spark()

    print("Loading training dataset...")

    # The training dataset is loaded with schema inference enabled.
    # This file contains GPS points in the minutes preceding each arrival,
    # paired with the true arrival delay computed earlier from GTFS data.
    df = spark.read.csv(
        r"C:\Users\utente\Desktop\sparkProject\training_dataset_line11.csv",
        header=True,
        inferSchema=True
    )

    print("Rows loaded:", df.count())

    # Timestamps are explicitly converted to Spark TimestampType to ensure
    # correct extraction of hour, minute, and weekday features.
    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
    df = df.withColumn("scheduled_time", F.to_timestamp("scheduled_time"))
    df = df.withColumn("actual_time", F.to_timestamp("actual_time"))

    # Data cleaning removes noisy samples and unrealistic delays.
    # Distance filtering removes misaligned map-matching artifacts,
    # and delay clipping removes extreme outliers.
    df = df.filter(F.col("distance_m") < 500)
    df = df.filter(F.col("true_delay_seconds").isNotNull())
    df = df.filter(F.abs(F.col("true_delay_seconds")) < 900)

    # Temporal feature extraction converts timestamps into structured features
    # such as hour, minute, weekday, and absolute minute-of-day representation.
    df = df.withColumn("hour", F.hour("timestamp"))
    df = df.withColumn("minute", F.minute("timestamp"))
    df = df.withColumn("weekday", F.dayofweek("timestamp") - 2)
    df = df.withColumn("time_of_day", F.col("hour") * 60 + F.col("minute"))

    # All numerical fields are cast explicitly to the appropriate numeric types,
    # ensuring that the VectorAssembler receives consistent feature inputs.
    df = df.withColumn("lat", F.col("lat").cast("double"))
    df = df.withColumn("lon", F.col("lon").cast("double"))
    df = df.withColumn("stop_sequence", F.col("stop_sequence").cast("int"))
    df = df.withColumn("distance_m", F.col("distance_m").cast("double"))
    df = df.withColumn("seconds_to_scheduled", F.col("seconds_to_scheduled").cast("double"))
    df = df.withColumn("true_delay_seconds", F.col("true_delay_seconds").cast("double"))

    # The list of selected features reflects spatial context, temporal context,
    # and relative time until the scheduled arrival.
    feature_cols = [
        "lat",
        "lon",
        "stop_sequence",
        "distance_m",
        "hour",
        "minute",
        "weekday",
        "time_of_day",
        "seconds_to_scheduled"
    ]

    # The VectorAssembler aggregates numerical features into a single feature vector,
    # ensuring compatibility with Spark ML estimators.
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    # The RandomForestRegressor is selected for its robustness to nonlinear interactions
    # and for its ability to model complex spatial-temporal relationships in bus movement.
    # A relatively large number of trees is used to maximize predictive stability.
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="true_delay_seconds",
        predictionCol="prediction",
        numTrees=400,
        maxDepth=25,
        minInstancesPerNode=2,
        seed=42
    )

    # The Pipeline ensures full reproducibility by chaining feature assembly
    # and model training into a single, atomic workflow.
    pipeline = Pipeline(stages=[assembler, rf])

    # The dataset is split into training and testing partitions
    # using a fixed seed to guarantee determinism across runs.
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    print("Train size:", train.count(), " Test size:", test.count())

    # The model is trained on the training split.
    # Spark distributes the training workload automatically across available CPU cores.
    print("Training model...")
    model = pipeline.fit(train)

    # Predictions are computed on the test split
    # using the same preprocessing steps encapsulated inside the Pipeline.
    preds = model.transform(test)

    # Two standard regression metrics are computed: MAE and RMSE.
    evaluator_mae = RegressionEvaluator(
        labelCol="true_delay_seconds",
        predictionCol="prediction",
        metricName="mae"
    )

    evaluator_rmse = RegressionEvaluator(
        labelCol="true_delay_seconds",
        predictionCol="prediction",
        metricName="rmse"
    )

    mae = evaluator_mae.evaluate(preds)
    rmse = evaluator_rmse.evaluate(preds)

    # Mean Absolute Percentage Error (MAPE) is computed manually.
    preds = preds.withColumn(
        "ape",
        F.abs(F.col("true_delay_seconds") - F.col("prediction")) /
        (F.abs(F.col("true_delay_seconds")) + F.lit(1.0))
    )
    mape = preds.agg(F.mean("ape")).first()[0] * 100.0

    print(f"MAE : {mae:.2f} seconds")
    print(f"RMSE: {rmse:.2f} seconds")
    print(f"MAPE: {mape:.2f}%")

    # Feature importances extracted from the trained Random Forest
    # provide interpretability by ranking the relative contribution
    # of each input feature to the final prediction.
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances

    rows = []
    for name, score in zip(feature_cols, importances):
        rows.append((name, float(score)))

    fi_df = spark.createDataFrame(rows, ["feature", "importance"])
    fi_df = fi_df.orderBy(F.col("importance").desc())

    write_single_csv(fi_df, "feature_importance.csv")
    print("Feature importances saved → feature_importance.csv")

    # The trained model is persisted using Spark ML's native save mechanism,
    # allowing later loading for batch evaluation or real-time inference.
    model.write().overwrite().save("delay_predictor_line11_spark")
    print("Model saved → delay_predictor_line11_spark")

    spark.stop()
