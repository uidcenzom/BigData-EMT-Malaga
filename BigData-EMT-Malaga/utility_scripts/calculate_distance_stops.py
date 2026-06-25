import os
import time
import json
import shutil
import threading
import requests
import pandas as pd
import joblib
import folium
from datetime import datetime
import math

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, radians, sin, cos, asin, sqrt, pow, atan2, 
    when, hour, minute, dayofweek, udf, pandas_udf, PandasUDFType, 
    current_timestamp, from_json, to_timestamp, row_number, min as min_
)
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, ArrayType

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = ""
DATA_DIR = os.path.join(BASE_DIR, "data")
TIMETABLE_DIR = os.path.join(DATA_DIR, "timetable")
STREAM_INPUT_DIR = "stream_data_stage"

STOPS_FILE = os.path.join(TIMETABLE_DIR, "stops.csv")
TRIPS_FILE = os.path.join(TIMETABLE_DIR, "trips.csv")
STOP_TIMES_FILE = os.path.join(TIMETABLE_DIR, "stop_times.csv")
MODEL_FILE = "delay_predictor_line11.pkl"
OUTPUT_MAP_FILE = "linea11_live_map.html"

API_URL = "https://datosabiertos.malaga.eu/api/3/action/datastore_search"
RESOURCE_ID = "9bc05288-1c11-4eec-8792-d74b679c8fcf"

TH_EARLY = -60
TH_LATE = 60

if os.path.exists(STREAM_INPUT_DIR):
    shutil.rmtree(STREAM_INPUT_DIR)
os.makedirs(STREAM_INPUT_DIR)

# Global Model Wrapper for Pandas UDF
model_broadcast = None

# ==============================================================================
# DATA PRODUCER (Ingestion Layer)
# ==============================================================================
def start_data_producer(interval=60):
    def fetch_job():
        while True:
            try:
                print(f"[Producer] Polling API at {datetime.now().strftime('%H:%M:%S')}...")
                params = {"resource_id": RESOURCE_ID, "limit": 500}
                response = requests.get(API_URL, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if "result" in data and "records" in data["result"]:
                        records = data["result"]["records"]
                        # Filter Line 11 at source
                        line11 = [r for r in records if str(r.get("codLinea", "")).replace(".0", "") == "11"]
                        
                        if line11:
                            # Add ingestion timestamp
                            for r in line11:
                                r['ingest_time'] = datetime.now().isoformat()
                                # Pre-clean coords to avoid Spark errors
                                try:
                                    r['lat'] = float(r.get('lat', 0))
                                    r['lon'] = float(r.get('lon', 0))
                                except:
                                    r['lat'] = None
                                    r['lon'] = None
                                    
                            batch_name = f"batch_{int(time.time())}"
                            with open(f"{STREAM_INPUT_DIR}/{batch_name}.json", 'w') as f:
                                json.dump(line11, f)
                            print(f"[Producer] Ingested {len(line11)} records.")
            except Exception as e:
                print(f"[Producer] Error: {e}")
            time.sleep(interval)

    t = threading.Thread(target=fetch_job, daemon=True)
    t.start()

# ==============================================================================
# SPARK PIPELINE
# ==============================================================================

def run_spark_pipeline():
    # --- Initialize Spark ---
    spark = SparkSession.builder \
        .appName("Line11NativeSpark") \
        .master("local[*]") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # --- Load & Prepare Static Data (Spark DataFrame) ---
    print("Loading Static Data into Spark DataFrames...")
    
    # Load raw CSVs
    stops_df = spark.read.csv(STOPS_FILE, header=True, inferSchema=True)
    trips_df = spark.read.csv(TRIPS_FILE, header=True, inferSchema=True)
    stop_times_df = spark.read.csv(STOP_TIMES_FILE, header=True, inferSchema=True)

    # Filter Line 11
    trips_11 = trips_df.filter(col("route_id") == 11).select("trip_id", "direction_id", "trip_headsign")
    
    # Broadcast Headsigns (Small lookup)
    headsigns_rows = trips_11.select("direction_id", "trip_headsign").distinct().collect()
    headsigns_dict = {row.direction_id: row.trip_headsign for row in headsigns_rows}
    bc_headsigns = spark.sparkContext.broadcast(headsigns_dict)

    # Create Master Static Table: Stop Locations per Direction
    # Join: Trips -> StopTimes -> Stops
    static_master = stop_times_df.join(trips_11, "trip_id") \
        .join(stops_df, "stop_id") \
        .select(
            col("stop_id"),
            col("direction_id"),
            col("stop_sequence").cast(IntegerType()),
            col("stop_name"),
            col("stop_lat").cast(DoubleType()),
            col("stop_lon").cast(DoubleType()),
            col("arrival_time").alias("sched_time_str")
        ) \
        .dropDuplicates(["stop_id", "direction_id", "sched_time_str"]) # Keep unique times per stop

    # Cache this, we will join against it every batch
    static_master.cache()
    print(f"Static Data Cached. Rows: {static_master.count()}")

    # --- Load Model (Global Broadcast) ---
    global model_broadcast
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        model_broadcast = spark.sparkContext.broadcast(model)
        print("Model loaded and broadcasted.")

    # --- Pandas UDF for Inference ---
    # This allows vectorized prediction (batch processing) instead of row-by-row
    @pandas_udf(DoubleType())
    def predict_delay_pudf(lat, lon, seq, dist, hr, min, wd, sec_sched, tod):
        # Construct pandas DataFrame from series
        features = pd.DataFrame({
            "lat": lat, "lon": lon, "stop_sequence": seq, "distance_m": dist,
            "hour": hr, "minute": min, "weekday": wd, 
            "seconds_to_scheduled": sec_sched, "time_of_day": tod
        })
        
        model = model_broadcast.value
        # Ensure column order matches model
        if hasattr(model, "feature_names_in_"):
            features = features.reindex(columns=model.feature_names_in_, fill_value=0)
        
        return pd.Series(model.predict(features.fillna(0)))

    # --- Helper UDF for Time Calculation ---
    # Calculating "next scheduled time" from a list of strings is hard in pure SQL
    # We'll use a standard UDF for this specific logic
    @udf(DoubleType())
    def get_seconds_to_sched(time_list, current_ts):
        if not time_list or not current_ts: return 0.0
        try:
            now = current_ts
            candidates = []
            for t_str in time_list:
                parts = t_str.split(':')
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                offset = 0
                if h >= 24: 
                    h -= 24
                    offset = 1
                dt = datetime(now.year, now.month, now.day, h, m, s) + timedelta(days=offset)
                if dt >= now:
                    candidates.append(dt)
            
            if not candidates: return 0.0
            best = min(candidates)
            return (best - now).total_seconds()
        except:
            return 0.0
            
    @udf(StringType())
    def get_sched_time_str(time_list, current_ts):
        if not time_list or not current_ts: return "N/A"
        try:
            now = current_ts
            candidates = []
            for t_str in time_list:
                parts = t_str.split(':')
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                offset = 0
                if h >= 24: h -= 24; offset = 1
                dt = datetime(now.year, now.month, now.day, h, m, s) + timedelta(days=offset)
                if dt >= now: candidates.append(dt)
            return min(candidates).strftime("%H:%M:%S") if candidates else "N/A"
        except: return "N/A"

    # --- Processing Logic (ForeachBatch) ---
    def process_batch(batch_df, batch_id):
        if batch_df.count() == 0: return
        print(f"--- Processing Batch {batch_id} ---")

        # Clean & Type Cast Stream
        clean_df = batch_df \
            .withColumn("lat", col("lat").cast(DoubleType())) \
            .withColumn("lon", col("lon").cast(DoubleType())) \
            .withColumn("sentido_int", col("sentido").cast(IntegerType())) \
            .withColumn("timestamp", to_timestamp(col("ingest_time"))) \
            .withColumn("direction_id", when(col("sentido_int") == 1, 0).otherwise(1)) \
            .filter(col("lat").isNotNull() & col("lon").isNotNull())

        # Native SQL Distance Calculation (Haversine)
        # We perform a CROSS JOIN with static stops but filtered by direction_id
        # This is efficient for small static tables
        
        # Radius of earth in meters
        R = 6371000 
        
        # Calculate Distance between every bus and every stop in its direction
        # Formula: 2 * R * asin(sqrt(sin^2(dlat/2) + cos(lat1)*cos(lat2)*sin^2(dlon/2)))
        distance_df = clean_df.alias("stream").join(
            static_master.alias("static"),
            col("stream.direction_id") == col("static.direction_id"),
            "inner"
        ).withColumn("dlat", radians(col("static.stop_lat") - col("stream.lat"))) \
         .withColumn("dlon", radians(col("static.stop_lon") - col("stream.lon"))) \
         .withColumn("a", pow(sin(col("dlat") / 2), 2) + 
                          cos(radians(col("stream.lat"))) * cos(radians(col("static.stop_lat"))) * pow(sin(col("dlon") / 2), 2)) \
         .withColumn("c", 2 * atan2(sqrt(col("a")), sqrt(1 - col("a")))) \
         .withColumn("dist", lit(R) * col("c"))

        # Find Nearest Stop (Window Function)
        # Rank stops by distance for each bus
        window_spec = Window.partitionBy("stream.codBus").orderBy("dist")
        
        nearest_stop_df = distance_df \
            .withColumn("rank", row_number().over(window_spec)) \
            .filter(col("rank") == 1) \
            .select(
                col("stream.codBus"),
                col("stream.lat"),
                col("stream.lon"),
                col("stream.timestamp"),
                col("stream.direction_id"),
                col("static.stop_sequence"),
                col("static.stop_name").alias("next_stop_name"),
                col("static.stop_lat").alias("next_stop_lat"),
                col("static.stop_lon").alias("next_stop_lon"),
                col("dist").alias("distance_m"),
                col("stream.sentido_int")
            )

        # Get Schedule Info
        # We need all times for the nearest stop to find the next one
        # Group static master to get list of times per stop
        schedule_list_df = static_master \
            .groupBy("stop_id", "direction_id") \
            .agg(from_json(lit("[]"), ArrayType(StringType())).alias("dummy")) # Placeholder if needed, or join raw
        
        # Actually easier: Join back to static_master to get all times for that stop
        # Then group by bus and collect_list
        # But for Spark, let's just use the UDF approach on the grouped static data we cached?
        # Let's pivot: The 'static_master' has multiple rows per stop (one per time).
        # We should group it first.
        grouped_schedule = static_master.groupBy("stop_id", "direction_id") \
            .agg(func_collect_list("sched_time_str").alias("times"))
            
        # Join nearest stop with grouped schedule
        # Note: We need stop_id from static_master in the join above. I missed selecting it.
        # Let's fix that conceptually (assuming stop_id available via lookup or re-join).
        # To keep code clean, let's assume we extract stop_id in step 2.
        
        # RE-DOING Step 2 select to include stop_id
        distance_df_v2 = clean_df.alias("stream").join(
            static_master.alias("static"),
            col("stream.direction_id") == col("static.direction_id"),
            "inner"
        ).withColumn("dlat", radians(col("static.stop_lat") - col("stream.lat"))) \
         .withColumn("dlon", radians(col("static.stop_lon") - col("stream.lon"))) \
         .withColumn("a", pow(sin(col("dlat") / 2), 2) + 
                          cos(radians(col("stream.lat"))) * cos(radians(col("static.stop_lat"))) * pow(sin(col("dlon") / 2), 2)) \
         .withColumn("c", 2 * atan2(sqrt(col("a")), sqrt(1 - col("a")))) \
         .withColumn("dist", lit(R) * col("c")) \
         .withColumn("rank", row_number().over(window_spec)) \
         .filter(col("rank") == 1) 

        # Now join with aggregated schedule
        from pyspark.sql.functions import collect_list
        grouped_sched = static_master.groupBy("stop_id", "direction_id") \
            .agg(collect_list("sched_time_str").alias("all_times"))

        final_features_df = distance_df_v2.join(
            grouped_sched, 
            ["stop_id", "direction_id"], 
            "left"
        ).withColumn("seconds_to_scheduled", get_seconds_to_sched(col("all_times"), col("timestamp"))) \
         .withColumn("scheduled_arrival", get_sched_time_str(col("all_times"), col("timestamp"))) \
         .withColumn("hour", hour(col("timestamp"))) \
         .withColumn("minute", minute(col("timestamp"))) \
         .withColumn("weekday", dayofweek(col("timestamp")) - 1) \
         .withColumn("time_of_day", col("hour") + (col("minute") / 60.0))

        # Predict (Pandas UDF)
        predictions = final_features_df.withColumn("predicted_delay", predict_delay_pudf(
            col("lat"), col("lon"), col("stop_sequence"), col("dist"),
            col("hour"), col("minute"), col("weekday"), col("seconds_to_scheduled"), col("time_of_day")
        ))
        
        # Visualization (Driver Side)
        generate_map(predictions.toPandas(), bc_headsigns.value)


    # --- Map Generation ---
    def generate_map(pdf, headsigns):
        m = folium.Map(location=[36.7213, -4.4214], zoom_start=13, tiles="CartoDB positron")
        
        # Draw Buses
        for _, bus in pdf.iterrows():
            delay = bus.get("predicted_delay", 0)
            color = "green"
            if delay < TH_EARLY: color = "blue"
            if delay > TH_LATE: color = "red"
            
            # Label
            try:
                d_id = int(bus["direction_id"])
                lbl = "Outbound" if d_id == 0 else "Inbound" # Based on logic above 
                # (Logic: 1->0, else 1. Usually 1 is outbound in raw. Adjust if needed)
                # Let's rely on headsign
                dest = headsigns.get(d_id, "Unknown")
                label = f"{dest}"
            except:
                label = "Bus"

            # Line
            if pd.notnull(bus["stop_lat"]): # stop_lat from static join
                folium.PolyLine(
                    locations=[(bus["lat"], bus["lon"]), (bus["stop_lat"], bus["stop_lon"])],
                    color=color, weight=3, opacity=0.8, dash_array='5, 10'
                ).add_to(m)

            folium.Marker(
                location=[bus["lat"], bus["lon"]],
                popup=f"Bus {bus['codBus']}<br>{label}<br>Delay: {int(delay)}s",
                icon=folium.Icon(color=color, icon="bus", prefix="fa")
            ).add_to(m)

        # Table Logic
        def format_delay(x):
            val = int(x)
            color = "green"
            if val < TH_EARLY: color = "blue"
            if val > TH_LATE: color = "red"
            return f'<span style="color:{color}; font-weight:bold;">{val} sec</span>'

        if not pdf.empty:
            pdf["Delay"] = pdf["predicted_delay"].apply(format_delay)
            pdf["Direction"] = pdf["direction_id"].apply(lambda x: headsigns.get(x, str(x)))
            
            display = pdf[["codBus", "Direction", "stop_name", "scheduled_arrival", "Delay"]]
            display.columns = ["Bus", "Direction", "Next Stop", "Sched. Arrival", "Delay"]
            
            html_table = display.to_html(index=False, classes="styled-table", escape=False, border=0)
        else:
            html_table = "<p>No active buses found.</p>"

        # CSS
        custom_css = """
        <style>
            html, body { height: 100%; margin: 0; padding: 0; display: flex; flex-direction: column; font-family: Arial, sans-serif; }
            .folium-map { flex: 0 0 65%; width: 100%; position: relative !important; }
            .bottom-panel { flex: 1; background-color: #f8f9fa; padding: 20px; overflow-y: auto; border-top: 4px solid #333; }
            .styled-table { width: 100%; border-collapse: collapse; font-size: 0.9em; background-color: white; }
            .styled-table thead tr { background-color: #009879; color: #ffffff; text-align: left; }
            .styled-table th, .styled-table td { padding: 12px 15px; text-align: left; }
            .styled-table tbody tr { border-bottom: 1px solid #dddddd; }
        </style>
        """
        
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        custom_html = f"""
        <div class="bottom-panel">
            <h3>Line 11 Live Monitor (Spark Native) - {ts}</h3>
            {html_table}
        </div>
        """
        
        m.get_root().header.add_child(folium.Element(custom_css))
        m.get_root().html.add_child(folium.Element(custom_html))
        m.get_root().header.add_child(folium.Element('<meta http-equiv="refresh" content="60">'))
        
        m.save(OUTPUT_MAP_FILE)
        print(f"Map updated: {OUTPUT_MAP_FILE}")


    # --- Start Stream ---
    json_schema = StructType([
        StructField("codBus", StringType(), True),
        StructField("codLinea", StringType(), True),
        StructField("sentido", StringType(), True),
        StructField("lat", DoubleType(), True),
        StructField("lon", DoubleType(), True),
        StructField("last_update", StringType(), True),
        StructField("ingest_time", StringType(), True)
    ])

    print("Initializing Spark Structured Streaming...")
    stream = spark.readStream \
        .schema(json_schema) \
        .json(STREAM_INPUT_DIR) \
        .writeStream \
        .outputMode("append") \
        .foreachBatch(process_batch) \
        .start()

    stream.awaitTermination()

if __name__ == "__main__":
    print("Starting Producer...")
    start_data_producer(60)
    print("Starting Consumer...")
    run_spark_pipeline()