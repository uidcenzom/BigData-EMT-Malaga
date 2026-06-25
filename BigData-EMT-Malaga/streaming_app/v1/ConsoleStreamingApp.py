import os
import time
import json
import shutil
import threading
import requests
import pandas as pd
import joblib
from datetime import datetime, timedelta
from geopy.distance import geodesic

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, from_json, current_timestamp, lit, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = ""
DATA_DIR = os.path.join(BASE_DIR, "data")
TIMETABLE_DIR = os.path.join(DATA_DIR, "timetable")
STREAM_INPUT_DIR = "stream_data_stage"

# Paths
STOPS_FILE = os.path.join(TIMETABLE_DIR, "stops.csv")
TRIPS_FILE = os.path.join(TIMETABLE_DIR, "trips.csv")
STOP_TIMES_FILE = os.path.join(TIMETABLE_DIR, "stop_times.csv")
MODEL_FILE = "models/delay_predictor_line11.pkl"

# API
API_URL = "https://datosabiertos.malaga.eu/api/3/action/datastore_search"
RESOURCE_ID = "9bc05288-1c11-4eec-8792-d74b679c8fcf"

# Clean up staging directory on start
if os.path.exists(STREAM_INPUT_DIR):
    shutil.rmtree(STREAM_INPUT_DIR)
os.makedirs(STREAM_INPUT_DIR)

# ==============================================================================
# 1. STATIC DATA MANAGER (GTFS)
# ==============================================================================
class GTFSLoader:
    def __init__(self):
        print("Loading GTFS data for Spark Broadcast...")
        self.stops = pd.read_csv(STOPS_FILE)
        self.trips = pd.read_csv(TRIPS_FILE)
        self.stop_times = pd.read_csv(STOP_TIMES_FILE)
        
        # Filter Line 11
        self.trips_11 = self.trips[self.trips["route_id"] == 11]
        self.stop_times_11 = self.stop_times[self.stop_times["trip_id"].isin(self.trips_11["trip_id"])]
        
        # Merge
        self.schedule = self.stop_times_11.merge(self.stops, on="stop_id", how="left")
        self.schedule = self.schedule.merge(self.trips_11[["trip_id", "direction_id"]], on="trip_id", how="left")
        
        # Types
        self.schedule["stop_sequence"] = pd.to_numeric(self.schedule["stop_sequence"], errors="coerce")
        self.schedule["arrival_time_str"] = self.schedule["arrival_time"].astype(str)
        
        # Headsigns lookup
        self.headsigns = {}
        for d_id in self.trips_11["direction_id"].unique():
            try:
                self.headsigns[d_id] = self.trips_11[self.trips_11["direction_id"] == d_id].iloc[0]["trip_headsign"]
            except:
                self.headsigns[d_id] = "Unknown"

        # Optimization: Convert schedule to a list of dictionaries for broadcasting
        self.stops_by_direction = {}
        for d in [0, 1]:
            subset = self.schedule[self.schedule["direction_id"] == d]
            unique_stops = subset[["stop_id", "stop_name", "stop_lat", "stop_lon", "stop_sequence"]].drop_duplicates().sort_values("stop_sequence")
            self.stops_by_direction[d] = unique_stops.to_dict('records')
            
        # Full schedule lookup (stop_id, direction_id) -> list of arrival times
        self.schedule_lookup = self.schedule.groupby(["stop_id", "direction_id"])["arrival_time_str"].apply(list).to_dict()

# ==============================================================================
# 2. PRODUCER: DATA FETCHER (Thread)
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
                        line11_records = [r for r in records if str(r.get("codLinea", "")).replace(".0", "") == "11"]
                        
                        if line11_records:
                            for r in line11_records:
                                r['ingest_time'] = datetime.now().isoformat()
                                
                            filename = f"{STREAM_INPUT_DIR}/batch_{int(time.time())}.json"
                            with open(filename, 'w') as f:
                                json.dump(line11_records, f)
                            print(f"[Producer] Wrote {len(line11_records)} records to {filename}")
            except Exception as e:
                print(f"[Producer] Error: {e}")
            
            time.sleep(interval)

    t = threading.Thread(target=fetch_job, daemon=True)
    t.start()
    return t

# ==============================================================================
# 3. CONSUMER: SPARK PIPELINE CLASS
# ==============================================================================
class Line11SparkPipeline:
    def __init__(self):
        self.spark = None
        self.bc_stops = None
        self.bc_schedule = None
        self.bc_headsigns = None
        self.bc_model = None
        self.gtfs_data = None

    def init_spark(self):
        """Initializes the Spark Session."""
        print("Initializing Spark...")
        self.spark = SparkSession.builder \
            .appName("Line11RealTimeDelay") \
            .config("spark.sql.shuffle.partitions", "2") \
            .master("local[*]") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")

    def load_and_broadcast_resources(self):
        """Loads GTFS data and Model, then broadcasts them."""
        self.gtfs_data = GTFSLoader()
        self.bc_stops = self.spark.sparkContext.broadcast(self.gtfs_data.stops_by_direction)
        self.bc_schedule = self.spark.sparkContext.broadcast(self.gtfs_data.schedule_lookup)
        self.bc_headsigns = self.spark.sparkContext.broadcast(self.gtfs_data.headsigns)
        
        if os.path.exists(MODEL_FILE):
            print(f"Loading Model {MODEL_FILE}...")
            model = joblib.load(MODEL_FILE)
            self.bc_model = self.spark.sparkContext.broadcast(model)
        else:
            print("ERROR: Model file not found. Pipeline will fail.")
            raise FileNotFoundError(MODEL_FILE)

    def _get_metadata_udf(self):
        """Defines and returns the Metadata UDF using the broadcast variables."""
        
        # Capture broadcast variables in closure
        bc_headsigns = self.bc_headsigns
        bc_stops = self.bc_stops
        bc_schedule = self.bc_schedule

        meta_schema = StructType([
            StructField("stop_sequence", IntegerType()),
            StructField("distance_m", DoubleType()),
            StructField("next_stop_name", StringType()),
            StructField("next_stop_lat", DoubleType()),
            StructField("next_stop_lon", DoubleType()),
            StructField("scheduled_arrival", StringType()),
            StructField("seconds_to_scheduled", DoubleType()),
            StructField("direction_label", StringType()),
            StructField("hour", IntegerType()),
            StructField("minute", IntegerType()),
            StructField("time_of_day", DoubleType()),
            StructField("weekday", IntegerType()),
            StructField("direction_id", IntegerType())
        ])

        def get_bus_metadata(lat, lon, sentido, ingest_time_str):
            try:
                try:
                    lat = float(lat)
                    lon = float(lon)
                except: return None
                    
                now = datetime.fromisoformat(ingest_time_str) if ingest_time_str else datetime.now()
                
                # Direction Logic
                try:
                    s_int = int(sentido)
                    direction_id = 0 if s_int == 1 else 1
                    type_lbl = "Outbound" if s_int == 1 else "Inbound"
                except:
                    direction_id = 0
                    type_lbl = "Unknown"
                
                headsign = bc_headsigns.value.get(direction_id, "Unknown")
                direction_label = f"{headsign} ({type_lbl})"
                
                # Find Nearest Stop
                stops = bc_stops.value.get(direction_id, [])
                if not stops: return None
                    
                # Manual Geodesic Min
                pos = (lat, lon)
                best_stop = None
                min_dist = float('inf')
                
                for s in stops:
                    dist = geodesic(pos, (s['stop_lat'], s['stop_lon'])).meters
                    if dist < min_dist:
                        min_dist = dist
                        best_stop = s
                
                if not best_stop: return None
                    
                # Schedule Lookup
                stop_id = best_stop['stop_id']
                times = bc_schedule.value.get((stop_id, direction_id), [])
                
                next_sched_time = None
                candidates = []
                for t_str in times:
                    try:
                        h, m, s = map(int, t_str.split(':'))
                        day_offset = 0
                        if h >= 24: h -= 24; day_offset = 1
                        
                        dt = datetime(now.year, now.month, now.day, h, m, s) + timedelta(days=day_offset)
                        if dt >= now: candidates.append(dt)
                    except: continue
                
                seconds_to_sched = 0.0
                sched_str = "N/A"
                if candidates:
                    next_sched_time = min(candidates)
                    seconds_to_sched = (next_sched_time - now).total_seconds()
                    sched_str = next_sched_time.strftime("%H:%M:%S")
                    
                # Time features
                hour = now.hour
                minute = now.minute
                tod = hour + (minute / 60.0)
                wd = now.weekday()
                
                return (
                    int(best_stop['stop_sequence']), float(min_dist), best_stop['stop_name'],
                    best_stop['stop_lat'], best_stop['stop_lon'], sched_str,
                    seconds_to_sched, direction_label, hour, minute, tod, wd, direction_id
                )
            except: return None

        return udf(get_bus_metadata, meta_schema)

    def _get_prediction_udf(self):
        """Defines and returns the Prediction UDF."""
        bc_model = self.bc_model

        @udf(DoubleType())
        def predict_delay_udf(lat, lon, seq, dist, hr, min, wd, sec_sched, tod):
            try:
                import pandas as pd
                
                data = {
                    "lat": [lat], "lon": [lon], "stop_sequence": [seq], "distance_m": [dist], 
                    "hour": [hr], "minute": [min], "weekday": [wd], 
                    "seconds_to_scheduled": [sec_sched], "time_of_day": [tod]
                }
                
                features = pd.DataFrame(data)
                features = features.fillna(0)
                
                model = bc_model.value
                
                if hasattr(model, "feature_names_in_"):
                    features = features.reindex(columns=model.feature_names_in_, fill_value=0)
                
                pred = model.predict(features)
                return float(pred[0])
            except Exception as e:
                print(f"PREDICTION ERROR: {e}")
                return 0.0
        
        return predict_delay_udf

    def update_batch(self, batch_df, batch_id):
        """ForeachBatch callback method."""
        print(f"Processing Batch {batch_id} with {batch_df.count()} buses...")
        if batch_df.count() == 0: return
        
        batch_df.select("codBus", "predicted_delay", "direction_label", "seconds_to_scheduled").show(truncate=False)

    def run(self):
        """Main execution method."""
        self.init_spark()
        self.load_and_broadcast_resources()
        
        udf_metadata = self._get_metadata_udf()
        predict_delay_udf = self._get_prediction_udf()
        
        json_schema = StructType([
            StructField("codBus", StringType(), True),
            StructField("codLinea", StringType(), True),
            StructField("sentido", StringType(), True),
            StructField("lat", StringType(), True),
            StructField("lon", StringType(), True),
            StructField("last_update", StringType(), True),
            StructField("ingest_time", StringType(), True)
        ])

        print("Initializing Spark Stream...")
        raw_stream = self.spark.readStream \
            .schema(json_schema) \
            .json(STREAM_INPUT_DIR)

        processed = raw_stream \
            .withColumn("meta", udf_metadata(col("lat"), col("lon"), col("sentido"), col("ingest_time"))) \
            .filter(col("meta").isNotNull()) \
            .select(
                col("codBus"),
                col("lat").cast(DoubleType()),
                col("lon").cast(DoubleType()),
                col("meta.stop_sequence").alias("stop_sequence"),
                col("meta.distance_m").alias("distance_m"),
                col("meta.next_stop_name").alias("next_stop_name"),
                col("meta.next_stop_lat").alias("next_stop_lat"),
                col("meta.next_stop_lon").alias("next_stop_lon"),
                col("meta.scheduled_arrival").alias("scheduled_arrival"),
                col("meta.seconds_to_scheduled").alias("seconds_to_scheduled"),
                col("meta.direction_label").alias("direction_label"),
                col("meta.hour").alias("hour"),
                col("meta.minute").alias("minute"),
                col("meta.time_of_day").alias("time_of_day"),
                col("meta.weekday").alias("weekday"),
            )

        predictions = processed.withColumn("predicted_delay", predict_delay_udf(
            col("lat"), col("lon"), col("stop_sequence"), col("distance_m"),
            col("hour"), col("minute"), col("weekday"), col("seconds_to_scheduled"), col("time_of_day")
        ))

        query = predictions.writeStream \
            .outputMode("append") \
            .foreachBatch(self.update_batch) \
            .start()

        query.awaitTermination()

if __name__ == "__main__":
    print("Starting Producer...")
    start_data_producer(60)
    
    print("Starting Spark Consumer...")
    pipeline = Line11SparkPipeline()
    pipeline.run()