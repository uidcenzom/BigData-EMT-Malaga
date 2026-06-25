"""
StatefulProcessor.py - Core Stateful Stream Processing for Bus Delay Prediction

This module implements the main processing logic for real-time bus delay prediction
using Spark Structured Streaming with stateful operations. It handles:
    - Per-bus state management (position, speed, predictions)
    - Physics-based delay prediction (heuristic model)
    - ML model inference via Spark MLlib
    - Arrival detection and ground truth calculation for model evaluation

The processor uses Spark's applyInPandasWithState for maintaining bus state across
micro-batches, enabling accurate speed estimation and prediction tracking.
"""

import os
import time
import pandas as pd
from datetime import datetime, timedelta
from geopy.distance import geodesic
from typing import Tuple, Iterator, Dict, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    StructType, StructField, DoubleType, StringType, 
    IntegerType, ArrayType
)
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

from VizualizationApp import VizualizationApp


# =============================================================================
# SPARK SCHEMAS
# =============================================================================

# Schema for per-bus state maintained across micro-batches
STATE_SCHEMA = StructType([
    StructField("last_lat", DoubleType()),
    StructField("last_lon", DoubleType()),
    StructField("last_ts", DoubleType()),              
    StructField("smoothed_speed", DoubleType()),       # Exponentially smoothed speed (m/s)
    StructField("prediction_buffer", ArrayType(ArrayType(DoubleType()))),  # [(timestamp, prediction), ...]
    StructField("last_processed_seq", IntegerType())   # Last stop sequence for arrival detection
])

# Schema for output records sent to visualization
OUTPUT_SCHEMA = StructType([
    StructField("bus_id", StringType()),
    StructField("lat", DoubleType()),
    StructField("lon", DoubleType()),
    StructField("next_stop_name", StringType()),
    StructField("next_stop_lat", DoubleType()),
    StructField("next_stop_lon", DoubleType()),
    StructField("seconds_to_scheduled", DoubleType()),
    StructField("scheduled_time", StringType()),
    StructField("predicted_delay_math", DoubleType()),
    StructField("is_arrival_event", IntegerType()),
    StructField("actual_delay", DoubleType()),         # Ground truth (valid only on arrival)
    StructField("final_math_error", DoubleType()),
    StructField("direction_id", IntegerType()),
    StructField("last_update", StringType()),
    # ML Feature columns for model inference
    StructField("stop_sequence", IntegerType()),
    StructField("distance_m", DoubleType()),
    StructField("hour", IntegerType()),
    StructField("minute", IntegerType()),
    StructField("weekday", IntegerType()),
    StructField("time_of_day", IntegerType())
])


class StatefulProcessor:
    """
    Handles stateful stream processing for bus delay predictions.
    
    This processor maintains per-bus state across streaming micro-batches,
    enabling speed estimation, delay prediction, and arrival detection.
    It supports both heuristic (physics-based) and ML model predictions.
    
    Attributes:
        spark: Active SparkSession
        config: Application configuration
        model: Loaded Spark ML model (optional)
        ml_prediction_cache: Stores predictions for evaluation against ground truth
        viz: Visualization app for dashboard updates
    """
    
    def __init__(self, spark, gtfs_loader, config):
        """
        Initialize the processor with required dependencies.
        
        Args:
            spark: Active SparkSession
            gtfs_loader: GTFSLoader instance with loaded timetable data
            config: Config instance with application settings
        """
        self.spark = spark
        self.config = config
        self.model = None
        
        # Cache for ML predictions - used to compare against ground truth on arrival
        # Structure: { bus_id: {'prediction': float, 'timestamp': float} }
        self.ml_prediction_cache: Dict[str, Dict] = {}
        
        # Broadcast static data to all workers
        self.bc_stops = spark.sparkContext.broadcast(gtfs_loader.stops_by_dir)
        self.bc_schedule = spark.sparkContext.broadcast(gtfs_loader.schedule_lookup)
        self.bc_destinations = spark.sparkContext.broadcast(gtfs_loader.destination_by_dir)
        
        # Load ML model if available
        self._load_ml_model()
        
        # Initialize visualization component
        self.viz = VizualizationApp(
            gtfs_loader.stops_by_dir, 
            config, 
            gtfs_loader.destination_by_dir
        )

    def _load_ml_model(self) -> None:
        """Attempt to load the Spark ML pipeline model for inference."""
        if os.path.exists(self.config.MODEL_PATH):
            try:
                self.model = PipelineModel.load(str(self.config.MODEL_PATH))
                print(">>> Spark ML Model loaded successfully.")
            except Exception as e:
                print(f">>> WARNING: Could not load ML model: {e}")
                print(">>> Falling back to heuristic predictions.")
        else:
            print(f">>> ML Model not found at {self.config.MODEL_PATH}")

    def process_stream(self, raw_df: DataFrame):
        """
        Set up the streaming pipeline with stateful processing.
        
        This method configures the Spark Structured Streaming query with:
        1. Data cleaning and type conversion
        2. Stateful per-bus processing via applyInPandasWithState
        3. Output to visualization dashboard
        
        Args:
            raw_df: Raw streaming DataFrame from CSV source
            
        Returns:
            StreamingQuery: The active streaming query
        """
        # Clean and preprocess incoming data
        clean_df = self._preprocess_stream(raw_df)
        
        # Get the stateful processing function (closure over broadcast vars)
        stateful_func = self._create_bus_state_function()
        
        # Apply stateful transformation grouped by bus ID
        processed_df = clean_df \
            .groupBy("codBus") \
            .applyInPandasWithState(
                func=stateful_func,
                outputStructType=OUTPUT_SCHEMA,
                stateStructType=STATE_SCHEMA,
                outputMode="Update",
                timeoutConf="ProcessingTimeTimeout"
            )
        
        # Start streaming query with visualization sink
        return processed_df.writeStream \
            .outputMode("update") \
            .foreachBatch(self._handle_batch) \
            .start()

    def _preprocess_stream(self, raw_df: DataFrame) -> DataFrame:
        """
        Clean and transform raw streaming data.
        
        Converts string columns to proper types and maps direction codes
        to standard direction IDs (0 = outbound, 1 = inbound).
        """
        return raw_df \
            .withColumn("lat", F.col("lat").cast("double")) \
            .withColumn("lon", F.col("lon").cast("double")) \
            .withColumn(
                "direction_id",
                F.when(F.col("sentido").cast("string").isin("1", "1.0", "Ida"), 0)
                 .otherwise(1)
            )

    def _create_bus_state_function(self):
        """
        Create the stateful processing function for per-bus updates.
        
        Returns a closure that captures broadcast variables and implements
        the core logic for speed estimation, delay prediction, and arrival detection.
        """
        # Capture broadcast values for worker access
        stops_data = self.bc_stops.value
        schedule_data = self.bc_schedule.value
        default_speed = self.config.DEFAULT_SPEED_MPS

        def update_bus_state(
            key: Tuple[str], 
            pdf_iter: Iterator[pd.DataFrame], 
            state_ref
        ) -> Iterator[pd.DataFrame]:
            """
            Process updates for a single bus across micro-batches.
            
            This function is called once per bus per micro-batch and maintains
            state across invocations for speed smoothing and prediction tracking.
            """
            bus_id = key[0]
            
            # Restore or initialize state
            state = _restore_state(state_ref, default_speed)
            results = []
            
            for pdf in pdf_iter:
                pdf = pdf.sort_values("last_update")
                
                for _, row in pdf.iterrows():
                    # Parse and validate row data
                    parsed = _parse_row(row)
                    if parsed is None:
                        continue
                    
                    curr_lat, curr_lon, curr_dt, curr_ts, direction = parsed
                    
                    # Skip duplicate timestamps (deduplication)
                    if state['last_ts'] is not None and curr_ts <= state['last_ts']:
                        continue
                    
                    # Get route stops for this direction
                    route_stops = _get_route_stops(stops_data, direction)
                    if not route_stops:
                        continue
                    
                    # Update speed estimate using exponential smoothing
                    state['speed'] = _update_speed(
                        state, curr_lat, curr_lon, curr_ts, default_speed
                    )
                    
                    # Find nearest stop and schedule
                    target_stop, dist_to_stop = _find_nearest_stop(
                        curr_lat, curr_lon, route_stops
                    )
                    current_stop_seq = int(target_stop['stop_sequence'])
                    
                    # Look up scheduled arrival time
                    best_sched_dt = _find_scheduled_time(
                        target_stop, direction, curr_dt, schedule_data
                    )
                    
                    # Calculate delay prediction using physics model
                    eta_seconds = dist_to_stop / max(state['speed'], 1.0)
                    sec_to_sched = (best_sched_dt - curr_dt).total_seconds() if best_sched_dt else eta_seconds
                    delay_prediction = eta_seconds - sec_to_sched
                    
                    sched_time_str = best_sched_dt.strftime("%H:%M:%S") if best_sched_dt else "N/A"
                    
                    # Check for arrival event and calculate ground truth
                    arrival_result = _check_arrival(
                        bus_id, current_stop_seq, state['last_seq'],
                        route_stops, curr_lat, curr_lon, curr_ts,
                        state['speed'], schedule_data, direction,
                        state['pred_buffer']
                    )
                    
                    is_arrival = arrival_result['is_arrival']
                    actual_delay = arrival_result['actual_delay']
                    math_error = arrival_result['math_error']
                    
                    if is_arrival:
                        state['pred_buffer'] = []  # Reset buffer after evaluation
                    
                    # Update state for next iteration
                    state['last_seq'] = current_stop_seq
                    state['pred_buffer'].append([float(curr_ts), float(delay_prediction)])
                    
                    # Build ML feature set
                    weekday = curr_dt.weekday()
                    weekday = -1 if weekday == 6 else weekday  # Sunday encoded as -1
                    time_of_day = curr_dt.hour * 60 + curr_dt.minute
                    
                    # Append result record
                    results.append({
                        "bus_id": bus_id,
                        "lat": curr_lat,
                        "lon": curr_lon,
                        "next_stop_name": target_stop['stop_name'],
                        "next_stop_lat": float(target_stop['stop_lat']),
                        "next_stop_lon": float(target_stop['stop_lon']),
                        "seconds_to_scheduled": float(sec_to_sched),
                        "scheduled_time": sched_time_str,
                        "predicted_delay_math": float(delay_prediction),
                        "is_arrival_event": 1 if is_arrival else 0,
                        "actual_delay": float(actual_delay),
                        "final_math_error": float(math_error),
                        "direction_id": direction,
                        "last_update": row['last_update'],
                        "stop_sequence": int(target_stop['stop_sequence']),
                        "distance_m": float(dist_to_stop),
                        "hour": int(curr_dt.hour),
                        "minute": int(curr_dt.minute),
                        "weekday": int(weekday),
                        "time_of_day": int(time_of_day)
                    })
                    
                    # Update position state
                    state['last_lat'] = curr_lat
                    state['last_lon'] = curr_lon
                    state['last_ts'] = curr_ts
            
            # Persist state for next micro-batch
            if state['last_lat'] is not None:
                # Limit buffer size to prevent memory issues
                if len(state['pred_buffer']) > 100:
                    state['pred_buffer'] = state['pred_buffer'][-100:]
                    
                state_ref.update((
                    float(state['last_lat']),
                    float(state['last_lon']),
                    float(state['last_ts']),
                    float(state['speed']),
                    state['pred_buffer'],
                    int(state['last_seq'])
                ))
            
            if results:
                yield pd.DataFrame(results)
        
        return update_bus_state

    def _handle_batch(self, df: DataFrame, batch_id: int) -> None:
        """
        Process each micro-batch: run ML inference and update visualization.
        
        This method runs on the driver and handles:
        1. ML model inference on the batch
        2. Caching predictions for later evaluation
        3. Evaluating predictions against ground truth on arrivals
        4. Updating the dashboard visualization
        """
        if df.count() == 0:
            return
        
        destinations = self.bc_destinations.value
        
        # Run ML inference and cache predictions
        ml_predictions = self._run_ml_inference(df)
        
        # Print batch summary
        self._print_batch_summary(df, batch_id, destinations)
        
        # Process rows and collect metrics
        rows = df.collect()
        live_data, math_errors, ml_errors = self._process_batch_rows(
            rows, ml_predictions, destinations
        )
        
        # Update visualization
        self.viz.update_dashboard(live_data, math_errors, ml_errors)
        
        # Print evaluation metrics
        self._print_metrics_summary()

    def _run_ml_inference(self, df: DataFrame) -> Dict[str, float]:
        """
        Run ML model inference on the current batch.
        
        Returns a dict mapping bus_id to predicted delay.
        Also caches predictions for later comparison with ground truth.
        """
        ml_predictions = {}
        
        if self.model is None:
            return ml_predictions
        
        try:
            preds_df = self.model.transform(df)
            pred_rows = preds_df.select("bus_id", "prediction").collect()
            ml_predictions = {r['bus_id']: r['prediction'] for r in pred_rows}
            
            # Cache predictions for evaluation
            current_time = time.time()
            for bus_id, pred in ml_predictions.items():
                self.ml_prediction_cache[bus_id] = {
                    'prediction': pred,
                    'timestamp': current_time
                }
            
            print(f"[ML] Inference completed for {len(ml_predictions)} buses")
            
        except Exception as e:
            print(f"[ML ERROR] Inference failed: {e}")
        
        return ml_predictions

    def _print_batch_summary(
        self, df: DataFrame, batch_id: int, destinations: Dict
    ) -> None:
        """Print a formatted summary of the current batch."""
        bus_count = df.count()
        print(f"\n{'='*70}")
        print(f"Processing Batch {batch_id} with {bus_count} buses...")
        print(f"{'='*70}")
        
        # Add direction labels for display
        display_df = df.withColumn(
            "direction_label",
            F.when(F.col("direction_id") == 0,
                   F.lit(f"{destinations.get(0, 'Unknown')} (Outbound)"))
             .otherwise(F.lit(f"{destinations.get(1, 'Unknown')} (Inbound)"))
        ).select(
            "bus_id",
            "predicted_delay_math",
            "scheduled_time",
            "next_stop_name",
            "direction_label"
        )
        display_df.show(truncate=False)

    def _process_batch_rows(
        self, 
        rows: List, 
        ml_predictions: Dict[str, float],
        destinations: Dict
    ) -> Tuple[List[Dict], List[float], List[float]]:
        """
        Process batch rows for visualization and evaluation.
        
        Returns:
            Tuple of (live_data, math_errors, ml_errors)
        """
        live_data = []
        math_errors = []
        ml_errors = []
        
        for r in rows:
            bus_id = r['bus_id']
            sched_str = r['scheduled_time'] or "N/A"
            
            # Format direction string
            d_id = r['direction_id']
            dir_label = "Outbound" if d_id == 0 else "Inbound"
            dest_name = destinations.get(d_id, "Unknown")
            direction_str = f"{dest_name} ({dir_label})"
            
            # Get ML prediction (fall back to heuristic if not available)
            ml_delay = ml_predictions.get(bus_id, r['predicted_delay_math'])
            
            # Select display delay based on configured mode
            if self.config.SELECTED_MODE == "machine_learning":
                display_delay = ml_delay
            else:
                display_delay = r['predicted_delay_math']
            
            live_data.append({
                "bus_id": bus_id,
                "lat": r['lat'],
                "lon": r['lon'],
                "next_stop": r['next_stop_name'],
                "sched_time": sched_str,
                "delay_math": r['predicted_delay_math'],
                "delay_ml": ml_delay,
                "delay_display": display_delay,
                "next_lat": r['next_stop_lat'],
                "next_lon": r['next_stop_lon'],
                "direction_formatted": direction_str,
                "prediction_method": self.config.SELECTED_MODE,
                "last_update": r['last_update']
            })
            
            # Handle arrival events - evaluate predictions
            if r['is_arrival_event'] == 1:
                actual_delay = r['actual_delay']
                
                # Evaluate heuristic prediction
                if r['final_math_error'] >= 0:
                    math_errors.append(r['final_math_error'])
                    print(f"[EVAL] Bus {bus_id}: Heuristic error = {r['final_math_error']:.1f}s")
                
                # Evaluate ML prediction from cache
                if bus_id in self.ml_prediction_cache:
                    cached = self.ml_prediction_cache[bus_id]['prediction']
                    ml_error = abs(cached - actual_delay)
                    ml_errors.append(ml_error)
                    print(f"[EVAL] Bus {bus_id}: ML error = {ml_error:.1f}s")
                    del self.ml_prediction_cache[bus_id]
        
        return live_data, math_errors, ml_errors

    def _print_metrics_summary(self) -> None:
        """Print cumulative evaluation metrics."""
        n_math = len(self.viz.global_math_errors)
        n_ml = len(self.viz.global_ml_errors)
        
        if n_math > 0 or n_ml > 0:
            mae_math = sum(self.viz.global_math_errors) / n_math if n_math > 0 else 0
            mae_ml = sum(self.viz.global_ml_errors) / n_ml if n_ml > 0 else 0
            print(f"\n--- Evaluation Metrics ---")
            print(f"Heuristic: {n_math} samples, MAE = {mae_math:.2f}s")
            print(f"ML Model:  {n_ml} samples, MAE = {mae_ml:.2f}s")
            print(f"--------------------------\n")


# =============================================================================
# HELPER FUNCTIONS (used within stateful processing)
# =============================================================================

def _restore_state(state_ref, default_speed: float) -> Dict:
    """Restore state from Spark state store or initialize defaults."""
    if state_ref.exists:
        s = state_ref.get
        return {
            'last_lat': s[0],
            'last_lon': s[1],
            'last_ts': s[2],
            'speed': s[3],
            'pred_buffer': list(s[4]) if s[4] else [],
            'last_seq': s[5] if len(s) > 5 and s[5] is not None else -1
        }
    return {
        'last_lat': None,
        'last_lon': None,
        'last_ts': None,
        'speed': default_speed,
        'pred_buffer': [],
        'last_seq': -1
    }


def _parse_row(row) -> Optional[Tuple]:
    """Parse and validate a data row. Returns None if invalid."""
    try:
        curr_lat = float(row['lat'])
        curr_lon = float(row['lon'])
        curr_dt = datetime.strptime(row['last_update'], "%Y-%m-%d %H:%M:%S")
        curr_ts = curr_dt.timestamp()
        direction = int(row['direction_id'])
        return curr_lat, curr_lon, curr_dt, curr_ts, direction
    except (ValueError, KeyError, TypeError):
        return None


def _get_route_stops(stops_data: Dict, direction: int) -> List[Dict]:
    """Get route stops for direction, with fallback to opposite direction."""
    route_stops = stops_data.get(direction, [])
    if not route_stops:
        # Try opposite direction as fallback
        direction = 1 if direction == 0 else 0
        route_stops = stops_data.get(direction, [])
    return route_stops


def _update_speed(
    state: Dict, 
    curr_lat: float, 
    curr_lon: float, 
    curr_ts: float,
    default_speed: float
) -> float:
    """
    Update speed estimate using raw instantaneous speed.
    
    Calculates speed directly from distance traveled and time elapsed.
    """
    if state['last_lat'] is None:
        return state['speed']
    
    dist_delta = geodesic(
        (state['last_lat'], state['last_lon']),
        (curr_lat, curr_lon)
    ).meters
    time_delta = curr_ts - state['last_ts']
    
    # Only update if there's meaningful movement
    if time_delta > 0 and dist_delta > 5:
        inst_speed = dist_delta / time_delta
        return inst_speed  # Use raw instantaneous speed
    
    return state['speed']


def _find_nearest_stop(
    lat: float, lon: float, route_stops: List[Dict]
) -> Tuple[Dict, float]:
    """Find the nearest stop to the current position."""
    target_stop = min(
        route_stops,
        key=lambda s: geodesic((lat, lon), (s['stop_lat'], s['stop_lon'])).meters
    )
    dist_to_stop = geodesic(
        (lat, lon), 
        (target_stop['stop_lat'], target_stop['stop_lon'])
    ).meters
    return target_stop, dist_to_stop


def _find_scheduled_time(
    stop: Dict, 
    direction: int, 
    current_dt: datetime,
    schedule_data: Dict
) -> Optional[datetime]:
    """Find the closest scheduled arrival time for a stop."""
    sched_key = (str(stop['stop_id']), direction)
    sched_times = schedule_data.get(sched_key, [])
    
    best_sched_dt = None
    min_diff = float('inf')
    
    for t_str in sched_times:
        try:
            h, m, s = map(int, t_str.split(':'))
            # Handle times past midnight (e.g., 25:00:00)
            day_offset = 1 if h >= 24 else 0
            h = h % 24
            sched_dt = current_dt.replace(
                hour=h, minute=m, second=s, microsecond=0
            ) + timedelta(days=day_offset)
            
            diff = abs((sched_dt - current_dt).total_seconds())
            # Only consider schedules within 2 hours
            if diff < min_diff and diff < 7200:
                min_diff = diff
                best_sched_dt = sched_dt
        except (ValueError, AttributeError):
            continue
    
    return best_sched_dt


def _check_arrival(
    bus_id: str,
    current_seq: int,
    last_seq: int,
    route_stops: List[Dict],
    curr_lat: float,
    curr_lon: float,
    curr_ts: float,
    speed: float,
    schedule_data: Dict,
    direction: int,
    pred_buffer: List
) -> Dict:
    """
    Check if bus has arrived at a stop and calculate ground truth delay.
    
    Arrival is detected when the stop sequence increases, indicating
    the bus has passed its previous target stop.
    
    Returns:
        Dict with 'is_arrival', 'actual_delay', and 'math_error'
    """
    result = {'is_arrival': False, 'actual_delay': 0.0, 'math_error': -1.0}
    
    # No arrival if sequence didn't increase
    if current_seq <= last_seq or last_seq < 0:
        return result
    
    result['is_arrival'] = True
    
    # Find the previous stop that we just passed
    prev_stop = None
    for s in route_stops:
        if int(s['stop_sequence']) == last_seq:
            prev_stop = s
            break
    
    if prev_stop is None:
        print(f"[ARRIVAL] Bus {bus_id}: Passed stop {last_seq} (stop data not found)")
        return result
    
    # Calculate actual arrival time by back-projecting from current position
    dist_from_prev = geodesic(
        (curr_lat, curr_lon),
        (prev_stop['stop_lat'], prev_stop['stop_lon'])
    ).meters
    
    time_since_passing = dist_from_prev / max(speed, 1.0)
    estimated_arrival_ts = curr_ts - time_since_passing
    estimated_arrival_dt = datetime.fromtimestamp(estimated_arrival_ts)
    
    # Find scheduled time for the previous stop
    actual_delay = _calculate_actual_delay(
        prev_stop, direction, estimated_arrival_dt, schedule_data
    )
    result['actual_delay'] = actual_delay
    
    print(f"[ARRIVAL] Bus {bus_id}: Passed stop {last_seq} -> {current_seq}")
    print(f"          Ground truth delay: {actual_delay:.1f}s")
    
    # Calculate prediction error from buffer
    errors = []
    for pred_ts, pred_delay in pred_buffer:
        if (curr_ts - pred_ts) < 600:  # Only recent predictions (10 min)
            errors.append(abs(pred_delay - actual_delay))
    
    if errors:
        result['math_error'] = sum(errors) / len(errors)
        print(f"[EVAL] Bus {bus_id}: Heuristic MAE = {result['math_error']:.1f}s ({len(errors)} samples)")
    
    return result


def _calculate_actual_delay(
    stop: Dict,
    direction: int,
    arrival_dt: datetime,
    schedule_data: Dict
) -> float:
    """Calculate actual delay by comparing arrival time to schedule."""
    sched_key = (str(stop['stop_id']), direction)
    sched_times = schedule_data.get(sched_key, [])
    
    if not sched_times:
        return 0.0
    
    # Find closest scheduled time
    best_sched_dt = None
    min_diff = float('inf')
    
    for t_str in sched_times:
        try:
            h, m, s = map(int, t_str.split(':'))
            day_offset = 1 if h >= 24 else 0
            h = h % 24
            sched_dt = arrival_dt.replace(
                hour=h, minute=m, second=s, microsecond=0
            ) + timedelta(days=day_offset)
            
            diff = abs((sched_dt - arrival_dt).total_seconds())
            if diff < min_diff and diff < 7200:
                min_diff = diff
                best_sched_dt = sched_dt
        except (ValueError, AttributeError):
            continue
    
    if best_sched_dt is None:
        return 0.0
    
    # Positive = late, negative = early
    return (arrival_dt - best_sched_dt).total_seconds()