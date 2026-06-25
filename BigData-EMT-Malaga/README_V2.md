# Real-Time Bus Delay Prediction System (Line 31 - EMT Malaga)
**Big Data Master Project - Final Submission**

## Project Description

Public transport unpredictability is a daily frustration for commuters. Buses in Malaga can be early (leaving before schedule), late, or on time - and there's usually no way to know until you're already waiting at the stop. This project addresses that problem by building a real-time streaming system that predicts bus delays for Line 31 using live GPS data.

The goal is not just to track where the bus is, but to predict *when* it will actually arrive at your stop. We combine two approaches:
1. A **physics-based heuristic** that uses distance and speed to estimate arrival time
2. A **Machine Learning model** (Random Forest) trained on historical data that learns patterns like rush hour delays, weekend behavior, etc.

The system processes live data, makes predictions, compares them to reality when buses arrive, and shows everything on a live dashboard. It's basically a proof-of-concept for what a smart transit system could look like.

---

## Data Sources

### EMT Real-time API
The city of Malaga provides an open data portal with live bus information. We poll this API every few seconds to get:
- `codBus` - Unique bus identifier
- `codLinea` - Line number (we filter for Line 31)
- `lat`, `lon` - GPS coordinates
- `sentido` - Direction (1 = outbound, 2 = return)
- `last_update` - Timestamp of the reading

The data comes as CSV and updates roughly every 30 seconds on the server side.

### GTFS Static Data
GTFS (General Transit Feed Specification) is a standard format for transit schedules. From EMT's GTFS feed, we extracted:
- **stops.txt** - All bus stop locations with coordinates
- **stop_times.txt** - Scheduled arrival times for each stop
- **trips.txt** - Route and direction information

We pre-processed this into lookup tables that map (stop_id, direction) -> list of scheduled times.

---

## Technology Stack

| Component | Technology | What We Use It For |
|-----------|------------|-------------------|
| **Core Framework** | Apache Spark (PySpark) | Distributed stream processing, handles all the heavy lifting |
| **Streaming** | Spark Structured Streaming | Processes data as micro-batches with exactly-once semantics |
| **Machine Learning** | Spark MLlib | Random Forest training and inference |
| **State Management** | `applyInPandasWithState` | Keeps track of each bus's history across batches |
| **Language** | Python 3.x | We wrote everything in Python |
| **Data Processing** | PySpark, pandas | Spark for distributed processing; pandas inside stateful streaming functions |
| **Geospatial** | geopy | Calculates distances between GPS coordinates (uses geodesic formula) |
| **Visualization** | Folium | Generates interactive maps with Leaflet.js |
| **HTTP Client** | requests | Polls the EMT API |
| **Styling** | Bootstrap | Makes the dashboard look decent |

---

## Architecture Overview

The system is structured as a data pipeline with four main components:

```
EMT API --> BusStreamProducer --> CSV Files --> Spark Streaming --> StatefulProcessor --> VizualizationApp --> HTML Dashboard
                                                      |
                                                      v
                                              ML Model (Random Forest)
```

### 1. Data Producer (`streaming_app/BusStreamProducer.py`)

This component is responsible for getting data into the system. It runs as a background thread (daemon) and does the following:

1. **Polls the API** - Makes an HTTP GET request to the EMT endpoint every N seconds (configurable)
2. **Parses the response** - The API returns CSV data, which we read into a pandas DataFrame
3. **Cleans the data** - Normalizes column names (the API uses Spanish names like 'latitud'), handles missing values, fixes data types
4. **Filters for our line** - We only care about Line 31, so we filter out everything else
5. **Deduplicates** - If the bus positions haven't changed since the last fetch, we skip writing (no point processing stale data)
6. **Writes to staging folder** - Each batch gets written as a new CSV file (e.g., `batch_1701792000.csv`)

Spark Structured Streaming monitors this folder and picks up new files automatically. This file-based approach is simple and robust - if something crashes, we don't lose data.

### 2. Stream Processor (`streaming_app/StatefulProcessor.py`)

This is the heart of the system. We use Spark Structured Streaming with **stateful processing** to handle the data.

#### Why Stateful?

The key insight is that you can't predict delays from a single GPS point. You need history:
- To calculate speed, you need the previous position and timestamp
- To detect if a bus just passed a stop, you need to know what stop it was heading to before
- To evaluate predictions, you need to remember what you predicted earlier

So we use `applyInPandasWithState`, which lets us maintain a "state" for each bus ID that persists across micro-batches. The state includes:
- `last_lat`, `last_lon` - Previous GPS position
- `last_ts` - Previous timestamp
- `smoothed_speed` - Exponentially smoothed speed estimate
- `prediction_buffer` - List of (timestamp, prediction) tuples for later evaluation
- `last_processed_seq` - Last stop sequence for arrival detection

#### Processing Logic

For each incoming record:

1. **Parse and validate** - Convert strings to proper types, skip invalid rows
2. **Restore state** - Load the previous state for this bus (or initialize if first time)
3. **Calculate speed** - Distance from last position / time elapsed. We apply exponential smoothing: `new_speed = 0.2 * instant + 0.8 * previous` to reduce GPS noise
4. **Find nearest stop** - Using geopy, we find the closest stop on the route
5. **Look up schedule** - Match the stop to its scheduled arrival times
6. **Predict delay**:
   - Heuristic: `ETA = distance / speed`, then `delay = ETA - time_to_scheduled`
   - ML: Run inference on the pre-trained model with features like hour, weekday, distance, etc.
7. **Detect arrivals** - If the stop sequence increased, the bus just passed a stop. This triggers evaluation.
8. **Update state** - Save current position, speed, and predictions for next batch

#### Output Schema

Each processed record has these fields:
- Bus identification (ID, direction)
- Position (lat, lon)
- Next stop info (name, coordinates, scheduled time)
- Predictions (heuristic delay, ML delay)
- Evaluation data (is_arrival, actual_delay, prediction_error)
- ML features (hour, minute, weekday, time_of_day, distance_m, stop_sequence)

### 3. Machine Learning (`machine_learning/`)

#### Dataset Creation (`creatoreDatasetTraining.py`)

Training a delay prediction model requires labeled data - we need to know what delay actually happened. This script:

1. Loads historical snapshots (we collected weeks of data)
2. For each GPS reading, finds the nearest stop
3. Identifies "arrival events" - when a bus gets within 150m of a stop
4. Reconstructs the actual arrival time and compares to schedule
5. Labels all readings leading up to an arrival with the actual delay that occurred
6. Outputs a training CSV with features and labels

The key features we engineered:
- `lat`, `lon` - Position
- `stop_sequence` - How far along the route
- `distance_m` - Distance to next stop
- `seconds_to_scheduled` - Time until scheduled arrival
- `hour`, `minute`, `weekday` - Temporal features
- `time_of_day` - Minutes since midnight

Target: `true_delay_seconds`

#### Model Training (`regressor.py`)

We train a **Random Forest Regressor** using Spark MLlib:

```python
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="true_delay_seconds",
    numTrees=400,
    maxDepth=25,
    minInstancesPerNode=2
)
```

We chose Random Forest because:
- Works well with tabular data
- Handles non-linear relationships
- Robust to outliers
- Provides feature importance scores

The pipeline includes a VectorAssembler to combine features, then the RF model. We do an 80/20 train/test split and evaluate using MAE and RMSE.

### 4. Visualization (`streaming_app/VizualizationApp.py`)

Since Spark is a backend engine (not a web server), we needed a way to display results. Our solution:

1. After each micro-batch, the `foreachBatch` sink fires
2. We collect the results to the driver
3. Generate a Folium map with:
   - Bus markers (color-coded: green=on time, orange=delayed, red=very late)
   - Dashed lines showing the route to next stop
   - Popups with details (bus ID, schedule, delay)
4. Generate an HTML table with all bus data
5. Write both to files (`index.html`, `map_component.html`)
6. The browser auto-refreshes to show updates

The dashboard also shows cumulative metrics - MAE for both heuristic and ML models, updated as predictions are evaluated.

---

## Why Stateful Streaming Was Necessary

This is worth emphasizing because it was a major design decision. Here's why stateless streaming would fail:

| Problem | Why Stateless Fails | Stateful Solution |
|---------|--------------------|--------------------|
| Speed calculation | No access to previous position | Store last_lat, last_lon, last_ts |
| Arrival detection | No memory of which stop was targeted | Store last_processed_seq |
| Speed smoothing | No historical speed to average with | Store smoothed_speed |
| Prediction evaluation | Can't compare predictions to actual arrival | Store prediction_buffer |

Spark's `applyInPandasWithState` gives us a clean API: for each key (bus_id), we get an iterator of new data and a state object. We can read/write the state, and Spark handles persistence, checkpointing, and recovery automatically.

---

## Challenges and Solutions

### 1. Serialization Issues

**Problem:** When Spark sends our code to worker nodes, it serializes (pickles) everything. We kept getting `ModuleNotFoundError: No module named 'StatefulProcessor'` because workers couldn't import our custom modules.

**Why it happens:** Spark workers have their own Python environment. If your module isn't installed or accessible there, imports fail.

**Solution:**
- Restructured code so that the stateful function is returned by a method (closure pattern)
- Moved helper functions outside the class to avoid serializing the entire class instance
- Made sure all dependencies (Config, GTFSLoader) are accessible or broadcast to workers

### 2. GPS Data Quality

**Problem:** Raw GPS coordinates are noisy. Buses would "teleport" hundreds of meters between readings, or drift while stationary. This made speed calculations unreliable.

**Example:**
- Reading 1: Position A at 10:00:00
- Reading 2: Position B at 10:00:05 (500m away - implies 100 m/s = 360 km/h?!)

**Solution:** Exponential smoothing for speed:
```python
if 0.1 < instant_speed < 25.0:  # Filter unrealistic speeds
    smoothed_speed = instant_speed * 0.2 + previous_speed * 0.8
```
This dampens sudden jumps while still responding to real speed changes.

### 3. Schedule Matching

**Problem:** GTFS uses a weird time format where times after midnight are represented as 24:00, 25:30, etc. (to indicate they belong to the previous day's service).

**Solution:** Parse the hour, detect if > 24, and add a day offset:
```python
day_offset = 1 if hour >= 24 else 0
hour = hour % 24
scheduled_dt += timedelta(days=day_offset)
```

### 4. Visualization from Backend

**Problem:** Spark processes data in distributed workers and aggregates at the driver. We wanted a real-time UI but didn't want to set up WebSockets or a proper web server.

**Solution:** The `foreachBatch` pattern. Every time a micro-batch completes:
1. Collect results to driver memory
2. Regenerate static HTML files
3. Browser refreshes (via meta tag) and shows updated data

It's not the most elegant solution, but it works well for a demo/prototype.

### 5. Arrival Detection

**Problem:** How do you know if a bus "arrived" at a stop? GPS isn't precise enough to detect exactly when it reaches the stop coordinates.

**Solution:** We use stop sequence changes as a proxy. The route has stops numbered 1, 2, 3, etc. When a bus's nearest stop changes from sequence N to N+1, we infer it passed stop N. We then back-calculate the approximate arrival time using the distance traveled since passing.

---

## Configuration (`streaming_app/Config.py`)

Key settings:
- `TARGET_LINE` - Which bus line to track (default: "31")
- `INTERVAL` - Seconds between API polls
- `STREAM_INPUT_DIR` - Where CSV files are staged
- `DASHBOARD_DIR` - Where HTML output goes
- `MODEL_PATH` - Path to the trained Spark ML model
- `SELECTED_MODE` - Which prediction to display ("heuristic" or "machine_learning")

---

## How to Run

1. Start the main application: `python MainApp.py`
2. This launches:
   - The producer thread (polls API, writes CSVs)
   - The Spark Streaming query (processes data, updates dashboard)
3. Open `dashboard/index.html` in a browser
4. Watch buses move in real-time with delay predictions

---

## Evaluation and Results

The system tracks prediction accuracy in real-time. Every time a bus arrives at a stop:
1. We know the actual delay (arrival time - scheduled time)
2. We compare it to our predictions from minutes ago
3. We calculate the absolute error and add it to our running totals

The dashboard shows cumulative MAE (Mean Absolute Error) for both models:
- **Heuristic MAE**: Typically around 45-90 seconds
- **ML Model MAE**: Typically 30-60 seconds (varies with data quality)

The ML model generally outperforms the heuristic because it learns patterns that simple physics can't capture (rush hour effects, specific problematic stops, etc.).

---

## Conclusion

This project demonstrates how Big Data tools (Spark Streaming, MLlib) can be applied to a real-world problem. We went from raw GPS coordinates to a working prediction system with visualization. The key technical insight is that stateful streaming is essential for this kind of time-series prediction task.

There's plenty of room for improvement - better ML models, more features, real-time web interface - but as a proof-of-concept for what's possible with open transit data, it works.
