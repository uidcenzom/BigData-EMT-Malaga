# Project
This project aims to address a common challenge in urban mobility: the unpredictability of public transport. By leveraging live data from Málaga's EMT public transport service, this project will develop a data-driven system to analyze and predict the real-time performance of Bus Line 31.

The primary goal is to build a machine learning model that can provide an accurate, minute-by-minute prediction of whether a specific bus is on time, late, or early. This moves beyond simple GPS tracking to provide predictive, actionable insights.

The project will deliver two key components:
1. A historical analysis of Line 31's performance, using clustering to identify systemic delay patterns (e.g., specific times of day, locations) that cause a bus to be late.
2. A real-time prediction engine built with Apache Spark. This engine will ingest live bus data, apply a trained regression model, and output a continuous stream of predictions (in minutes) for the bus's expected deviation from its schedule.

## Description
The goal of this project is to analyze and predict in real time the behaviour of the bus Line 31 in the city of Malaga, using the open data provided by EMT public transport service.

1. **Clustering:** Use clustering to analyze and estimate if the bus line is late based on the time of the day.
2. **Classification/Streaming:** Use the Dataset of real-time points of the buses as well the Static schedule. And then build a regression ML Model, which will predicte in minutes if the bus is late, on time or is early

## Explenation of the chosen Project

We have chosen it, as it is a very practical issue in day to day life. The busses sometimes in Malaga can be early and leave before the schedule, can be late or on Time. Our model is to predict with real-time data how late or early its gonne be.


## Terms and Data to be Used:

**Data Sources:**
- Emt Real-time API: Which provides live data for all active buses
- Emt Static GTFS: Which provides the official schedule including stop times.

**Technology:**
- Framework: Apache Spark
- Language: Python
- Libraries:
    - SparkML: To build the regression pipeline
    - Spark Structured Streaming: To process live data and make real-time predictions
    - Padas: For initial data exploration and clearning

## Timeline
|Day|Task|
|----|----|
|Nov 11| **Project Start:** Define Project Outline, Start exploring the data and show the project proposal|
|Nov 18 | **Data Preprocessing & Alignment:** Write scripts to parse the schedules. Begin map-matching task to align historical GPS data with the official bus stops. |
| Nov 25 | **Feature Engineering:** Complete the creation of the training dataset. Calculating tyhe historical delay based on the schedule for the bus line at every stop. And start training and tuning the regression model. |
| Dec 2 | **Stream Integration and Preparation for Presentation:** Evaluate the Model and integrate it with Spark Stream to make real-time delay predictions. As well prepare the final demo and presentation of the results.




### Data Structure


|Header|Meaning|Description|
|----|----|----|
|_id| Record ID | Internal unique identifier for the record in the dataset |
| codBus | Bus Code | Unique Identifier for each physical bus |
| codLinea | Line Code | Identifier for the bus line or route, Indicates which route the bus belongs to |
| sentido | Direction | Indicates the direction of the Bus. 1 = outbound / 2 = return |
| lon. | Longtitude | Coordinate - Longtitude |
| lat. | Latitude | Coordinate - Latitude |
| codParIni | Starting/Stop Code | Initial Bus stop or current tartget bus stop? |
| last_update | Last Update Timestamp | Date and time of the most recent update |


# 1. streaming_linea11.py — Real-Time Data Collection

### What it does
- Downloads real-time GPS data for all buses from OpenData Málaga every minute.  
- Filters only buses belonging to Line 11.  
- Computes:
  - current stop,
  - next stop,
  - distance to the next stop,
  - next scheduled arrival time (from GTFS).  

- Saves two files:
  - `linea11_YYYYMMDD_HHMMSS.csv` → historical snapshot archive  
  - `linea11_status_live.csv` → minute-by-minute live dataset for predictions  

### Why it is essential
- The historical snapshots are needed to reconstruct true delays.  
- The live CSV is used by the prediction engine to forecast delays in real time.  


# 2. creatoreDatasetTraining.py — Training Dataset Construction

### What it does
- Loads all previously saved snapshots.  
- For each GPS position:
  - finds the nearest stop,
  - determines the route order using `stop_sequence`,
  - detects when the bus has “arrived” at a stop (within 150 meters).  

- For each detected arrival:
  - reconstructs the scheduled arrival time using GTFS,
  - computes the true delay at that stop.  

- For the minutes preceding each arrival:
  - collects all snapshots,
  - assigns each of them the delay that will actually occur.  

- Produces the final training dataset:
  - `training_dataset_line11.csv`

### Why it is essential
It transforms raw, unordered GPS data into a supervised learning dataset:
current bus state → delay that will happen at the next stop

This is exactly the target that the regression model must learn.  


# 3. regressor.py — Model Training

### What it does
- Loads the training dataset.  
- Builds features such as:
  - position (lat, lon),
  - distance to the next stop,
  - stop sequence,
  - seconds until scheduled arrival,
  - hour, minute, weekday.  

- Trains a Random Forest Regressor.  
- Evaluates the model using MAE.  
- Saves the trained model:
  - `delay_predictor_line11.pkl`

### Why it is essential
This script contains the intelligence of the system.  
It learns from historical behaviour how delays form, evolve, and depend on the real-time state of the bus.  


# 4. predictor.py — Real-Time Delay Prediction

### What it does
- Loads:
  - the trained model,
  - the live dataset `linea11_status_live.csv`.  

- Computes the same features used during training.  
- Produces a prediction for each active bus:
  - expected delay in seconds at the next stop.  

- Optionally displays or saves these results.  

### Why it is essential
It answers the core question of the entire project:
Given the current state of the bus, will it arrive late or early at the next stop?




##
