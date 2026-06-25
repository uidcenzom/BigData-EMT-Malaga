"""
Config.py - Application Configuration

Centralized configuration for the EMT Malaga bus tracking application.
Contains all configurable parameters including paths, API endpoints,
and prediction mode settings.
"""

import os
from pathlib import Path


class PredictionMode:
    """Available prediction modes for delay vizualization on the map."""
    ML_MODEL = "machine_learning"
    HEURISTIC = "heuristic"


class Config:
    """
    Application configuration settings.
    
    This class contains all configurable parameters for the streaming
    application. Modify these values to change the target bus line,
    prediction mode, or file paths.
    
    Attributes:
        TARGET_LINE: Bus line number to track (e.g., "11")
        SELECTED_MODE: Active prediction visualization mode (ML or heuristic)
        DEFAULT_SPEED_MPS: Default bus speed in meters per second
        INTERVAL: API polling interval in seconds
    """
    
    # Target bus line to monitor
    TARGET_LINE = "11"
    
    # Prediction mode (affects which delay is displayed on map)
    SELECTED_MODE = PredictionMode.HEURISTIC
    
    # Default bus speed (m/s) when no movement data available
    # 5.5 m/s ≈ 20 km/h (typical urban bus speed)
    DEFAULT_SPEED_MPS = 5.5
    
    # API polling interval in seconds
    INTERVAL = 60
    
    # --- Path Configuration ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    TIMETABLE_DIR = DATA_DIR / "timetable"
    STREAM_INPUT_DIR = BASE_DIR / "stream_data_stage"
    DASHBOARD_DIR = BASE_DIR / "dashboard"
    MODEL_PATH = BASE_DIR / "models" / "delay_predictor_spark"
    
    # EMT Malaga Open Data API
    API_URL = "https://datosabiertos.malaga.eu/recursos/transporte/EMT/EMTlineasUbicaciones/lineasyubicaciones.csv"
    
    # GTFS file paths
    STOPS_FILE = os.path.join(TIMETABLE_DIR, "stops.csv")
    TRIPS_FILE = os.path.join(TIMETABLE_DIR, "trips.csv")
    STOP_TIMES_FILE = os.path.join(TIMETABLE_DIR, "stop_times.csv")