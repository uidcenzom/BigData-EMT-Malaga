"""
BusStreamProducer.py - Real-time Bus Data Producer

This module fetches live bus location data from the EMT Malaga open data API
and writes it to CSV files for consumption by Spark Structured Streaming.

The producer runs in a background thread and handles:
    - Periodic API polling with configurable interval
    - Data cleaning and normalization
    - Duplicate detection to avoid redundant writes
    - Error handling for network issues
"""

import os
import io
import time
import shutil
import threading
from datetime import datetime
from typing import Optional

import requests
import pandas as pd

from Config import Config

# Configure Spark environment
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['JAVA_OPTS'] = '-Djava.io.tmpdir=/tmp'


class BusStreamProducer:
    """
    Produces streaming data by polling the EMT Malaga API.
    
    This class runs a background thread that periodically fetches bus
    locations from the public API, processes the data, and writes CSV
    files that Spark Streaming can consume.
    
    Attributes:
        config: Application configuration
        running: Flag to control the fetch loop
        last_snapshot: Previous data for duplicate detection
    """
    
    # Column name mapping from API format to internal format
    COLUMN_MAP = {
        "codlinea": "codLinea",
        "codbus": "codBus",
        "latitud": "lat",
        "longitud": "lon",
        "codparini": "codParIni"
    }
    
    def __init__(self, config: Config):
        """
        Initialize the producer.
        
        Args:
            config: Config instance with API URL and paths
        """
        self.config = config
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_snapshot: Optional[pd.DataFrame] = None
    
    def start(self) -> None:
        """Start the background fetch thread."""
        if self.running:
            return
        
        self._setup_directory()
        self.running = True
        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """Stop the background fetch thread."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _setup_directory(self) -> None:
        """Create a clean staging directory for stream files."""
        if os.path.exists(self.config.STREAM_INPUT_DIR):
            shutil.rmtree(self.config.STREAM_INPUT_DIR)
        os.makedirs(self.config.STREAM_INPUT_DIR)
    
    def _fetch_loop(self) -> None:
        """
        Main fetch loop running in background thread.
        
        Continuously polls the API at the configured interval and
        writes new data to CSV files for Spark to consume.
        """
        print(f">>> Producer started for Line {self.config.TARGET_LINE}...")
        
        # Columns expected by Spark schema
        output_columns = [
            "codBus", "codLinea", "sentido", 
            "lat", "lon", "codParIni", "last_update"
        ]
        
        while self.running:
            try:
                df = self._fetch_and_process()
                
                if df is not None and not df.empty:
                    # Check for duplicates (skip if positions unchanged)
                    if not self._is_duplicate(df):
                        self._write_batch(df, output_columns)
                        self.last_snapshot = df.copy()
                elif df is not None:
                    print(f"[PRODUCER] Line {self.config.TARGET_LINE} not active")
                    
            except requests.exceptions.Timeout:
                print("[PRODUCER] Request timed out, retrying...")
            except requests.exceptions.RequestException as e:
                print(f"[PRODUCER] Network error: {e}")
            except Exception as e:
                print(f"[PRODUCER] Unexpected error: {e}")
            
            time.sleep(self.config.INTERVAL)
    
    def _fetch_and_process(self) -> Optional[pd.DataFrame]:
        """
        Fetch data from API and process it.
        
        Returns:
            Processed DataFrame or None if fetch failed
        """
        response = requests.get(self.config.API_URL, timeout=10)
        
        if response.status_code != 200:
            print(f"[PRODUCER] API returned status {response.status_code}")
            return None
        
        # Parse CSV with error handling for truncated responses
        try:
            df = pd.read_csv(
                io.StringIO(response.text),
                on_bad_lines='skip'
            )
        except pd.errors.ParserError as e:
            print(f"[PRODUCER] CSV parse error (truncated data): {e}")
            return None
        
        return self._clean_data(df)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize the raw API data.
        
        Args:
            df: Raw DataFrame from API
            
        Returns:
            Cleaned DataFrame filtered to target line
        """
        # Normalize column names
        df.columns = df.columns.str.lower()
        df = df.rename(columns=self.COLUMN_MAP)
        
        # Clean line number (remove decimal suffix like "11.0")
        df["codLinea"] = df["codLinea"].astype(str).str.split('.').str[0]
        
        # Filter to target line
        df = df[df["codLinea"] == self.config.TARGET_LINE].copy()
        
        return df
    
    def _is_duplicate(self, df: pd.DataFrame) -> bool:
        """Check if this data is a duplicate of the last fetch."""
        if self.last_snapshot is None:
            return False
        
        try:
            return df["last_update"].values[0] == self.last_snapshot["last_update"].values[0]
        except (KeyError, IndexError):
            return False
    
    def _write_batch(self, df: pd.DataFrame, columns: list) -> None:
        """Write a batch of data to CSV for Spark to consume."""
        # Filter to expected columns only
        available_cols = [c for c in columns if c in df.columns]
        df_output = df[available_cols]
        
        # Generate unique filename
        filename = f"batch_{int(time.time())}.csv"
        filepath = os.path.join(self.config.STREAM_INPUT_DIR, filename)
        
        df_output.to_csv(filepath, index=False)
        
        fetch_time = datetime.now().strftime("%H:%M:%S")
        print(f"[PRODUCER] Fetched {len(df_output)} buses at {fetch_time}")