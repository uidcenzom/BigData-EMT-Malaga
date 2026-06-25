"""
MainApp.py - Application Entry Point

Main entry point for the EMT Malaga bus delay prediction streaming application.
This script initializes all components and starts the Spark Structured Streaming
pipeline for real-time bus tracking and delay prediction.

Usage:
    cd streaming_app && python MainApp.py
    OR
    python streaming_app/MainApp.py

The application will:
    1. Start fetching live bus data from the EMT API
    2. Load GTFS timetable data for schedule lookups
    3. Process bus positions using stateful streaming
    4. Generate a live dashboard with predictions
"""

import sys
import os
import warnings

# Suppress warnings in main process AND Spark workers (via env var)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Add streaming_app directory to path so Spark workers can find modules
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from pyspark.sql import SparkSession

from Config import Config
from BusStreamProducer import BusStreamProducer
from GTFSLoader import GTFSLoader
from StatefulProcessor import StatefulProcessor


class MainApp:
    """
    Main application orchestrator.
    
    Coordinates all components of the streaming pipeline:
    - BusStreamProducer: Fetches live data from API
    - GTFSLoader: Loads static timetable data
    - StatefulProcessor: Processes stream with predictions
    """
    
    def __init__(self, config: Config):
        """
        Initialize the application.
        
        Args:
            config: Application configuration instance
        """
        self.config = config
        self.producer = BusStreamProducer(config)
    
    def run(self) -> None:
        """
        Start the streaming application.
        
        This method:
        1. Starts the background data producer
        2. Initializes Spark session
        3. Loads static GTFS data
        4. Starts the streaming query
        """
        # Start background data fetching
        self.producer.start()
        
        # Initialize Spark
        spark = self._create_spark_session()
        
        try:
            # Load static timetable data
            print("\n>>> Loading GTFS timetable data...")
            gtfs_loader = GTFSLoader(spark, self.config)
            
            # Initialize the stateful processor
            print("\n>>> Initializing stream processor...")
            processor = StatefulProcessor(spark, gtfs_loader, self.config)
            
            # Create input stream
            print("\n>>> Starting streaming query...")
            raw_stream = self._create_input_stream(spark)
            
            # Start processing
            query = processor.process_stream(raw_stream)
            
            print("\n>>> Application running. Press Ctrl+C to stop.\n")
            query.awaitTermination()
            
        except KeyboardInterrupt:
            print("\n>>> Shutting down...")
            self.producer.stop()
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure the Spark session."""
        spark = SparkSession.builder \
            .master("local[*]") \
            .appName("EMT_Bus_Delay_Prediction") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        
        # Reduce log verbosity
        spark.sparkContext.setLogLevel("ERROR")
        
        return spark
    
    def _create_input_stream(self, spark: SparkSession):
        """
        Create the streaming DataFrame from CSV files.
        
        Reads CSV files written by BusStreamProducer from the
        staging directory.
        """
        schema = (
            "codBus STRING, codLinea STRING, sentido STRING, "
            "lat DOUBLE, lon DOUBLE, codParIni STRING, last_update STRING"
        )
        
        return spark.readStream \
            .format("csv") \
            .option("header", "true") \
            .schema(schema) \
            .load(str(self.config.STREAM_INPUT_DIR))


if __name__ == "__main__":
    config = Config()
    app = MainApp(config)
    app.run()