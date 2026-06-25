"""
VizualizationApp.py - Real-time Dashboard Visualization

Generates an interactive HTML dashboard showing live bus positions on a map
along with delay predictions and evaluation metrics. The dashboard is
auto-refreshed by the browser.

Features:
    - Interactive Folium map with bus markers and route lines
    - Color-coded delay status (early, on-time, delayed, very late)
    - Live data table with both heuristic and ML predictions
    - Cumulative accuracy metrics (MAE) for model evaluation
"""

import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

import folium
import pandas as pd

from Config import Config


class VizualizationApp:
    """
    Generates and updates the real-time bus tracking dashboard.
    
    This class creates an HTML dashboard that displays bus positions
    on a map and shows prediction accuracy metrics. The dashboard
    consists of a Folium map and a data table.
    
    Attributes:
        static_stops: Route geometry (stops per direction)
        config: Application configuration
        global_math_errors: Accumulated heuristic prediction errors
        global_ml_errors: Accumulated ML prediction errors
    """
    
    # Delay thresholds in seconds for color coding
    EARLY_THRESHOLD = -60      # More than 1 min early
    ON_TIME_THRESHOLD = 60    # Up to 1 min late
    DELAYED_THRESHOLD = 300    # Up to 5 min late
    
    def __init__(
        self, 
        static_stops: Dict, 
        config: Config, 
        destination_by_dir: Dict = None
    ):
        """
        Initialize the visualization app.
        
        Args:
            static_stops: Dict mapping direction_id to list of stops
            config: Application configuration
            destination_by_dir: Dict mapping direction_id to destination name
        """
        self.static_stops = static_stops
        self.config = config
        self.destination_by_dir = destination_by_dir or {}
        
        # Accumulated prediction errors for MAE calculation
        self.global_math_errors: List[float] = []
        self.global_ml_errors: List[float] = []
        
        # Initialize dashboard directory
        self._setup_directory()
    
    def _setup_directory(self) -> None:
        """Create a clean dashboard output directory."""
        if os.path.exists(self.config.DASHBOARD_DIR):
            shutil.rmtree(self.config.DASHBOARD_DIR)
        os.makedirs(self.config.DASHBOARD_DIR)
    
    def get_delay_status(self, delay_seconds: float) -> Tuple[str, str]:
        """
        Get color and status label based on delay.
        
        Args:
            delay_seconds: Predicted delay (positive = late, negative = early)
            
        Returns:
            Tuple of (color, status_label)
        """
        if delay_seconds < self.EARLY_THRESHOLD:
            return "blue", "Early"
        elif delay_seconds <= self.ON_TIME_THRESHOLD:
            return "green", "On Time"
        elif delay_seconds <= self.DELAYED_THRESHOLD:
            return "orange", "Delayed"
        else:
            return "red", "Very Late"
    
    def update_dashboard(
        self, 
        bus_data_list: List[Dict], 
        new_math_errors: List[float], 
        new_ml_errors: List[float]
    ) -> None:
        """
        Update the dashboard with new bus data and metrics.
        
        This method accumulates prediction errors over time and
        regenerates the HTML dashboard.
        
        Args:
            bus_data_list: List of bus data dictionaries
            new_math_errors: New heuristic prediction errors from this batch
            new_ml_errors: New ML prediction errors from this batch
        """
        # Accumulate errors for MAE calculation
        self.global_math_errors.extend(new_math_errors)
        self.global_ml_errors.extend(new_ml_errors)
        
        # Calculate current metrics
        metrics = self._calculate_metrics()
        
        # Generate dashboard
        self._generate_dashboard(bus_data_list, metrics)
    
    def _calculate_metrics(self) -> Dict:
        """Calculate current accuracy metrics."""
        n_math = len(self.global_math_errors)
        n_ml = len(self.global_ml_errors)
        
        mae_math = sum(self.global_math_errors) / n_math if n_math > 0 else 0.0
        mae_ml = sum(self.global_ml_errors) / n_ml if n_ml > 0 else 0.0
        
        return {
            "sample_size_math": n_math,
            "sample_size_ml": n_ml,
            "heuristic_mae": mae_math,
            "ml_mae": mae_ml
        }
    
    def _generate_dashboard(self, bus_data_list: List[Dict], metrics: Dict) -> None:
        """Generate the complete HTML dashboard."""
        # Create map
        bus_map = self._create_map(bus_data_list)
        
        # Create data table
        table_html = self._create_table(bus_data_list)
        
        # Get timestamp
        timestamp = self._get_timestamp(bus_data_list)
        
        # Generate full HTML
        full_html = self._create_full_html(table_html, metrics, timestamp)
        
        # Save files
        map_path = os.path.join(self.config.DASHBOARD_DIR, "map_component.html")
        index_path = os.path.join(self.config.DASHBOARD_DIR, "index.html")
        
        bus_map.save(map_path)
        with open(index_path, "w") as f:
            f.write(full_html)
    
    def _create_map(self, bus_data_list: List[Dict]) -> folium.Map:
        """Create the Folium map with bus markers and stops."""
        # Center on Malaga
        m = folium.Map(
            location=[36.7213, -4.4214],
            zoom_start=13,
            tiles="CartoDB positron"
        )
        
        # Add static stops
        self._add_stops_to_map(m)
        
        # Add bus markers
        for bus in bus_data_list:
            self._add_bus_to_map(m, bus)
        
        return m
    
    def _add_stops_to_map(self, m: folium.Map) -> None:
        """Add static stop markers to the map."""
        for direction in [0, 1]:
            for stop in self.static_stops.get(direction, []):
                folium.CircleMarker(
                    location=[stop['stop_lat'], stop['stop_lon']],
                    radius=2,
                    color="gray",
                    tooltip=f"{stop['stop_name']} ({stop['stop_code']})"
                ).add_to(m)
    
    def _add_bus_to_map(self, m: folium.Map, bus: Dict) -> None:
        """Add a single bus marker and route line to the map."""
        lat, lon = bus.get('lat'), bus.get('lon')
        if not lat:
            return
        
        delay = bus['delay_display']
        color, status = self.get_delay_status(delay)
        
        # Draw line to next stop
        if bus.get('next_lat'):
            folium.PolyLine(
                locations=[[lat, lon], [bus['next_lat'], bus['next_lon']]],
                color=color,
                weight=3,
                dash_array='5, 5'
            ).add_to(m)
        
        # Add bus marker with popup
        popup_content = (
            f"Bus: {bus['bus_id']}<br>"
            f"Next: {bus['next_stop']}<br>"
            f"Sched: {bus['sched_time']}<br>"
            f"Delay: {int(delay)}s"
        )
        
        folium.Marker(
            location=[lat, lon],
            popup=popup_content,
            icon=folium.Icon(color=color, icon="bus", prefix="fa")
        ).add_to(m)
    
    def _create_table(self, bus_data_list: List[Dict]) -> str:
        """Create the HTML data table."""
        if not bus_data_list:
            return "<p>No active buses.</p>"
        
        df = pd.DataFrame(bus_data_list)
        
        # Select and rename columns for display
        columns = [
            "bus_id", "direction_formatted", "next_stop", 
            "sched_time", "delay_math", "delay_ml"
        ]
        view = df[columns].copy()
        view.columns = [
            "Bus ID", "Direction", "Next Stop",
            "Sched. Arrival", "Heuristic (s)", "ML Model (s)"
        ]
        
        # Format delay values as integers
        view["Heuristic (s)"] = view["Heuristic (s)"].astype(int)
        view["ML Model (s)"] = view["ML Model (s)"].astype(int)
        
        return view.to_html(
            classes="table table-striped table-sm",
            index=False,
            border=0
        )
    
    def _get_timestamp(self, bus_data_list: List[Dict]) -> str:
        """Extract timestamp from bus data or use current time."""
        if bus_data_list and 'last_update' in bus_data_list[0]:
            return bus_data_list[0]['last_update'].split(' ')[-1]
        return datetime.now().strftime("%H:%M:%S")
    
    def _create_full_html(
        self, 
        table_html: str, 
        metrics: Dict, 
        timestamp: str
    ) -> str:
        """Generate the complete dashboard HTML."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Line {self.config.TARGET_LINE} Live</title>
            <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
            <style>
                body, html {{ height: 100%; margin: 0; overflow: hidden; }}
                #map-container {{ height: 55%; border-bottom: 2px solid #444; }}
                #table-container {{ height: 45%; overflow-y: auto; padding: 15px; }}
                .table th, .table td {{ text-align: left !important; vertical-align: middle; }}
                .metrics-panel {{
                    background: #e9ecef;
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 5px;
                    display: flex;
                    justify-content: space-around;
                }}
            </style>
        </head>
        <body>
            <div style="background:#333; color:white; padding:5px 15px; display:flex; justify-content:space-between;">
                <span><b>Line {self.config.TARGET_LINE} Monitor</b></span>
                <span>Updated: {timestamp}</span>
            </div>
            <div id="map-container">
                <iframe src="map_component.html" style="width:100%; height:100%; border:none;"></iframe>
            </div>
            <div id="table-container">
                <div class="metrics-panel">
                    <div style="color:#007bff">
                        <b>Heuristic:</b> {metrics['sample_size_math']} samples, MAE={metrics['heuristic_mae']:.1f}s
                    </div>
                    <div style="color:#6f42c1">
                        <b>ML Model:</b> {metrics['sample_size_ml']} samples, MAE={metrics['ml_mae']:.1f}s
                    </div>
                </div>
                {table_html}
            </div>
        </body>
        </html>
        """