import os
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    # Adjust path if your data folder is elsewhere
    DATA_DIR = os.path.join("data") 
    TIMETABLE_DIR = os.path.join(DATA_DIR, "timetable")
    
    STOPS_FILE = os.path.join(TIMETABLE_DIR, "stops.csv")
    TRIPS_FILE = os.path.join(TIMETABLE_DIR, "trips.csv")
    STOP_TIMES_FILE = os.path.join(TIMETABLE_DIR, "stop_times.csv")
    
    # The Stop you want to check (Partial match works, e.g. "Alameda")
    TARGET_STOP_NAME = "Avda. de Andalucía - Jardines Picasso" 

# ==============================================================================
# SCHEDULE VIEWER
# ==============================================================================
def load_data():
    print("Loading GTFS Data...")
    try:
        stops = pd.read_csv(Config.STOPS_FILE)
        trips = pd.read_csv(Config.TRIPS_FILE)
        stop_times = pd.read_csv(Config.STOP_TIMES_FILE)
    except FileNotFoundError as e:
        print(f"Error: Could not find GTFS files at {Config.TIMETABLE_DIR}")
        print(f"Details: {e}")
        return None, None, None

    # Filter for Line 11
    # Note: Ensure route_id is string for consistent comparison
    trips["route_id"] = trips["route_id"].astype(str)
    trips_11 = trips[trips["route_id"] == "11"]
    
    if trips_11.empty:
        print("Error: No trips found for Route ID 11.")
        return None, None, None

    # Merge to get a Master Table: [trip_id, arrival_time, stop_name, direction_id, headsign]
    merged = stop_times.merge(trips_11[["trip_id", "direction_id", "trip_headsign"]], on="trip_id")
    merged = merged.merge(stops[["stop_id", "stop_name"]], on="stop_id")
    
    return merged, trips_11, stops

def list_all_stops(merged_df):
    """Prints all unique stops serving Line 11"""
    print("\n--- STOPS ON LINE 11 ---")
    unique_stops = merged_df[["stop_name", "direction_id"]].drop_duplicates().sort_values(["direction_id", "stop_name"])
    
    for d in unique_stops["direction_id"].unique():
        print(f"\n[Direction {d}]")
        stops_in_dir = unique_stops[unique_stops["direction_id"] == d]
        for s in stops_in_dir["stop_name"]:
            print(f" - {s}")

def show_schedule_for_stop(merged_df, target_name):
    """Prints the schedule for a specific stop"""
    print(f"\n\n=== SCHEDULE FOR: '{target_name}' ===")
    
    # Filter by name (case insensitive partial match)
    mask = merged_df["stop_name"].str.contains(target_name, case=False, na=False)
    schedule = merged_df[mask]
    
    if schedule.empty:
        print(f"No stops found matching '{target_name}'. Check the list above.")
        return

    # Group by Direction (Usually 0 and 1)
    for d in schedule["direction_id"].unique():
        subset = schedule[schedule["direction_id"] == d].sort_values("arrival_time")
        
        # Get Headsign (Destination)
        headsign = subset.iloc[0]["trip_headsign"]
        print(f"\n--> Direction {d}: To {headsign}")
        print(f"    Total Trips: {len(subset)}")
        print("-" * 40)
        
        # Format Times nicely
        times = sorted(subset["arrival_time"].unique())
        
        # Print in rows of 10 for readability
        for i in range(0, len(times), 8):
            print("   " + "  ".join(times[i:i+8]))
        print("-" * 40)

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    data, _, _ = load_data()
    
    if data is not None:
        # Uncomment this to see all stop names
        # list_all_stops(data)
        
        # Show Schedule
        show_schedule_for_stop(data, Config.TARGET_STOP_NAME)