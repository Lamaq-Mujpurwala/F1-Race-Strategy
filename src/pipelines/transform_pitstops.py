import pandas as pd
import yaml
import os

def create_pit_stop_dataset(raw_laps_df: pd.DataFrame, min_pit_delta: float) -> pd.DataFrame:
    """
    Processes a raw DataFrame of lap data to create a clean dataset for
    training a pit stop delta prediction model.

    Args:
        raw_laps_df (pd.DataFrame): The combined DataFrame of all laps from all seasons.
        min_pit_delta (float): The minimum realistic time for a pit stop, to filter outliers.

    Returns:
        pd.DataFrame: A clean DataFrame where each row represents a pit stop event.
    """
    print("--- Starting Pit Stop Dataset Creation ---")

    # Filter for laps where a pit stop occurred.
    pit_stops_df = raw_laps_df[
        raw_laps_df['PitInTime'].notna() & raw_laps_df['PitOutTime'].notna()
    ].copy()
    print(f"Found {len(pit_stops_df)} total pit stop events.")

    # Convert time columns to timedelta format for calculation.
    pit_stops_df['PitInTime'] = pd.to_timedelta(pit_stops_df['PitInTime'])
    pit_stops_df['PitOutTime'] = pd.to_timedelta(pit_stops_df['PitOutTime'])

    # Calculate the target variable, 'PitDelta'.
    pit_stops_df['PitDelta'] = (pit_stops_df['PitOutTime'] - pit_stops_df['PitInTime']).dt.total_seconds()
    print("Calculated 'PitDelta' (total time in pits in seconds).")

    # Select the final features and the target variable.
    final_columns = ['PitDelta', 'Track', 'Year', 'Driver']
    
    # Add 'Year' column if it doesn't exist (for older data structures)
    if 'Year' not in pit_stops_df.columns and 'Date' in pit_stops_df.columns:
        pit_stops_df['Year'] = pd.to_datetime(pit_stops_df['Date']).dt.year

    final_df = pit_stops_df[final_columns]

    # Final cleaning: drop NaNs and unrealistic pit stop times.
    final_df = final_df.dropna(subset=['PitDelta'])
    final_df = final_df[final_df['PitDelta'] > min_pit_delta]
    print(f"Cleaned dataset has {len(final_df)} valid pit stop rows.")
    print("--- Pit Stop Dataset Creation Complete ---")
    
    return final_df

if __name__ == '__main__':
    # Load parameters from params.yaml
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    
    config = params['transform_pitstops']
    
    print(f"Loading combined data from: {config['input_path']}")
    master_df = pd.read_csv(config['input_path'])
    
    # Create the pit stop dataset
    pit_stop_data = create_pit_stop_dataset(master_df, config['min_pit_delta'])
    
    # Save the processed data
    os.makedirs(os.path.dirname(config['output_path']), exist_ok=True)
    pit_stop_data.to_csv(config['output_path'], index=False)
    print(f"Sample processed data saved to: {config['output_path']}")

