import pandas as pd
from typing import Dict, Any
import os
import yaml

params = yaml.safe_load(open("params.yaml"))['transform']

def clean_data(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Performs initial cleaning of the raw lap data.
    """
    print("--- Starting Data Cleaning ---")
    df_cleaned = df.dropna(subset=['LapTimeinSeconds']).copy()
    print(f"Dropped {len(df) - len(df_cleaned)} rows with missing LapTimeinSeconds.")

    dry_compounds = ['SOFT', 'MEDIUM', 'HARD']
    original_rows = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned['Compound'].isin(dry_compounds)]
    print(f"Dropped {original_rows - len(df_cleaned)} rows for non-dry compounds.")

    original_rows = len(df_cleaned)
    #race_median_times = df_cleaned.groupby(['Year', 'Track'])['LapTimeinSeconds'].transform('median')
    #utlier_threshold = params.get('outlier_lap_time_percentage', 2.0)
    #df_cleaned = df_cleaned[df_cleaned['LapTimeinSeconds'] < (race_median_times * outlier_threshold)]
    #print(f"Dropped {original_rows - len(df_cleaned)} outlier laps.")
    
    print("--- Data Cleaning Complete ---")
    return df_cleaned


def create_features_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the final columns needed for the clean dataset that will be
    stored in PostgreSQL. No one-hot encoding is done here.
    """
    print("--- Selecting Features for Database ---")
    relevant_cols = [
        'LapTimeinSeconds',
        'TyreLife',
        'LapNumber',
        'Compound',
        'Track',
        'Year',
        'Driver',
        'AirTemp',
        'TrackTemp'
    ]
    # Drop rows with missing values in the selected columns, if any
    df_selected = df[relevant_cols].dropna().copy()
    
    print(f"Selected {len(relevant_cols)} relevant columns.")
    print(f"Final shape for DB: {df_selected.shape}")
    
    return df_selected

def process_and_save_features(input_path: str, output_path: str, params: Dict[str, Any]):
    """
    Orchestrates the full data processing pipeline: loads data, cleans it,
    selects features, and saves the final dataset.

    Args:
        input_path (str): Path to the raw input CSV file.
        output_path (str): Path to save the processed output CSV file.
        params (Dict[str, Any]): Dictionary of parameters for the pipeline.
    """
    print("--- Running Full Feature Engineering Pipeline ---")
    
    # Ensure input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    # Load data
    df_raw = pd.read_csv(input_path)
    print(f"Loaded raw data from {input_path}, shape: {df_raw.shape}")

    # Run processing steps
    df_cleaned = clean_data(df_raw, params)
    df_final = create_features_for_db(df_cleaned)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the final dataset
    df_final.to_csv(output_path, index=False)
    print(f"--- Pipeline Complete. Processed data saved to: '{output_path}' ---")


# Example of how to run this module directly for testing
if __name__ == '__main__':
    # This block allows you to run this script directly to test the full pipeline.
    # In production, DVC or Airflow would call the `process_and_save_features` function.
    
    # --- Configuration for the test run ---
    input_path = params['input_path']
    output_path = params['output_path']
    params = {'outlier_lap_time_percentage': params['outlier_lap_time_percentage']}

    process_and_save_features(
        input_path=input_path,
        output_path=output_path,
        params=params
    )
