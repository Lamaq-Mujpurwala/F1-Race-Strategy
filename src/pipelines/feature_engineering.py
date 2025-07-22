import pandas as pd
import os

# Import the functions from our pipeline module
from src.pipelines.data_processing.nodes import clean_data, create_features_for_db

# --- Configuration ---
INTERIM_DATA_DIR = 'data/interim'
PROCESSED_DATA_DIR = 'data/processed'
MASTER_FILENAME = 'all_laps_master.csv'
OUTPUT_FILENAME = 'features_for_model.csv'

def test_pipeline():
    """
    Tests the data cleaning and feature selection pipeline on the master dataset.
    This simulates the steps that DVC will eventually automate.
    """
    print("--- Starting Feature Engineering Test ---")
    
    # 1. Load the master dataset
    master_file_path = os.path.join(INTERIM_DATA_DIR, MASTER_FILENAME)
    if not os.path.exists(master_file_path):
        print(f"Error: Master file not found at '{master_file_path}'.")
        print("Please run 'scripts/combine_datasets.py' first.")
        return
        
    print(f"Loading master data from: {master_file_path}")
    master_df = pd.read_csv(master_file_path)

    # 2. Define parameters (this would come from params.yaml in a DVC pipeline)
    pipeline_params = {
        'outlier_lap_time_percentage': 1.08
    }

    # 3. Run the cleaning step
    df_cleaned = clean_data(master_df, pipeline_params)

    # 4. Run the feature selection step
    df_final_features = create_features_for_db(df_cleaned)

    # 5. Save the final output for inspection
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILENAME)
    df_final_features.to_csv(output_path, index=False)
    
    print("\n--- Test Complete ---")
    print(f"Successfully created final feature dataset at: '{output_path}'")
    print("\nFinal Data Head:")
    print(df_final_features.head())
    print(f"\nFinal Data Shape: {df_final_features.shape}")

if __name__ == '__main__':
    # To run this script, navigate to your project root in the terminal
    # and execute: python scripts/test_feature_engineering.py
    test_pipeline()
