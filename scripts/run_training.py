import pandas as pd
import yaml
import os
import mlflow

# Import the main training function from our source code.
from model.train import train_model

# --- Configuration ---
PARAMS_FILE = "params.yaml"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
MODELS_TO_TRAIN = ['ridge', 'random_forest', 'xgboost']

def run_full_training():
    """
    Main function to orchestrate the training of all specified models.
    This script is designed to be run by the CI/CD pipeline.
    """
    print("--- Starting Full Model Training and Registration Pipeline ---")

    # 1. Load the processed data from DVC
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Processed data not found at '{PROCESSED_DATA_PATH}'. "
            "Ensure the DVC data pipeline has run successfully first."
        )
    
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    data = pd.read_csv(PROCESSED_DATA_PATH)

    # --- THE FIX ---
    # 2. Standardize column names to lowercase to match params.yaml
    # This makes the pipeline robust to the casing of the input CSV.
    data.columns = [col.lower() for col in data.columns]
    print("Standardized DataFrame columns to lowercase.")

    # 3. Loop through and train each model
    for model_name in MODELS_TO_TRAIN:
        print(f"\n--- Triggering training for: {model_name} ---")
        try:
            # The train_model function will handle the MLflow logging and registration
            train_model(data=data, model_name=model_name)
        except Exception as e:
            print(f"!!! ERROR training model {model_name}: {e}")
            # In a real pipeline, you might want to continue or fail the whole job
            continue
    
    print("\n--- Full Model Training and Registration Pipeline Complete ---")

if __name__ == '__main__':
    # This allows us to run the script directly from the command line
    run_full_training()
