import pandas as pd
import yaml
import os
import mlflow

# Import the main training function from our source code.
# We need to add src to the path for the import to work when run from the root.
import sys
sys.path.append('src')
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

    # 2. Loop through and train each model
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
