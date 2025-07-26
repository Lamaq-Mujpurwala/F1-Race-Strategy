import mlflow
import os
import joblib

# --- Configuration ---
MLFLOW_MODEL_NAME = "tire_degradation_model_v1"
MODEL_ALIAS = "production"
MODEL_ARTIFACT_NAME = "model.joblib"
PREPROCESSOR_ARTIFACT_NAME = "preprocessor.joblib"
    
class ModelLoader:
    """
    A dedicated class to handle loading model artifacts from the MLflow Model Registry.
    """
    def __init__(self):
        print("Initializing ModelLoader...")
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self):
        """
        Connects to the MLflow server, finds the model version with the 'production'
        alias, gets its run ID, and downloads the model and preprocessor artifacts.
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            # 1. Get the details of the model version aliased as 'production'
            print(f"Fetching model version with alias '{MODEL_ALIAS}' for model '{MLFLOW_MODEL_NAME}'...")
            model_version_details = client.get_model_version_by_alias(
                name=MLFLOW_MODEL_NAME, 
                alias=MODEL_ALIAS
            )
            run_id = model_version_details.run_id
            print(f"Found production model: Version {model_version_details.version}, Run ID: {run_id}")

            # 2. Download the preprocessor artifact from that run
            print(f"Downloading artifact: {PREPROCESSOR_ARTIFACT_NAME}")
            local_preprocessor_path = client.download_artifacts(
                run_id=run_id, 
                path=PREPROCESSOR_ARTIFACT_NAME
            )
            self.preprocessor = joblib.load(local_preprocessor_path)
            print("Preprocessor loaded successfully.")

            # 3. Download the model artifact from that same run
            print(f"Downloading artifact: {MODEL_ARTIFACT_NAME}")
            local_model_path = client.download_artifacts(
                run_id=run_id, 
                path=MODEL_ARTIFACT_NAME
            )
            self.model = joblib.load(local_model_path)
            print("Model loaded successfully.")
            
            print("--- ModelLoader initialization complete. ---")

        except Exception as e:
            print(f"FATAL: Error loading artifacts from MLflow Registry: {e}")
            print("The simulator will not be able to make predictions.")
            self.model = None
            self.preprocessor = None

# Create a single, global instance of the loader.
# The model will be loaded once when the application starts.
model_loader = ModelLoader()
