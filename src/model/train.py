import pandas as pd
import mlflow
import yaml
from sklearn.model_selection import train_test_split
import joblib

# Import our custom modules
from model.preprocessing import create_preprocessor, fit_and_save_preprocessor
from model.models import MODEL_GETTERS
from model.evaluate import get_regression_metrics

# --- Configuration ---
PARAMS_FILE = "params.yaml" # This should be accessible in the Airflow environment
PREPROCESSOR_FILENAME = "preprocessor.joblib"
MODEL_FILENAME = "model.joblib"

def train_model(data: pd.DataFrame, model_name: str):
    """
    Main function to orchestrate a single model training run.

    Args:
        data (pd.DataFrame): The clean input DataFrame from PostgreSQL.
        model_name (str): The name of the model to train (e.g., 'xgboost').
    """
    print(f"--- Starting training run for model: {model_name} ---")

    # 1. Load parameters from the YAML file
    with open(PARAMS_FILE) as f:
        params = yaml.safe_load(f)
    
    model_params = params['models'][model_name]
    target_variable = params['base']['target_variable']
    
    # 2. Start MLflow Run
    # This will log to the DagsHub server configured in our .env file
    with mlflow.start_run(run_name=f"{model_name}_training_run") as run:
        mlflow.log_params(model_params)
        print(f"MLflow run started. Run ID: {run.info.run_id}")

        # 3. Separate Target and Features & Perform Train-Test Split
        X = data.drop(target_variable, axis=1)
        y = data[target_variable]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=params['base']['test_size'], 
            random_state=params['base']['random_state']
        )
        print("Train-test split complete.")

        # 4. Preprocessing: Fit and save the preprocessor
        preprocessor = create_preprocessor(
            X_train,
            categorical_features=params['features']['categorical'],
            numerical_features=params['features']['numerical']
        )
        
        fitted_preprocessor = fit_and_save_preprocessor(preprocessor, X_train, PREPROCESSOR_FILENAME)
        mlflow.log_artifact(PREPROCESSOR_FILENAME)
        print("Preprocessor fitted, saved, and logged to MLflow.")

        # 5. Transform the data using the fitted preprocessor
        X_train_transformed = fitted_preprocessor.transform(X_train)
        X_test_transformed = fitted_preprocessor.transform(X_test)
        print("Data transformation complete.")

        # 6. Get and Train the Model
        model_getter = MODEL_GETTERS[model_name]
        model = model_getter(model_params)
        
        print(f"Training {model_name} model...")
        # For XGBoost, we can use the validation set for early stopping
        if model_name == 'xgboost':
            model.fit(X_train_transformed, y_train, eval_set=[(X_test_transformed, y_test)], verbose=False)
        else:
            model.fit(X_train_transformed, y_train)
        print("Model training complete.")

        # 7. Evaluate the Model and Log Metrics
        y_pred = model.predict(X_test_transformed)
        metrics = get_regression_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        print("Metrics calculated and logged to MLflow.")

        # 8. Log the trained model itself
        joblib.dump(model, MODEL_FILENAME)
        mlflow.log_artifact(MODEL_FILENAME)
        print("Model artifact saved and logged to MLflow.")

    print(f"--- Training run for {model_name} complete. ---")

# Example of how to run this script (for local testing)
# In production, Airflow will call the train_model function directly.
if __name__ == '__main__':
    # This is a placeholder for local testing.
    # You would need to have a 'params.yaml' file and a data source.
    print("This script is intended to be called by an Airflow DAG.")
