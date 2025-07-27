import mlflow
import os
import yaml

# --- Configuration ---
PARAMS_FILE = "params.yaml"
MLFLOW_MODEL_NAME = "tire_degradation_model_v1"
# We define the STAGE we want to transition to, which DagsHub will map to an ALIAS
PROMOTION_STAGE = "Production" 
PRIMARY_METRIC = "metrics.mae" 
COMPARISON_METRICS = ["metrics.mae", "metrics.r2_score"]

def promote_best_model():
    """
    Compares newly trained models against the current production model.
    If a new model is better, it is registered and then promoted by
    transitioning its stage to 'Production', which sets the 'production' alias on DagsHub.
    """
    print("--- Starting Automated Model Promotion ---")
    
    client = mlflow.tracking.MlflowClient()

    # 1. Find the best of the newly trained models from the most recent experiment
    print("Searching for the best newly trained model...")
    experiment = client.get_experiment_by_name("Default")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"{PRIMARY_METRIC} ASC"] # Lower MAE is better
    )
    
    if runs.empty:
        print("No new runs found. Exiting.")
        return

    best_new_run = runs.iloc[0]
    best_new_run_id = best_new_run["run_id"]
    best_new_metrics = {metric: best_new_run[metric] for metric in COMPARISON_METRICS}
    
    print(f"Best new model is in Run ID: {best_new_run_id}")
    print(f"  - New Model MAE: {best_new_metrics['metrics.mae']:.4f}")
    print(f"  - New Model R2 Score: {best_new_metrics['metrics.r2_score']:.4f}")

    # 2. Get the current production model's metrics using the 'stages' parameter
    try:
        # This is the compatible way to find the current production model
        prod_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[PROMOTION_STAGE])
        if prod_versions:
            prod_model_version = prod_versions[0]
            prod_run_id = prod_model_version.run_id
            prod_run = client.get_run(prod_run_id)
            prod_metrics = prod_run.data.metrics
            
            print(f"Found current production model: Version {prod_model_version.version}, Run ID: {prod_run_id}")
            print(f"  - Production MAE: {prod_metrics.get('mae', float('inf')):.4f}")
            print(f"  - Production R2 Score: {prod_metrics.get('r2_score', float('-inf')):.4f}")
        else:
            raise Exception("No versions found in Production stage.")

    except Exception as e:
        print(f"No existing production model found. The new model will be promoted automatically. Error: {e}")
        prod_metrics = {"mae": float('inf'), "r2_score": float('-inf')}

    # 3. Compare and decide whether to promote
    new_model_is_better = (
        best_new_metrics['metrics.mae'] < prod_metrics.get('mae', float('inf')) and
        best_new_metrics['metrics.r2_score'] > prod_metrics.get('r2_score', float('-inf'))
    )

    if new_model_is_better:
        print("\nNew model is better than the current production model. Promoting...")
        
        # --- THE FIX ---
        # 4. Register the model from the best run's artifact URI using a low-level client call.
        model_uri = f"runs:/{best_new_run_id}/model"
        print(f"Registering new model version from URI: {model_uri}")
        
        # This creates a new version of the model.
        new_version = client.create_model_version(
            name=MLFLOW_MODEL_NAME,
            source=model_uri,
            run_id=best_new_run_id
        )
        print(f"Model successfully registered as Version {new_version.version}.")

        # 5. Transition the new version to the 'Production' stage.
        # This is the compatible API call that DagsHub supports, and it will
        # result in the 'production' alias being set in the UI.
        print(f"Transitioning Version {new_version.version} to stage '{PROMOTION_STAGE}'...")
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=new_version.version,
            stage=PROMOTION_STAGE,
            archive_existing_versions=True # This moves the old production model to 'Archived'
        )
        print(f"Successfully promoted Version {new_version.version} to '{PROMOTION_STAGE}'.")

    else:
        print("\nNew model is not better than the current production model. No promotion will occur.")

if __name__ == "__main__":
    promote_best_model()
