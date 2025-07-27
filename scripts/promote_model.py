import mlflow
import os
import yaml

# --- Configuration ---
PARAMS_FILE = "params.yaml"
MLFLOW_MODEL_NAME = "tire_degradation_model_v1"
PROMOTION_ALIAS = "production"
# The metric to sort by to find the best of the newly trained models
PRIMARY_METRIC = "metrics.mae" 
# The metrics to compare against the current production model
COMPARISON_METRICS = ["metrics.mae", "metrics.r2_score"]

def promote_best_model():
    """
    Compares the newly trained models against the current production model.
    If a new model is better based on the defined metrics, it is promoted
    by setting its alias to 'production'.
    """
    print("--- Starting Automated Model Promotion ---")
    
    client = mlflow.tracking.MlflowClient()

    # 1. Find the best of the newly trained models from the most recent experiment
    print("Searching for the best newly trained model...")
    # We assume the default experiment 'Default' is used
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

    # 2. Get the current production model's metrics
    try:
        prod_model_version = client.get_model_version_by_alias(name=MLFLOW_MODEL_NAME, alias=PROMOTION_ALIAS)
        prod_run_id = prod_model_version.run_id
        prod_run = client.get_run(prod_run_id)
        prod_metrics = prod_run.data.metrics
        
        print(f"Found current production model: Version {prod_model_version.version}, Run ID: {prod_run_id}")
        print(f"  - Production MAE: {prod_metrics.get('mae', float('inf')):.4f}")
        print(f"  - Production R2 Score: {prod_metrics.get('r2_score', float('-inf')):.4f}")

    except Exception as e:
        print(f"No existing production model found. The new model will be promoted automatically. Error: {e}")
        prod_metrics = {"mae": float('inf'), "r2_score": float('-inf')} # Set values that guarantee promotion

    # 3. Compare and decide whether to promote
    # We promote if the new model has a lower MAE AND a higher R2 Score.
    new_model_is_better = (
        best_new_metrics['metrics.mae'] < prod_metrics.get('mae', float('inf')) and
        best_new_metrics['metrics.r2_score'] > prod_metrics.get('r2_score', float('-inf'))
    )

    if new_model_is_better:
        print("\nNew model is better than the current production model. Promoting...")
        
        # Find the registered model version associated with the best new run
        new_model_version = None
        for mv in client.search_model_versions(f"run_id='{best_new_run_id}'"):
            if mv.name == MLFLOW_MODEL_NAME:
                new_model_version = mv.version
                break
        
        if new_model_version:
            # Set the 'production' alias for the new version
            client.set_registered_model_alias(
                name=MLFLOW_MODEL_NAME,
                alias=PROMOTION_ALIAS,
                version=new_model_version
            )
            print(f"Successfully promoted Version {new_model_version} to '{PROMOTION_ALIAS}'.")
        else:
            print(f"Error: Could not find a registered model version for Run ID {best_new_run_id}.")

    else:
        print("\nNew model is not better than the current production model. No promotion will occur.")

if __name__ == "__main__":
    # This function name was incorrect in the original file, corrected here.
    promote_best_model()
