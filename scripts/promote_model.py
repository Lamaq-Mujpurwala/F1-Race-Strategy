import mlflow
import os
import yaml

# --- Configuration ---
PARAMS_FILE = "params.yaml"
MLFLOW_MODEL_NAME = "tire_degradation_model_v1"
# This is the alias we want our production model to have.
PROMOTION_ALIAS = "production" 
# On DagsHub's MLflow 2.2.0, setting the stage to "Production" is the
# compatible way to assign the "production" alias.
PROMOTION_STAGE_EQUIVALENT = "Production"
PRIMARY_METRIC = "metrics.mae" 
COMPARISON_METRICS = ["metrics.mae", "metrics.r2_score"]

def promote_best_model():
    """
    Compares newly trained models against the current production model.
    If a new model is better, it is registered and then promoted by
    transitioning its stage, which sets the 'production' alias on DagsHub.
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

    # 2. Get the current production model's metrics by finding which version has the alias
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
        prod_metrics = {"mae": float('inf'), "r2_score": float('-inf')}

    # 3. Compare and decide whether to promote
    new_model_is_better = (
        best_new_metrics['metrics.mae'] < prod_metrics.get('mae', float('inf')) and
        best_new_metrics['metrics.r2_score'] > prod_metrics.get('r2_score', float('-inf'))
    )

    if new_model_is_better:
        print("\nNew model is better than the current production model. Promoting...")
        
        # 4. Register the model from the best run's artifact URI
        model_uri = f"runs:/{best_new_run_id}/model"
        print(f"Registering new model version from URI: {model_uri}")
        
        # This creates a new version of the model (e.g., Version 4)
        new_version_details = client.create_model_version(
            name=MLFLOW_MODEL_NAME,
            source=model_uri,
            run_id=best_new_run_id
        )
        new_model_version = new_version_details.version
        print(f"Model successfully registered as Version {new_model_version}.")

        # 5. Promote the new version by transitioning its stage.
        # This is the compatible API call that DagsHub's MLflow 2.2.0 server
        # understands. It will result in the 'production' alias being set in the UI.
        print(f"Transitioning Version {new_model_version} to stage '{PROMOTION_STAGE_EQUIVALENT}' to set the '{PROMOTION_ALIAS}' alias...")
        client.transition_model_version_stage(
            name=MLFLOW_MODEL_NAME,
            version=new_model_version,
            stage=PROMOTION_STAGE_EQUIVALENT,
            archive_existing_versions=True # This moves the old production model to 'Archived'
        )
        print(f"Successfully promoted Version {new_model_version}.")

    else:
        print("\nNew model is not better than the current production model. No promotion will occur.")

if __name__ == "__main__":
    promote_best_model()
