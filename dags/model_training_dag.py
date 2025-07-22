from __future__ import annotations

import pendulum
import pandas as pd
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Import the main training function from our source code.
# This works because src/ is copied into our Docker image.
from src.model.train import train_model

# --- Configuration ---
POSTGRES_TABLE_NAME = "clean_lap_data"
# This list controls which models will be trained. Add or remove model names here.
MODELS_TO_TRAIN = ['ridge', 'random_forest', 'xgboost']

@dag(
    dag_id="model_training_pipeline",
    schedule=None,  # This DAG is designed to be triggered by the data_ingestion_pipeline
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["model-training", "mlflow"],
    doc_md="""
    ### Model Training Pipeline
    This DAG fetches the clean data from PostgreSQL and trains multiple regression models in parallel.
    Each model's parameters, metrics, and artifacts (preprocessor, model file) are logged
    to a single MLflow experiment on DagsHub for easy comparison.
    """,
)
def model_training_pipeline():
    """
    This pipeline defines the tasks for training and evaluating our models.
    """

    @task
    def get_data_from_postgres() -> pd.DataFrame:
        """
        Connects to Postgres and fetches the entire clean_lap_data table.
        """
        print(f"Fetching data from PostgreSQL table: {POSTGRES_TABLE_NAME}")
        hook = PostgresHook(postgres_conn_id="postgres_default")
        df = hook.get_pandas_df(sql=f"SELECT * FROM {POSTGRES_TABLE_NAME}")
        print(f"Successfully fetched {len(df)} rows.")
        return df

    # --- Dynamic Task Generation ---
    # This is the core of our experimental setup. We create a separate, parallel
    # training task for each model defined in the MODELS_TO_TRAIN list.
    
    # The first task fetches the data.
    clean_data_df = get_data_from_postgres()

    # We then loop through our list of models.
    for model_name in MODELS_TO_TRAIN:
        # The @task decorator is used to create a unique task for each model.
        # The task_id is dynamically generated based on the model's name.
        @task(task_id=f"train_{model_name}_model")
        def train_specific_model(df: pd.DataFrame, model_to_train: str):
            """
            A dynamically generated task that calls our main training script
            for a specific model.
            """
            # This is where we call the orchestrator function from our src/model/train.py script
            train_model(data=df, model_name=model_to_train)

        # Define the dependency: each training task depends on the data fetching task.
        train_specific_model(clean_data_df, model_name)

# Instantiate the DAG
model_training_pipeline()
