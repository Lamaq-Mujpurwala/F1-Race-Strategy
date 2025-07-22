from __future__ import annotations

import pendulum
import pandas as pd
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
import os

# Import the data processing functions from our source code
from src.processing.preprocessing import clean_data, create_features_for_db

# --- Configuration ---
MASTER_CSV_PATH = "/opt/airflow/data/combined/all_laps.csv"
POSTGRES_TABLE_NAME = "clean_lap_data"
MODEL_TRAINING_DAG_ID = "model_training_pipeline"


@dag(
    dag_id="data_ingestion_pipeline",
    schedule=None,
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["data-processing", "local-data"],
    doc_md="""
    ### Data Ingestion Pipeline (Manual SQL)
    This DAG reads a local CSV, processes it, manually creates a table in PostgreSQL,
    and then inserts the data.
    """,
)
def data_ingestion_pipeline():
    """
    This pipeline defines the tasks for manually ingesting and processing F1 data.
    """

    @task
    def process_data_from_csv():
        """
        Reads the local CSV and processes it into a clean DataFrame.
        """
        print(f"Reading data from fixed path: {MASTER_CSV_PATH}...")
        if not os.path.exists(MASTER_CSV_PATH):
            raise FileNotFoundError(f"Master CSV file not found at: {MASTER_CSV_PATH}.")
            
        df = pd.read_csv(MASTER_CSV_PATH)
        params = {"outlier_lap_time_percentage": 1.08}
        df_cleaned = clean_data(df, params)
        df_final = create_features_for_db(df_cleaned)
        
        # Convert DataFrame to a list of tuples for insertion
        return [tuple(x) for x in df_final.to_numpy()]

    @task
    def create_table_in_postgres():
        """
        Connects to Postgres and creates the target table, dropping it first if it exists.
        """
        print(f"Ensuring table '{POSTGRES_TABLE_NAME}' exists...")
        hook = PostgresHook(postgres_conn_id="postgres_default")
        
        # SQL to drop the table if it exists and then create it with the correct schema
        create_table_sql = f"""
            DROP TABLE IF EXISTS {POSTGRES_TABLE_NAME};
            CREATE TABLE {POSTGRES_TABLE_NAME} (
                LapTimeinSeconds FLOAT,
                TyreLife INT,
                LapNumber INT,
                Compound VARCHAR(20),
                Track VARCHAR(50),
                Year INT,
                Driver VARCHAR(10),
                AirTemp FLOAT,
                TrackTemp FLOAT
            );
        """
        hook.run(create_table_sql)
        print(f"Table '{POSTGRES_TABLE_NAME}' created successfully.")

    @task
    def insert_data_to_postgres(data_to_insert: list):
        """
        Inserts the processed data into the PostgreSQL table.
        """
        if not data_to_insert:
            print("No data to insert. Skipping.")
            return

        print(f"Inserting {len(data_to_insert)} rows into '{POSTGRES_TABLE_NAME}'...")
        hook = PostgresHook(postgres_conn_id="postgres_default")
        
        # Define the target columns for the insert operation
        target_fields = [
            'LapTimeinSeconds', 'TyreLife', 'LapNumber', 'Compound',
            'Track', 'Year', 'Driver', 'AirTemp', 'TrackTemp'
        ]
        
        # Use the hook's insert_rows method for efficient bulk insertion
        hook.insert_rows(
            table=POSTGRES_TABLE_NAME,
            rows=data_to_insert,
            target_fields=target_fields
        )
        print("Data insertion complete.")

    trigger_model_training = TriggerDagRunOperator(
        task_id="trigger_model_training_dag",
        trigger_dag_id=MODEL_TRAINING_DAG_ID,
        wait_for_completion=False,
    )

    # --- Define Task Dependencies ---
    processed_data = process_data_from_csv()
    table_creation_task = create_table_in_postgres()
    
    # The insert task depends on both the data being processed and the table being ready
    insertion_task = insert_data_to_postgres(processed_data)
    
    table_creation_task >> insertion_task
    processed_data >> insertion_task
    
    insertion_task >> trigger_model_training


# Instantiate the DAG
data_ingestion_pipeline()
