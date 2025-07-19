# F1 Race Strategy Optimizer 🏎️

This project aims to predict the optimal race strategy for a Formula 1 race by modeling tire degradation, pit stop deltas, and other race variables. The system uses historical data to train a suite of models and an optimization algorithm to recommend the best pit stop laps and tire choices to minimize total race time.

## ✨ Tech Stack

* **Data Collection & Versioning**: `FastF1`, `Python`, `DVC`, `Dagshub`
* **Data Orchestration & Warehousing**: `Docker`, `Astro CLI`, `Airflow`, `PostgreSQL`
* **Model Training & Tracking**: `Scikit-learn`, `XGBoost`, `MLflow`
* **Deployment & CI/CD**: `Flask`, `Render`, `GitHub Actions`
* **Frontend**: `Streamlit`

## 🏗️ Project Architecture

1.  **ETL Pipeline**: A Python script using **FastF1** fetches historical race data. **DVC** versions this data and pushes it to **Dagshub**.
2.  **Data Ingestion**: An **Airflow** DAG, running in a Docker container managed by **Astro CLI**, orchestrates the data pipeline. It extracts raw data, transforms it into features, and loads it into a **PostgreSQL** database.
3.  **Model Building**: A training script fetches feature data from Postgres, trains the prediction models, and logs experiments, parameters, and artifacts to **MLflow**.
4.  **Backend API**: A **Flask** application loads the best model from the MLflow registry and exposes an API endpoint for inference. This API is containerized and deployed on **Render**.
5.  **CI/CD**: **GitHub Actions** automates the entire process. On a push to `main`, it runs tests, triggers the model training pipeline, and if a new model is promoted, it deploys the updated Flask API to Render.
6.  **Frontend**: A **Streamlit** application provides a user-friendly interface to interact with the model by calling the deployed Flask API.

## 📁 Project Structure

f1-strategy-optimizer/
│
├── .dvc/                   # DVC metadata
├── .github/
│   └── workflows/          # GitHub Actions CI/CD workflows
│       └── main.yml
│
├── data/
│   ├── raw/                # Raw data pulled by FastF1 (tracked by DVC)
│   │   └── race_data.csv.dvc
│   └── processed/          # Processed data for modeling (tracked by DVC)
│       └── features.csv.dvc
│
├── dags/                     # Airflow DAG definitions
│   └── etl_dag.py
│
├── src/
│   ├── api/                # Flask API
│   │   ├── app.py
│   │   └── Dockerfile
│   │
│   ├── model/              # Model training and evaluation scripts
│   │   ├── train.py
│   │   └── predict.py
│   │
│   ├── data/               # Data collection and processing scripts
│   │   └── make_dataset.py
│   │
│   └── frontend/           # Streamlit UI
│       └── app.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile              # For Airflow environment (managed by Astro)
├── packages.txt            # OS packages for Docker
├── requirements.txt        # Python dependencies
├── README.md               # This file

