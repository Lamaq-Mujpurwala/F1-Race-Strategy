# F1 Race Strategy Optimizer 🏎️

#Streamlit App - https://f1-race-strategy-nwqyq2z2dzwc5qdzwwtttw.streamlit.app/

This project aims to predict the optimal race strategy for a Formula 1 race by modeling tire degradation. The system uses historical race data to train an XGBoost model, which is then served via an API. A full CI/CD pipeline automates the weekly retraining and deployment process.

---

## ✨ Tech Stack

* **Data Analysis & Versioning**: `pandas`, `numpy`, `FastF1`, `DVC`, `Dagshub`
* **Model Training & Tracking**: `Scikit-learn`, `XGBoost`, `MLflow`
* **Orchestration & Development Environment**: `Docker`, `Docker Compose`, `Airflow`
* **Deployment & CI/CD**: `Docker`, `Huggingface Spaces`, `GitHub Actions`

---

## 🏗️ Project Architecture

1.  **Exploratory Data Analysis (EDA)**: Initial analysis was performed using pandas, numpy, and FastF1 to understand data trends and build hypotheses for the model.
2.  **ETL Pipeline**: A data pipeline was built using DVC to handle data versioning.
3.  **Orchestrated Model Training**: A production-ready Airflow container, managed with Docker Compose, is used for developing and running model training DAGs. An XGBoost model is trained using scikit-learn, and MLflow is used to log experiments and register model versions on Dagshub.
4.  **Inference API Deployment**: A Dockerfile is used to create an inference API that fetches the production-ready model from the Dagshub/MLflow registry. This API is deployed on Huggingface Spaces.
5.  **Automated CI/CD**: A full CI/CD pipeline using GitHub Actions runs every Monday. It automatically fetches new data, retrains the model, promotes the new version to the registry if performance improves, and restarts the API endpoint on Huggingface.

---


## 📁 Project Structure

```
f1-strategy-optimizer/
│
├── .github/                    # GitHub Actions workflows for CI/CD
│   └── workflows/
│
├── Data/                       # Data folder tracked using DVC and Dagshub
├── Dags/                       # Airflow DAGs for orchestrating ML workflows
├── EDA/                        # Exploratory Data Analysis notebooks and scripts
├── F1-strategy-api/            # Inference API files deployed on Huggingface Spaces
├── Src/                        # Source code for local development
│
├── docker-compose.yml          # Docker Compose file for Airflow local setup
├── dvc.yaml                    # DVC pipeline stages configuration
├── dvc-setup-file.txt          # Instructions for setting up DVC inside Airflow container
├── params.yaml                 # Parameters for data processing and model training
├── requirements.txt            # Core dependencies for the project
├── requirements-data.txt       # Dependencies specific to the data pipeline
├── requirements-model.txt      # Dependencies specific to the model pipeline
└── template_env.txt            # Template for environment variable configuration
```
