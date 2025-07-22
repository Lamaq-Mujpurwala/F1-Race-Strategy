import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import List, Any

# Define which columns are categorical and which are numerical
CATEGORICAL_FEATURES = ['Compound', 'Track', 'Year', 'Driver']
NUMERICAL_FEATURES = ['TyreLife', 'LapNumber', 'AirTemp', 'TrackTemp']
PREPROCESSOR_FILENAME = "preprocessor.joblib"

def create_preprocessor(
    df: pd.DataFrame, 
    categorical_features: List[str], 
    numerical_features: List[str]
) -> ColumnTransformer:
    """
    Creates a ColumnTransformer to handle both categorical and numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame to determine feature types.
        categorical_features (List[str]): List of column names to be one-hot encoded.
        numerical_features (List[str]): List of column names to be passed through.

    Returns:
        ColumnTransformer: An unfitted ColumnTransformer object.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='drop' # Drop any columns not specified
    )
    return preprocessor

def fit_and_save_preprocessor(
    preprocessor: ColumnTransformer, 
    X_train: pd.DataFrame, 
    output_path: str
) -> ColumnTransformer:
    """
    Fits the preprocessor on the training data and saves it to a file.

    Args:
        preprocessor (ColumnTransformer): The ColumnTransformer to fit.
        X_train (pd.DataFrame): The training data DataFrame.
        output_path (str): The path to save the fitted preprocessor artifact.

    Returns:
        ColumnTransformer: The fitted preprocessor object.
    """
    print("Fitting the preprocessor on the training data...")
    preprocessor.fit(X_train)
    
    print(f"Saving preprocessor artifact to {output_path}")
    joblib.dump(preprocessor, output_path)
    
    return preprocessor

# This is a helper function that your Airflow task in train.py would call
def full_preprocessing_pipeline(X_train: pd.DataFrame):
    """
    Orchestrates the creation, fitting, and saving of the preprocessor.
    This would be called from your main training script.
    """
    # 1. Create the preprocessor object
    preprocessor_obj = create_preprocessor(X_train, CATEGORICAL_FEATURES, NUMERICAL_FEATURES)

    # 2. Fit it on the training data and save the artifact
    fitted_preprocessor = fit_and_save_preprocessor(
        preprocessor_obj, 
        X_train, 
        PREPROCESSOR_FILENAME
    )
    
    # In a real MLflow run, you would now log PREPROCESSOR_FILENAME as an artifact
    # mlflow.log_artifact(PREPROCESSOR_FILENAME)
    
    print("Preprocessing pipeline complete.")
    return fitted_preprocessor

