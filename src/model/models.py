from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from typing import Dict, Any

def get_ridge_model(params: Dict[str, Any]):
    """
    Creates a Ridge Regression model with specified parameters.
    """
    # alpha: The strength of the regularization. Higher values mean stronger regularization.
    model = Ridge(
        alpha=params.get('alpha', 1.0),
        random_state=params.get('random_state', 42)
    )
    return model

def get_random_forest_model(params: Dict[str, Any]):
    """
    Creates a Random Forest Regressor model with specified parameters.
    """
    # n_estimators: The number of trees in the forest.
    # max_depth: The maximum depth of each tree.
    # min_samples_leaf: The minimum number of samples required to be at a leaf node.
    # n_jobs=-1: Use all available CPU cores for training.
    model = RandomForestRegressor(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 10),
        min_samples_leaf=params.get('min_samples_leaf', 4),
        random_state=params.get('random_state', 42),
        n_jobs=-1
    )
    return model

def get_xgboost_model(params: Dict[str, Any]):
    """
    Creates an XGBoost Regressor model with specified parameters.
    """
    # CRITICAL FIX: Changed from XGBClassifier to XGBRegressor for this task.
    # n_estimators: Number of boosting rounds.
    # learning_rate: Step size shrinkage to prevent overfitting.
    # max_depth: Maximum depth of a tree.
    # subsample: Fraction of samples to be used for fitting the individual base learners.
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params.get('n_estimators', 1000),
        learning_rate=params.get('learning_rate', 0.05),
        max_depth=params.get('max_depth', 5),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        early_stopping_rounds=params.get('early_stopping_rounds', 50),
        random_state=params.get('random_state', 42),
        n_jobs=-1
    )
    return model

# A helper dictionary to easily access the model functions
MODEL_GETTERS = {
    'ridge': get_ridge_model,
    'random_forest': get_random_forest_model,
    'xgboost': get_xgboost_model
}
