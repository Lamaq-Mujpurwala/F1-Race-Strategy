import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict

def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates a set of standard regression metrics.

    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted values from the model.

    Returns:
        Dict[str, float]: A dictionary containing the calculated metrics.
    """
    print("Calculating evaluation metrics...")
    
    # Mean Absolute Error (MAE) - The average absolute difference between prediction and actual.
    # Good for understanding the average error in the original units (seconds).
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Squared Error (MSE) - Penalizes larger errors more heavily.
    mse = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error (RMSE) - The square root of MSE, putting it back in original units.
    rmse = np.sqrt(mse)
    
    # R-squared (R2) Score - The proportion of the variance in the target that is predictable
    # from the features. A score of 1.0 is perfect.
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2
    }
    
    print(f"Metrics: {metrics}")
    return metrics

