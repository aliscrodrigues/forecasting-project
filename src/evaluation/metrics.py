import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error (vectorized)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error (vectorized)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a summary of evaluation metrics."""
    print("Evaluating model performance...")
    if y_true.size == 0 or y_pred.size == 0:
        print("  → Cannot evaluate (empty arrays)")
        return {"mae": float("nan"), "rmse": float("nan")}
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    print(f"  → MAE:  {mae_val:.4f}")
    print(f"  → RMSE: {rmse_val:.4f}")
    return {"mae": mae_val, "rmse": rmse_val}
