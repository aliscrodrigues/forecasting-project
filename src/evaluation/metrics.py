import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error (vectorized)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error (vectorized)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split: str = "",
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAE relative to mean demand."""
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mae_over_mean": float("nan"),
        }

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    y_mean = float(np.mean(y_true))
    mae_over_mean = mae_val / y_mean if y_mean else float("nan")
    return {"mae": mae_val, "rmse": rmse_val, "mae_over_mean": mae_over_mean}
