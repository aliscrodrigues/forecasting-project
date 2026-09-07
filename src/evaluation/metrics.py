import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error (vectorized)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error (vectorized)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pct_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute percentage bias: 100 * sum(pred - true) / sum(true)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    total_true = float(np.sum(y_true))
    if total_true == 0:
        return float("nan")
    return float(100.0 * np.sum(y_pred - y_true) / total_true)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, RMSE, MAE/mean, and percentage bias."""
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mae_over_mean": float("nan"),
            "pct_bias": float("nan"),
        }

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    y_mean = float(np.mean(y_true))
    mae_over_mean = mae_val / y_mean if y_mean else float("nan")
    return {
        "mae": mae_val,
        "rmse": rmse_val,
        "mae_over_mean": mae_over_mean,
        "pct_bias": pct_bias(y_true, y_pred),
    }


def evaluate_predictions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Evaluate multiple models on the same test targets."""
    rows: list[dict[str, float | str]] = []
    for model_name, y_pred in predictions.items():
        metrics = evaluate(y_true, y_pred)
        rows.append({"model": model_name, **metrics})
    return pd.DataFrame(rows)


def format_metrics_table(metrics_df: pd.DataFrame) -> str:
    """Format model metrics as a readable table ordered by MAE."""
    ordered = metrics_df.sort_values("mae", ascending=True).reset_index(drop=True)
    lines = [f"{'model':<20} {'mae':>10} {'rmse':>10} {'mae/mean':>10} {'%bias':>10}"]
    for _, row in ordered.iterrows():
        lines.append(
            f"{row['model']:<20} "
            f"{row['mae']:>10.2f} "
            f"{row['rmse']:>10.2f} "
            f"{row['mae_over_mean']:>10.4f} "
            f"{row['pct_bias']:>9.2f}%"
        )
    return "\n".join(lines)
