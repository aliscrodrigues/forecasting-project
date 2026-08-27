import numpy as np
import pandas as pd


def _extract_xy(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(data[feature_columns].values, dtype=np.float64)
    y = np.asarray(data[target_column].values, dtype=np.float64)
    return X, y


def time_based_split(
    data: pd.DataFrame,
    date_column: str,
    feature_columns: list[str],
    target_column: str,
    train_end: str,
    val_end: str,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Split data by time cutoffs and return (X, y) NumPy arrays per partition.

    - Train: dates <= train_end
    - Validation: train_end < dates <= val_end
    - Test: dates > val_end
    """
    print(
        f"Time-based split (train until {train_end}, validation until {val_end})..."
    )
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train_df = data[data[date_column] <= train_end_dt]
    val_df = data[
        (data[date_column] > train_end_dt) & (data[date_column] <= val_end_dt)
    ]
    test_df = data[data[date_column] > val_end_dt]

    train = _extract_xy(train_df, feature_columns, target_column)
    val = _extract_xy(val_df, feature_columns, target_column)
    test = _extract_xy(test_df, feature_columns, target_column)

    print(
        f"  → Train: X={train[0].shape}, y={train[1].shape} | "
        f"Val: X={val[0].shape}, y={val[1].shape} | "
        f"Test: X={test[0].shape}, y={test[1].shape}"
    )
    return train, val, test
