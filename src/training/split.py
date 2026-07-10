import numpy as np
import pandas as pd


def temporal_split(
    data: pd.DataFrame,
    date_column: str,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets using temporal cutoffs.

    - Train: dates <= train_end
    - Validation: train_end < dates <= val_end
    - Test: dates > val_end
    """
    print(
        f"Splitting data with temporal split "
        f"(train until {train_end}, validation until {val_end})..."
    )
    train_end_dt = pd.Timestamp(train_end)
    val_end_dt = pd.Timestamp(val_end)

    train = data[data[date_column] <= train_end_dt].copy()
    val = data[
        (data[date_column] > train_end_dt) & (data[date_column] <= val_end_dt)
    ].copy()
    test = data[data[date_column] > val_end_dt].copy()

    print(f"  → Train: {len(train)} | Validation: {len(val)} | Test: {len(test)}")
    return train, val, test


def split_features_target(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and target vector as NumPy arrays."""
    print(f"Splitting features ({len(feature_columns)} cols) and target ({target_column})...")
    X = np.asarray(data[feature_columns].values, dtype=np.float64)
    y = np.asarray(data[target_column].values, dtype=np.float64)
    print(f"  → X: {X.shape}, y: {y.shape}")
    return X, y
