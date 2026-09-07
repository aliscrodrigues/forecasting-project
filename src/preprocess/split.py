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


def _normalize_month(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalize a date-like value to the first day of its calendar month."""
    return pd.Timestamp(value).to_period("M").to_timestamp()


def time_based_split(
    data: pd.DataFrame,
    date_column: str,
    feature_columns: list[str],
    target_column: str,
    reference_month: str,
    validation_months: int,
    forecast_horizon: int = 1,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    pd.Series,
]:
    """Split data by forecast target month.

    Returns ``(train, val, test, test_target_dates)`` where each partition is
    ``(X, y)`` NumPy arrays and ``test_target_dates`` is the target-month Series
    for test rows (same order as ``y_test``).

    ``reference_month`` is the forecast target month of interest. ``validation_months``
    is the number of calendar months immediately before that reference used for
    validation. Partitions are assigned using each row's target date
    (origin date + ``forecast_horizon`` months):

    - Train: target month < validation window start
    - Validation: validation window start <= target month < reference month
    - Test: target month >= reference month
    """
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1.")
    if validation_months < 1:
        raise ValueError("validation_months must be at least 1.")

    reference_dt = _normalize_month(reference_month)
    validation_start = reference_dt - pd.DateOffset(months=validation_months)

    working = data.copy()
    origin_dates = pd.to_datetime(working[date_column])
    target_dates = origin_dates + pd.DateOffset(months=forecast_horizon)

    train_mask = target_dates < validation_start
    val_mask = (target_dates >= validation_start) & (target_dates < reference_dt)
    test_mask = target_dates >= reference_dt

    train_df = working.loc[train_mask]
    val_df = working.loc[val_mask]
    test_df = working.loc[test_mask]

    train = _extract_xy(train_df, feature_columns, target_column)
    val = _extract_xy(val_df, feature_columns, target_column)
    test = _extract_xy(test_df, feature_columns, target_column)
    test_target_dates = target_dates[test_mask]
    return train, val, test, test_target_dates
