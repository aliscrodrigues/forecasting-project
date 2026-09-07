import numpy as np
import pandas as pd

SKU_COLUMN = "item_id"
DATE_COLUMN = "date"
TARGET_COLUMN = "sales"
MONTH_COLUMN = "month"
TARGET_AHEAD_COLUMN = "target"

CALENDAR_FEATURE_COLUMNS = (
    "month_sin",
    "month_cos",
    "momentum_1m",
    "yoy_change",
)


def preprocess_series(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sales into monthly time series per SKU across all stores."""
    data = data.copy()
    data[MONTH_COLUMN] = data[DATE_COLUMN].dt.to_period("M").dt.to_timestamp()
    monthly = (
        data.groupby([SKU_COLUMN, MONTH_COLUMN])[TARGET_COLUMN].sum().reset_index()
    )
    monthly = monthly.sort_values([SKU_COLUMN, MONTH_COLUMN]).reset_index(drop=True)
    return monthly


def _lag_column_names(lags: tuple[int, ...]) -> list[str]:
    return [f"lag_{lag}" for lag in lags]


def _rolling_column_names(rolling_windows: tuple[int, ...]) -> list[str]:
    return [f"rolling_mean_{window}" for window in rolling_windows]


def _add_lag_features(
    monthly: pd.DataFrame,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    """Add per-SKU lag features relative to the origin month.

    ``lag_0`` is demand at the origin month (``shift(0)``). For horizon 1,
    ``lag_0`` is the month before the target and is not the same as ``target``.
    """
    monthly = monthly.copy()
    for lag in lags:
        monthly[f"lag_{lag}"] = monthly.groupby(SKU_COLUMN)[TARGET_COLUMN].shift(lag)
    return monthly


def _add_derived_features(
    monthly: pd.DataFrame,
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    """Add rolling means and simple calendar/momentum features."""
    monthly = monthly.copy()
    grouped = monthly.groupby(SKU_COLUMN)[TARGET_COLUMN]

    for window in rolling_windows:
        monthly[f"rolling_mean_{window}"] = grouped.transform(
            lambda values, w=window: values.rolling(w, min_periods=w).mean()
        )

    month = monthly[MONTH_COLUMN].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * month / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * month / 12)
    if "lag_0" in monthly.columns:
        monthly["momentum_1m"] = monthly["lag_0"] - monthly["lag_1"]
        monthly["yoy_change"] = monthly["lag_0"] - monthly["lag_12"]
    else:
        monthly["momentum_1m"] = monthly["lag_1"] - monthly["lag_2"]
        monthly["yoy_change"] = monthly["lag_1"] - monthly["lag_12"]
    return monthly


def build_dataset(
    monthly: pd.DataFrame,
    lags: tuple[int, ...],
    rolling_windows: tuple[int, ...],
    horizon: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """Build tabular features and a one-step-ahead target."""
    lag_columns = _lag_column_names(lags)
    rolling_columns = _rolling_column_names(rolling_windows)
    df = _add_lag_features(monthly, lags)
    df = _add_derived_features(df, rolling_windows)
    feature_columns = lag_columns + rolling_columns + list(CALENDAR_FEATURE_COLUMNS)
    df[TARGET_AHEAD_COLUMN] = df.groupby(SKU_COLUMN)[TARGET_COLUMN].shift(-horizon)
    df = df.dropna().reset_index(drop=True)
    return df, feature_columns
