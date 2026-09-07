import pandas as pd

SKU_COLUMN = "item_id"
DATE_COLUMN = "date"
TARGET_COLUMN = "sales"
MONTH_COLUMN = "month"
TARGET_AHEAD_COLUMN = "target"


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


def _add_lag_features(
    monthly: pd.DataFrame,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    monthly = monthly.copy()
    for lag in lags:
        monthly[f"lag_{lag}"] = monthly.groupby(SKU_COLUMN)[TARGET_COLUMN].shift(lag)
    return monthly


def build_dataset(
    monthly: pd.DataFrame,
    lags: tuple[int, ...],
    horizon: int = 1,
) -> tuple[pd.DataFrame, list[str]]:
    """Build tabular lag features and a one-step-ahead target.

    Returns the dataset (NaN rows dropped) and the lag column names.
    """
    lag_columns = _lag_column_names(lags)
    df = _add_lag_features(monthly, lags)
    df[TARGET_AHEAD_COLUMN] = df.groupby(SKU_COLUMN)[TARGET_COLUMN].shift(-horizon)
    df = df.dropna().reset_index(drop=True)
    return df, lag_columns
