import numpy as np
import pandas as pd

SKU_COLUMN = "item_id"
DATE_COLUMN = "date"
TARGET_COLUMN = "sales"
MONTH_COLUMN = "month"
TARGET_AHEAD_COLUMN = "target"


def aggregate_monthly_by_sku(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sales into monthly totals per SKU across all stores."""
    print("Aggregating daily sales into monthly totals per SKU...")
    data = data.copy()
    data[MONTH_COLUMN] = data[DATE_COLUMN].dt.to_period("M").dt.to_timestamp()
    monthly = (
        data.groupby([SKU_COLUMN, MONTH_COLUMN])[TARGET_COLUMN]
        .sum()
        .reset_index()
    )
    monthly = monthly.sort_values([SKU_COLUMN, MONTH_COLUMN]).reset_index(drop=True)
    print(f"  → Aggregated to {len(monthly)} monthly records "
          f"({monthly[SKU_COLUMN].nunique()} SKUs)")
    return monthly


def lag_column_names(lags: tuple[int, ...]) -> list[str]:
    """Return lag feature column names for the given lag values."""
    print(f"Defining lag columns: {lags}")
    return [f"lag_{lag}" for lag in lags]


def add_lag_features(
    monthly: pd.DataFrame,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    """Add lag columns for each SKU time series."""
    print(f"Creating lag features: {lags}")
    monthly = monthly.copy()
    for lag in lags:
        monthly[f"lag_{lag}"] = monthly.groupby(SKU_COLUMN)[TARGET_COLUMN].shift(lag)
    return monthly


def build_supervised_dataset(
    monthly: pd.DataFrame,
    lags: tuple[int, ...],
    horizon: int = 1,
) -> pd.DataFrame:
    """Build tabular features (X) and one-step-ahead target (y).

    Returns the full DataFrame with lag features and the target column,
    with rows containing NaN dropped.
    """
    print(f"Building supervised dataset (X, y) with horizon={horizon}...")
    df = add_lag_features(monthly, lags)
    # Target is the sales value `horizon` steps ahead (shifted backwards)
    df[TARGET_AHEAD_COLUMN] = df.groupby(SKU_COLUMN)[TARGET_COLUMN].shift(-horizon)
    # Drop rows where lags or target are NaN
    df = df.dropna().reset_index(drop=True)
    print(f"  → Supervised dataset: {len(df)} samples")
    return df


def to_numpy(
    features: pd.DataFrame,
    target: pd.Series,
    lags: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert feature matrix and target vector to NumPy arrays."""
    print("Converting features and target to NumPy arrays...")
    X = np.asarray(features.values, dtype=np.float64)
    y = np.asarray(target.values, dtype=np.float64)
    print(f"  → X shape: {X.shape}, y shape: {y.shape}")
    return X, y
