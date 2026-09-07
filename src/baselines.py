"""StatsForecast baseline models for the test period."""

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, WindowAverage

from preprocess.split import _normalize_month
from preprocess.transform import MONTH_COLUMN, SKU_COLUMN, TARGET_COLUMN

BASELINE_MODEL_NAMES: tuple[str, ...] = (
    "Naive",
    "WindowAverage_3",
    "WindowAverage_6",
    "WindowAverage_12",
    "SeasonalNaive_12",
)


def forecast_baselines(
    monthly: pd.DataFrame,
    supervised: pd.DataFrame,
    reference_month: str,
    forecast_horizon: int = 1,
) -> dict[str, np.ndarray]:
    """Forecast test-period targets with StatsForecast baselines.

    For each origin month in the test window, history is truncated to that month
    and ``forecast(h=forecast_horizon)`` is called — equivalent to one-step
    ahead forecasts at every test origin without cross-validation.
    """
    reference_dt = _normalize_month(reference_month)
    target_dates = pd.to_datetime(supervised[MONTH_COLUMN]) + pd.DateOffset(
        months=forecast_horizon
    )
    test_rows = supervised.loc[target_dates >= reference_dt].copy()
    test_rows["target_date"] = target_dates[target_dates >= reference_dt]

    sf_df = monthly.rename(
        columns={SKU_COLUMN: "unique_id", MONTH_COLUMN: "ds", TARGET_COLUMN: "y"}
    )[["unique_id", "ds", "y"]]

    models = [
        Naive(alias="Naive"),
        WindowAverage(window_size=3, alias="WindowAverage_3"),
        WindowAverage(window_size=6, alias="WindowAverage_6"),
        WindowAverage(window_size=12, alias="WindowAverage_12"),
        SeasonalNaive(season_length=12, alias="SeasonalNaive_12"),
    ]
    sf = StatsForecast(models=models, freq="MS", n_jobs=1)

    forecast_parts: list[pd.DataFrame] = []
    for origin in sorted(test_rows[MONTH_COLUMN].unique()):
        cutoff = pd.Timestamp(origin)
        history = sf_df[sf_df["ds"] <= cutoff]
        forecast_parts.append(sf.forecast(df=history, h=forecast_horizon))

    all_forecasts = pd.concat(forecast_parts, ignore_index=True)

    predictions: dict[str, np.ndarray] = {}
    for name in BASELINE_MODEL_NAMES:
        merged = test_rows.merge(
            all_forecasts[["unique_id", "ds", name]],
            left_on=[SKU_COLUMN, "target_date"],
            right_on=["unique_id", "ds"],
            how="left",
        )
        if merged[name].isna().any():
            raise ValueError(f"Missing baseline predictions for {name}")
        predictions[name] = merged[name].to_numpy(dtype=np.float64)
    return predictions
