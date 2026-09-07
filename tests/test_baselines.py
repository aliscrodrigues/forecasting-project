import unittest

import numpy as np
import pandas as pd

from baselines import BASELINE_MODEL_NAMES, forecast_baselines
from preprocess.split import _normalize_month
from preprocess.transform import MONTH_COLUMN, SKU_COLUMN, TARGET_COLUMN


def _monthly_series(
    sku: str,
    start: str,
    months: int,
    values: list[float] | None = None,
) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=months, freq="MS")
    if values is None:
        values = [float(i + 1) for i in range(months)]
    return pd.DataFrame(
        {
            SKU_COLUMN: [sku] * months,
            MONTH_COLUMN: dates,
            TARGET_COLUMN: values,
        }
    )


def _supervised_from_monthly(monthly: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    supervised = monthly.copy()
    supervised["target"] = supervised.groupby(SKU_COLUMN)[TARGET_COLUMN].shift(-horizon)
    return supervised.dropna().reset_index(drop=True)


class TestBaselines(unittest.TestCase):
    def setUp(self):
        monthly_a = _monthly_series("A", "2020-01-01", 24)
        monthly_b = _monthly_series(
            "B",
            "2020-01-01",
            24,
            values=[float(i * 2) for i in range(1, 25)],
        )
        self.monthly = pd.concat([monthly_a, monthly_b], ignore_index=True)
        self.supervised = _supervised_from_monthly(self.monthly, horizon=1)
        self.reference_month = "2021-06"

    def _test_row_count(self) -> int:
        reference_dt = _normalize_month(self.reference_month)
        target_dates = pd.to_datetime(self.supervised[MONTH_COLUMN]) + pd.DateOffset(
            months=1
        )
        return int((target_dates >= reference_dt).sum())

    def test_baseline_model_names(self):
        self.assertEqual(
            BASELINE_MODEL_NAMES,
            (
                "Naive",
                "WindowAverage_3",
                "WindowAverage_6",
                "WindowAverage_12",
                "SeasonalNaive_12",
            ),
        )

    def test_forecast_baselines_shapes_match_test_split(self):
        predictions = forecast_baselines(
            self.monthly,
            self.supervised,
            self.reference_month,
        )
        expected_rows = self._test_row_count()

        for model_name in BASELINE_MODEL_NAMES:
            self.assertIn(model_name, predictions)
            self.assertEqual(len(predictions[model_name]), expected_rows)

    def test_naive_matches_last_observed_value(self):
        predictions = forecast_baselines(
            self.monthly,
            self.supervised,
            self.reference_month,
        )
        reference_dt = _normalize_month(self.reference_month)
        target_dates = pd.to_datetime(self.supervised[MONTH_COLUMN]) + pd.DateOffset(
            months=1
        )
        test_rows = self.supervised.loc[target_dates >= reference_dt]
        expected = test_rows[TARGET_COLUMN].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(predictions["Naive"], expected)

    def test_predictions_have_no_missing_values(self):
        predictions = forecast_baselines(
            self.monthly,
            self.supervised,
            self.reference_month,
        )
        for model_name in BASELINE_MODEL_NAMES:
            self.assertFalse(np.isnan(predictions[model_name]).any())


if __name__ == "__main__":
    unittest.main()
