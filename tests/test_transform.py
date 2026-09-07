import unittest

import pandas as pd

from preprocess.transform import (
    _add_derived_features,
    _add_lag_features,
    build_dataset,
    preprocess_series,
)


class TestTransform(unittest.TestCase):
    def test_preprocess_series_aggregates_to_monthly(self):
        daily = pd.DataFrame(
            {
                "item_id": ["A", "A", "A", "B"],
                "date": pd.to_datetime(
                    [
                        "2015-01-05",
                        "2015-01-10",
                        "2015-02-01",
                        "2015-01-15",
                    ]
                ),
                "sales": [10, 20, 30, 5],
            }
        )

        monthly = preprocess_series(daily)

        self.assertEqual(len(monthly), 3)
        self.assertIn("month", monthly.columns)

        sku_a_jan = monthly[
            (monthly["item_id"] == "A") & (monthly["month"] == "2015-01-01")
        ]["sales"].iloc[0]
        self.assertEqual(sku_a_jan, 30)

        sku_b_jan = monthly[
            (monthly["item_id"] == "B") & (monthly["month"] == "2015-01-01")
        ]["sales"].iloc[0]
        self.assertEqual(sku_b_jan, 5)

    def test_add_lag_features_per_sku(self):
        monthly = pd.DataFrame(
            {
                "item_id": ["A", "A", "A", "B", "B"],
                "month": pd.to_datetime(
                    [
                        "2015-01-01",
                        "2015-02-01",
                        "2015-03-01",
                        "2015-01-01",
                        "2015-02-01",
                    ]
                ),
                "sales": [10, 20, 30, 100, 200],
            }
        )

        result = _add_lag_features(monthly, lags=(1, 2))

        self.assertTrue(pd.isna(result.loc[0, "lag_1"]))
        self.assertTrue(pd.isna(result.loc[0, "lag_2"]))
        self.assertTrue(pd.isna(result.loc[3, "lag_1"]))
        self.assertEqual(result.loc[1, "lag_1"], 10)
        self.assertEqual(result.loc[2, "lag_1"], 20)
        self.assertEqual(result.loc[2, "lag_2"], 10)
        self.assertEqual(result.loc[4, "lag_1"], 100)

    def test_add_derived_features(self):
        monthly = pd.DataFrame(
            {
                "item_id": ["A"] * 4,
                "month": pd.date_range("2015-01-01", periods=4, freq="MS"),
                "sales": [10.0, 20.0, 30.0, 40.0],
            }
        )
        with_lags = _add_lag_features(monthly, lags=(1, 2, 3, 6, 12))
        result = _add_derived_features(with_lags, rolling_windows=(3, 6, 12))

        self.assertAlmostEqual(result.loc[3, "rolling_mean_3"], 30.0)
        self.assertAlmostEqual(result.loc[3, "momentum_1m"], 10.0)
        self.assertIn("month_sin", result.columns)
        self.assertIn("yoy_change", result.columns)

    def test_lag_0_is_origin_sales_and_differs_from_target(self):
        monthly = pd.DataFrame(
            {
                "item_id": ["A"] * 15,
                "month": pd.date_range("2015-01-01", periods=15, freq="MS"),
                "sales": [float(i + 1) for i in range(15)],
            }
        )
        dataset, feature_columns = build_dataset(
            monthly,
            lags=(0, 1, 2, 3, 6, 12),
            rolling_windows=(3, 6, 12),
        )

        self.assertIn("lag_0", feature_columns)
        self.assertGreater(len(dataset), 0)
        pd.testing.assert_series_equal(
            dataset["lag_0"],
            dataset["sales"],
            check_names=False,
        )
        self.assertTrue((dataset["lag_0"] != dataset["target"]).all())

    def test_build_dataset_includes_rolling_and_calendar_features(self):
        monthly = pd.DataFrame(
            {
                "item_id": ["A"] * 15,
                "month": pd.date_range("2015-01-01", periods=15, freq="MS"),
                "sales": [float(i + 1) for i in range(15)],
            }
        )
        dataset, feature_columns = build_dataset(
            monthly,
            lags=(1, 2, 3, 6, 12),
            rolling_windows=(3, 6, 12),
        )

        self.assertIn("rolling_mean_3", feature_columns)
        self.assertIn("month_cos", feature_columns)
        self.assertIn("target", dataset.columns)
        self.assertGreater(len(dataset), 0)


if __name__ == "__main__":
    unittest.main()
