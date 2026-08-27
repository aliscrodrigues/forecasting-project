import unittest

import pandas as pd

from preprocess.transform import _add_lag_features, preprocess_series


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


if __name__ == "__main__":
    unittest.main()
