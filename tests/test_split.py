import unittest

import numpy as np
import pandas as pd

from preprocess.split import time_based_split


def _monthly_frame(
    start: str,
    months: int,
    values: list[float] | None = None,
) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=months, freq="MS")
    if values is None:
        values = [float(i + 1) for i in range(months)]
    return pd.DataFrame(
        {
            "date": dates,
            "lag_1": values,
            "target": [value * 10 for value in values],
        }
    )


class TestTimeBasedSplit(unittest.TestCase):
    def setUp(self):
        self.feature_columns = ["lag_1"]

    def test_split_sizes_with_reference_target_month(self):
        df = _monthly_frame("2014-06-01", 19)

        train, val, test = time_based_split(
            df,
            date_column="date",
            feature_columns=self.feature_columns,
            target_column="target",
            reference_month="2015-06",
            validation_months=6,
            forecast_horizon=1,
        )

        _, y_train = train
        _, y_val = val
        _, y_test = test

        self.assertEqual(len(y_train), 5)
        self.assertEqual(len(y_val), 6)
        self.assertEqual(len(y_test), 8)

    def test_split_values_with_reference_target_month(self):
        df = _monthly_frame("2014-06-01", 19)

        train, val, test = time_based_split(
            df,
            date_column="date",
            feature_columns=self.feature_columns,
            target_column="target",
            reference_month="2015-06",
            validation_months=6,
            forecast_horizon=1,
        )

        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test

        np.testing.assert_array_equal(y_train, np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        np.testing.assert_array_equal(
            y_val, np.array([60.0, 70.0, 80.0, 90.0, 100.0, 110.0])
        )
        np.testing.assert_array_equal(
            y_test,
            np.array([120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]),
        )
        np.testing.assert_array_equal(
            X_train[:, 0], np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        np.testing.assert_array_equal(
            X_val[:, 0], np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        )
        np.testing.assert_array_equal(
            X_test[:, 0], np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
        )

    def test_year_rollover_validation_window(self):
        df = _monthly_frame("2014-08-01", 8)

        train, val, test = time_based_split(
            df,
            date_column="date",
            feature_columns=self.feature_columns,
            target_column="target",
            reference_month="2015-02",
            validation_months=3,
            forecast_horizon=1,
        )

        _, y_train = train
        _, y_val = val
        _, y_test = test

        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_val), 3)
        self.assertEqual(len(y_test), 3)

    def test_normalizes_mid_month_reference(self):
        df = _monthly_frame("2014-06-01", 19)

        train, val, test = time_based_split(
            df,
            date_column="date",
            feature_columns=self.feature_columns,
            target_column="target",
            reference_month="2015-06-15",
            validation_months=6,
            forecast_horizon=1,
        )

        _, y_train = train
        _, y_val = val
        _, y_test = test

        self.assertEqual(len(y_train), 5)
        self.assertEqual(len(y_val), 6)
        self.assertEqual(len(y_test), 8)

    def test_forecast_horizon_shifts_target_dates(self):
        df = _monthly_frame("2014-06-01", 19)

        train, val, test = time_based_split(
            df,
            date_column="date",
            feature_columns=self.feature_columns,
            target_column="target",
            reference_month="2015-06",
            validation_months=6,
            forecast_horizon=2,
        )

        _, y_train = train
        _, y_val = val
        _, y_test = test

        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_val), 6)
        self.assertEqual(len(y_test), 9)

    def test_rejects_invalid_horizon(self):
        df = _monthly_frame("2014-06-01", 6)
        with self.assertRaises(ValueError):
            time_based_split(
                df,
                date_column="date",
                feature_columns=self.feature_columns,
                target_column="target",
                reference_month="2015-06",
                validation_months=2,
                forecast_horizon=0,
            )

    def test_rejects_invalid_validation_months(self):
        df = _monthly_frame("2014-06-01", 6)
        with self.assertRaises(ValueError):
            time_based_split(
                df,
                date_column="date",
                feature_columns=self.feature_columns,
                target_column="target",
                reference_month="2015-06",
                validation_months=0,
                forecast_horizon=1,
            )


if __name__ == "__main__":
    unittest.main()
