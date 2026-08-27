import unittest

import numpy as np
import pandas as pd

from preprocess.split import time_based_split


class TestTimeBasedSplit(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2015-01-01",
                        "2015-06-01",
                        "2015-07-01",
                        "2015-12-01",
                        "2016-01-01",
                    ]
                ),
                "lag_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "target": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        self.feature_columns = ["lag_1"]

    def test_split_sizes(self):
        train, val, test = time_based_split(
            self.df,
            date_column="date",
            feature_columns=self.feature_columns,
            target_column="target",
            train_end="2015-06-01",
            val_end="2015-12-01",
        )

        _, y_train = train
        _, y_val = val
        _, y_test = test

        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(y_val), 2)
        self.assertEqual(len(y_test), 1)

    def test_split_values(self):
        train, val, test = time_based_split(
            self.df,
            date_column="date",
            feature_columns=self.feature_columns,
            target_column="target",
            train_end="2015-06-01",
            val_end="2015-12-01",
        )

        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test

        np.testing.assert_array_equal(y_train, np.array([10.0, 20.0]))
        np.testing.assert_array_equal(y_val, np.array([30.0, 40.0]))
        np.testing.assert_array_equal(y_test, np.array([50.0]))
        np.testing.assert_array_equal(X_train[:, 0], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(X_val[:, 0], np.array([3.0, 4.0]))
        np.testing.assert_array_equal(X_test[:, 0], np.array([5.0]))


if __name__ == "__main__":
    unittest.main()
