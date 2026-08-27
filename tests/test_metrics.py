import unittest

import numpy as np

from evaluation.metrics import mae, rmse


class TestMetrics(unittest.TestCase):
    def test_mae_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        self.assertEqual(mae(y_true, y_pred), 0.0)

    def test_mae_known_value(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        self.assertAlmostEqual(mae(y_true, y_pred), 7 / 3)

    def test_rmse_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([3.0, 4.0, 0.0])
        expected = np.sqrt(25 / 3)
        self.assertAlmostEqual(rmse(y_true, y_pred), expected, places=5)

    def test_mae_multiple_cases(self):
        cases = [
            ([1, 2, 3], [1, 2, 3], 0.0),
            ([0, 0], [3, 4], 3.5),
        ]
        for y_true, y_pred, expected in cases:
            with self.subTest(y_true=y_true, y_pred=y_pred):
                result = mae(np.array(y_true, dtype=np.float64), np.array(y_pred))
                self.assertAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
