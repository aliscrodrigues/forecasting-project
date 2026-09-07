import unittest

import numpy as np
import pandas as pd

from evaluation.metrics import (
    evaluate,
    evaluate_predictions,
    format_metrics_table,
    pct_bias,
)


class TestMetrics(unittest.TestCase):
    def test_mae_perfect_prediction(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        self.assertEqual(evaluate(y_true, y_pred)["mae"], 0.0)

    def test_pct_bias_over_forecast(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        self.assertAlmostEqual(pct_bias(y_true, y_pred), 10.0)

    def test_pct_bias_under_forecast(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([90.0, 180.0])
        self.assertAlmostEqual(pct_bias(y_true, y_pred), -10.0)

    def test_evaluate_predictions_returns_all_models(self):
        y_true = np.array([30.0, 40.0])
        predictions = {
            "MLP": np.array([31.0, 39.0]),
            "Naive": np.array([32.0, 38.0]),
        }
        metrics_df = evaluate_predictions(y_true, predictions)
        self.assertEqual(len(metrics_df), 2)
        self.assertSetEqual(set(metrics_df["model"]), {"MLP", "Naive"})

    def test_format_metrics_table_orders_by_mae(self):
        metrics_df = pd.DataFrame(
            [
                {
                    "model": "MLP",
                    "mae": 5.0,
                    "rmse": 6.0,
                    "mae_over_mean": 0.1,
                    "pct_bias": 3.0,
                },
                {
                    "model": "Naive",
                    "mae": 2.0,
                    "rmse": 3.0,
                    "mae_over_mean": 0.05,
                    "pct_bias": -1.0,
                },
            ]
        )
        table = format_metrics_table(metrics_df)
        self.assertLess(table.index("Naive"), table.index("MLP"))


if __name__ == "__main__":
    unittest.main()
