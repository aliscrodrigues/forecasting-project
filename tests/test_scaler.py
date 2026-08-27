import unittest

import numpy as np

from preprocess.scaler import scale_features, scale_target, unscale_target


class TestScaler(unittest.TestCase):
    def test_scale_features_standardize_train_stats(self):
        X_train = np.array([[1.0, 5.0], [2.0, 10.0], [3.0, 15.0]])
        X_val = np.array([[2.0, 10.0]])
        X_test = np.array([[4.0, 20.0]])

        X_train_s, X_val_s, X_test_s = scale_features(
            X_train, X_val, X_test, method="standardize"
        )

        self.assertAlmostEqual(X_train_s[:, 0].mean(), 0.0, places=5)
        self.assertAlmostEqual(X_train_s[:, 0].std(), 1.0, places=5)
        self.assertAlmostEqual(X_train_s[:, 1].mean(), 0.0, places=5)
        self.assertAlmostEqual(X_train_s[:, 1].std(), 1.0, places=5)
        self.assertEqual(X_val_s.shape, (1, 2))
        self.assertEqual(X_test_s.shape, (1, 2))

    def test_scale_features_normalize_maps_train_to_unit_interval(self):
        X_train = np.array([[0.0, 10.0], [5.0, 20.0], [10.0, 30.0]])
        X_val = np.array([[5.0, 20.0]])
        X_test = np.array([[10.0, 30.0]])

        X_train_s, X_val_s, X_test_s = scale_features(
            X_train, X_val, X_test, method="normalize"
        )

        np.testing.assert_allclose(X_train_s.min(axis=0), np.zeros(2))
        np.testing.assert_allclose(X_train_s.max(axis=0), np.ones(2))
        np.testing.assert_allclose(X_val_s, np.array([[0.5, 0.5]]))
        np.testing.assert_allclose(X_test_s, np.array([[1.0, 1.0]]))

    def test_scale_target_roundtrip(self):
        y_train = np.array([10.0, 20.0, 30.0])
        y_val = np.array([15.0])
        y_test = np.array([25.0])

        y_train_s, y_val_s, y_test_s, loc, scale = scale_target(y_train, y_val, y_test)

        np.testing.assert_allclose(unscale_target(y_train_s, loc, scale), y_train)
        np.testing.assert_allclose(unscale_target(y_val_s, loc, scale), y_val)
        np.testing.assert_allclose(unscale_target(y_test_s, loc, scale), y_test)

    def test_unknown_method_raises(self):
        X = np.array([[1.0]])
        with self.assertRaises(ValueError):
            scale_features(X, X, X, method="invalid")


if __name__ == "__main__":
    unittest.main()
