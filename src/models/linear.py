import numpy as np


class LinearRegressionModel:
    """Linear regression model implemented with NumPy (Normal Equation).

    Uses the closed-form solution: w = (X^T X)^{-1} X^T y
    with a bias term appended to the feature matrix.
    """

    def __init__(self) -> None:
        self.weights: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model weights using the Normal Equation."""
        print("Training linear regression model...")
        if X.size == 0 or y.size == 0:
            print("  → Skipping fit (empty arrays)")
            return
        # Add bias column (column of ones)
        X_b = np.column_stack([np.ones(X.shape[0]), X])
        # Normal equation: w = (X^T X)^{-1} X^T y
        # Using pseudo-inverse for numerical stability
        self.weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        print(f"  → Weights shape: {self.weights.shape} "
              f"(bias + {X.shape[1]} features)")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the given feature matrix."""
        print("Generating predictions with linear regression...")
        if self.weights is None or X.size == 0:
            print("  → Returning empty predictions (model not fitted or empty input)")
            return np.array([])
        X_b = np.column_stack([np.ones(X.shape[0]), X])
        predictions = X_b @ self.weights
        print(f"  → Generated {len(predictions)} predictions")
        return predictions
