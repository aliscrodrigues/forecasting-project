"""Fit feature scaling on the training set and apply it to val/test."""

import numpy as np

_EPS = 1e-12


def _fit_standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < _EPS, 1.0, std)
    return mean, std


def _fit_normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    scale = x_max - x_min
    scale = np.where(scale < _EPS, 1.0, scale)
    return x_min, scale


def _apply(X: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - loc) / scale


def scale_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    method: str = "standardize",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale features using training-set statistics only.

    ``standardize`` uses z-score (mean/std). ``normalize`` uses min-max to [0, 1].
    Validation and test arrays are transformed with the same parameters.
    """
    if method == "standardize":
        loc, scale = _fit_standardize(X_train)
    elif method == "normalize":
        loc, scale = _fit_normalize(X_train)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'normalize' or 'standardize'.")

    print(f"Scaling features with {method} (fit on train, apply to val/test)...")
    X_train_s = _apply(X_train, loc, scale)
    X_val_s = _apply(X_val, loc, scale)
    X_test_s = _apply(X_test, loc, scale)
    print(
        f"  → Train X={X_train_s.shape} mean={X_train_s.mean():.4f} std={X_train_s.std():.4f}"
    )
    return X_train_s, X_val_s, X_test_s


def scale_target(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Standardize targets using training-set mean and std."""
    y_train_col = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    loc, scale = _fit_standardize(y_train_col)
    loc_f, scale_f = float(loc.item()), float(scale.item())

    print("Scaling target with standardize (fit on train, apply to val/test)...")
    y_train_s = _apply(np.asarray(y_train, dtype=np.float64).reshape(-1, 1), loc, scale).ravel()
    y_val_s = _apply(np.asarray(y_val, dtype=np.float64).reshape(-1, 1), loc, scale).ravel()
    y_test_s = _apply(np.asarray(y_test, dtype=np.float64).reshape(-1, 1), loc, scale).ravel()
    print(f"  → Train y mean={y_train_s.mean():.4f} std={y_train_s.std():.4f}")
    return y_train_s, y_val_s, y_test_s, loc_f, scale_f


def unscale_target(y_scaled: np.ndarray, loc: float, scale: float) -> np.ndarray:
    """Invert target standardization back to original demand units."""
    return np.asarray(y_scaled, dtype=np.float64) * scale + loc
