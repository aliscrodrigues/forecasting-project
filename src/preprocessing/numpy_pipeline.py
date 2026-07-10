"""NumPy-based preprocessing pipeline.

This module provides vectorized functions for:
- Normalization (Min-Max scaling to [0, 1])
- Standardization (Z-score: zero mean, unit variance)
- Statistics computation (mean, std, min, max per feature)
- Dimension analysis (shape reports for debugging/logging)
- Data preparation for PyTorch training
"""

import numpy as np


# ---------------------------------------------------------------------------
# Normalization (Min-Max)
# ---------------------------------------------------------------------------


def normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-Max normalization: scales each feature to the range [0, 1].

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    -------
    X_norm : scaled array
    x_min  : per-feature minimum (for inverse transform)
    x_max  : per-feature maximum (for inverse transform)
    """
    print("  [NumPy] Applying Min-Max normalization...")
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    # Avoid division by zero for constant features
    range_ = x_max - x_min
    range_[range_ == 0] = 1.0
    X_norm = (X - x_min) / range_
    print(f"    → Output range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
    return X_norm, x_min, x_max


def denormalize(
    X_norm: np.ndarray, x_min: np.ndarray, x_max: np.ndarray
) -> np.ndarray:
    """Inverse of Min-Max normalization."""
    range_ = x_max - x_min
    range_[range_ == 0] = 1.0
    return X_norm * range_ + x_min


# ---------------------------------------------------------------------------
# Standardization (Z-score)
# ---------------------------------------------------------------------------


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardization: transforms to zero mean and unit variance.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)

    Returns
    -------
    X_std : standardized array
    mean  : per-feature mean
    std   : per-feature standard deviation
    """
    print("  [NumPy] Applying Z-score standardization...")
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Avoid division by zero for constant features
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    X_std = (X - mean) / std_safe
    print(f"    → Mean after: {X_std.mean(axis=0).mean():.6f}, "
          f"Std after: {X_std.std(axis=0).mean():.6f}")
    return X_std, mean, std


def destandardize(
    X_std: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Inverse of Z-score standardization."""
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return X_std * std_safe + mean


# ---------------------------------------------------------------------------
# Statistics (vectorized)
# ---------------------------------------------------------------------------


def compute_feature_statistics(X: np.ndarray, feature_names: list[str] | None = None) -> dict:
    """Compute descriptive statistics for each feature using vectorized NumPy ops.

    Returns a dictionary with per-feature mean, std, min, max, and median.
    """
    print("  [NumPy] Computing feature statistics (vectorized)...")
    n_samples, n_features = X.shape
    stats = {
        "n_samples": n_samples,
        "n_features": n_features,
        "mean": X.mean(axis=0),
        "std": X.std(axis=0),
        "min": X.min(axis=0),
        "max": X.max(axis=0),
        "median": np.median(X, axis=0),
    }
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    print(f"    {'Feature':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"    {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
    for i, name in enumerate(feature_names):
        print(f"    {name:<12} {stats['mean'][i]:>10.2f} {stats['std'][i]:>10.2f} "
              f"{stats['min'][i]:>10.2f} {stats['max'][i]:>10.2f}")
    return stats


# ---------------------------------------------------------------------------
# Dimension analysis
# ---------------------------------------------------------------------------


def analyze_dimensions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Analyze and report array dimensions across train/val/test splits.

    This is essential to validate shapes before feeding data into PyTorch.
    """
    print("\n  [NumPy] Dimension analysis:")
    report = {
        "train": {"X": X_train.shape, "y": y_train.shape},
        "val": {"X": X_val.shape, "y": y_val.shape},
        "test": {"X": X_test.shape, "y": y_test.shape},
    }
    total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f"    {'Split':<12} {'X shape':<20} {'y shape':<15} {'% of total':>10}")
    print(f"    {'─' * 12} {'─' * 20} {'─' * 15} {'─' * 10}")
    for split_name, arrays in report.items():
        pct = arrays["X"][0] / total_samples * 100 if total_samples > 0 else 0
        print(f"    {split_name:<12} {str(arrays['X']):<20} "
              f"{str(arrays['y']):<15} {pct:>9.1f}%")
    print(f"    {'─' * 12} {'─' * 20} {'─' * 15} {'─' * 10}")
    print(f"    {'TOTAL':<12} {total_samples:<20} {'':<15} {'100.0%':>10}")
    print(f"\n    Input dimension (n_features): {X_train.shape[1]}")
    print(f"    Ready for PyTorch: input_size={X_train.shape[1]}, "
          f"output_size=1")
    report["total_samples"] = total_samples
    report["n_features"] = X_train.shape[1]
    return report


# ---------------------------------------------------------------------------
# Applying transform on validation/test using training statistics
# ---------------------------------------------------------------------------


def normalize_with_params(
    X: np.ndarray, x_min: np.ndarray, x_max: np.ndarray
) -> np.ndarray:
    """Normalize using pre-computed min/max (from training set)."""
    range_ = x_max - x_min
    range_[range_ == 0] = 1.0
    return (X - x_min) / range_


def standardize_with_params(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Standardize using pre-computed mean/std (from training set)."""
    std_safe = std.copy()
    std_safe[std_safe == 0] = 1.0
    return (X - mean) / std_safe


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str = "standardize",
    feature_names: list[str] | None = None,
) -> dict:
    """Execute the full NumPy preprocessing pipeline.

    Steps:
    1. Compute and display feature statistics (training set)
    2. Normalize or standardize features
    3. Apply same transform to val/test using training params
    4. Report dimensions for PyTorch readiness

    Parameters
    ----------
    method : 'normalize' (Min-Max) or 'standardize' (Z-score)

    Returns
    -------
    Dictionary with transformed arrays and metadata.
    """
    print("\n" + "=" * 60)
    print(" NumPy Preprocessing Pipeline")
    print("=" * 60)

    # Step 1 – Feature statistics on raw training data
    print("\n[Step 1] Raw feature statistics (training set):")
    stats = compute_feature_statistics(X_train, feature_names)

    # Step 2 – Transform features
    print(f"\n[Step 2] Scaling method: {method}")
    if method == "normalize":
        X_train_t, param1, param2 = normalize(X_train)
        X_val_t = normalize_with_params(X_val, param1, param2)
        X_test_t = normalize_with_params(X_test, param1, param2)
    elif method == "standardize":
        X_train_t, param1, param2 = standardize(X_train)
        X_val_t = standardize_with_params(X_val, param1, param2)
        X_test_t = standardize_with_params(X_test, param1, param2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'normalize' or 'standardize'.")

    # Step 3 – Verify transformed stats
    print("\n[Step 3] Transformed feature statistics (training set):")
    compute_feature_statistics(X_train_t, feature_names)

    # Step 4 – Dimension analysis
    print("\n[Step 4] Dimension analysis:")
    dim_report = analyze_dimensions(
        X_train_t, y_train, X_val_t, y_val, X_test_t, y_test
    )

    print("\n" + "=" * 60)
    print(" Pipeline complete – data ready for PyTorch")
    print("=" * 60 + "\n")

    return {
        "X_train": X_train_t,
        "y_train": y_train,
        "X_val": X_val_t,
        "y_val": y_val,
        "X_test": X_test_t,
        "y_test": y_test,
        "params": (param1, param2),
        "method": method,
        "stats": stats,
        "dimensions": dim_report,
    }

