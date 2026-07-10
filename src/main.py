import numpy as np

from src.data.loader import load_sales_data, validate_sales_schema
from src.evaluation.metrics import evaluate
from src.models.linear import LinearRegressionModel
from src.preprocessing.numpy_pipeline import preprocess_pipeline
from src.preprocessing.transform import (
    MONTH_COLUMN,
    TARGET_AHEAD_COLUMN,
    aggregate_monthly_by_sku,
    build_supervised_dataset,
    lag_column_names,
)
from src.training.split import split_features_target, temporal_split
from src.utils.config import default_config


def main() -> None:
    """Run the monthly SKU demand forecasting pipeline.

    Full implementation using NumPy for:
    - Data loading and validation
    - Monthly aggregation and lag features
    - Temporal train/val/test split
    - NumPy preprocessing (normalization, standardization, statistics)
    - Linear regression training and prediction
    - Evaluation with MAE and RMSE
    """
    print("=" * 60)
    print(" M5 Monthly SKU Demand Forecasting Pipeline")
    print("=" * 60 + "\n")

    # ─── Step 1: Configuration ────────────────────────────────────────────
    config = default_config()

    # ─── Step 2: Load and validate data ───────────────────────────────────
    raw_data = load_sales_data(str(config.data_path))
    data = validate_sales_schema(raw_data)

    # ─── Step 3: Aggregate monthly by SKU ─────────────────────────────────
    monthly = aggregate_monthly_by_sku(data)

    # ─── Step 4: Build supervised dataset with lag features ───────────────
    supervised = build_supervised_dataset(
        monthly,
        lags=config.lags,
        horizon=config.forecast_horizon,
    )
    lag_columns = lag_column_names(config.lags)

    # ─── Step 5: Temporal split ───────────────────────────────────────────
    train_df, val_df, test_df = temporal_split(
        supervised,
        date_column=MONTH_COLUMN,
        train_end=config.train_end,
        val_end=config.val_end,
    )

    # ─── Step 6: Extract features and target arrays (NumPy) ──────────────
    X_train, y_train = split_features_target(train_df, lag_columns, TARGET_AHEAD_COLUMN)
    X_val, y_val = split_features_target(val_df, lag_columns, TARGET_AHEAD_COLUMN)
    X_test, y_test = split_features_target(test_df, lag_columns, TARGET_AHEAD_COLUMN)

    # ─── Step 7: NumPy Preprocessing Pipeline ─────────────────────────────
    # (normalization/standardization, statistics, dimension analysis)
    pipeline_result = preprocess_pipeline(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        method="standardize",
        feature_names=lag_columns,
    )

    # Extract preprocessed arrays
    X_train_pp = pipeline_result["X_train"]
    X_val_pp = pipeline_result["X_val"]
    X_test_pp = pipeline_result["X_test"]

    # ─── Step 8: Train linear regression model ────────────────────────────
    print("\n─── Model Training ───")
    model = LinearRegressionModel()
    model.fit(X_train_pp, y_train)

    # ─── Step 9: Predict and evaluate on validation ───────────────────────
    print("\n─── Validation Results ───")
    val_predictions = model.predict(X_val_pp)
    val_metrics = evaluate(y_val, val_predictions)

    # ─── Step 10: Predict and evaluate on test ────────────────────────────
    print("\n─── Test Results ───")
    test_predictions = model.predict(X_test_pp)
    test_metrics = evaluate(y_test, test_predictions)

    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Final Summary")
    print("=" * 60)
    print(f"  Training samples:    {X_train_pp.shape[0]}")
    print(f"  Validation samples:  {X_val_pp.shape[0]}")
    print(f"  Test samples:        {X_test_pp.shape[0]}")
    print(f"  Features (lags):     {X_train_pp.shape[1]}")
    print(f"  Validation MAE:      {val_metrics['mae']:.4f}")
    print(f"  Validation RMSE:     {val_metrics['rmse']:.4f}")
    print(f"  Test MAE:            {test_metrics['mae']:.4f}")
    print(f"  Test RMSE:           {test_metrics['rmse']:.4f}")
    print("=" * 60)
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
