import torch

from data.loader import load_sales_data, validate_sales_schema
from evaluation.metrics import evaluate
from models.neural import NeuralModel
from preprocess.scaler import scale_features, scale_target, unscale_target
from preprocess.split import time_based_split
from preprocess.transform import (
    MONTH_COLUMN,
    TARGET_AHEAD_COLUMN,
    build_dataset,
    preprocess_series,
)
from train.loop import train_model
from utils.config import PROJECT_ROOT, default_config
from utils.pytorch import get_device, make_loader, predict


def main() -> None:
    """Run the monthly SKU demand forecasting pipeline."""
    print("=" * 60)
    print(" M5 Monthly SKU Demand Forecasting Pipeline")
    print("=" * 60 + "\n")

    config = default_config()

    raw_data = load_sales_data(str(config.data_path))
    data = validate_sales_schema(raw_data)

    monthly = preprocess_series(data)
    supervised, lag_columns = build_dataset(
        monthly,
        lags=config.lags,
        horizon=config.forecast_horizon,
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_based_split(
        supervised,
        date_column=MONTH_COLUMN,
        feature_columns=lag_columns,
        target_column=TARGET_AHEAD_COLUMN,
        train_end=config.train_end,
        val_end=config.val_end,
    )

    X_train_pp, X_val_pp, X_test_pp = scale_features(
        X_train, X_val, X_test, method="standardize"
    )
    y_train_s, y_val_s, y_test_s, y_loc, y_scale = scale_target(
        y_train, y_val, y_test
    )

    device = get_device()
    train_loader = make_loader(
        X_train_pp, y_train_s, batch_size=config.batch_size, shuffle=True
    )
    val_loader = make_loader(
        X_val_pp, y_val_s, batch_size=config.batch_size, shuffle=False
    )
    print(f"  Device: {device}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = NeuralModel(input_dim=X_train_pp.shape[1], hidden_dim=64, output_dim=1)

    print("\n─── Starting Neural Network Training ───")
    model = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
    )

    model_path = PROJECT_ROOT / "demand_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    print("\n─── Final Evaluation (original demand units) ───")
    val_predictions = unscale_target(predict(model, X_val_pp, device), y_loc, y_scale)
    test_predictions = unscale_target(predict(model, X_test_pp, device), y_loc, y_scale)
    val_metrics = evaluate(y_val, val_predictions, split="validation")
    test_metrics = evaluate(y_test, test_predictions, split="test")

    print("\n" + "=" * 60)
    print(" Final Summary")
    print("=" * 60)
    print(f"  Training samples:    {X_train_pp.shape[0]}")
    print(f"  Validation samples:  {X_val_pp.shape[0]}")
    print(f"  Test samples:        {X_test_pp.shape[0]}")
    print(f"  Features (lags):     {X_train_pp.shape[1]}")
    print(f"  Validation MAE:      {val_metrics['mae']:.4f}  (MAE/mean={val_metrics['mae_over_mean']:.4f})")
    print(f"  Validation RMSE:     {val_metrics['rmse']:.4f}")
    print(f"  Test MAE:            {test_metrics['mae']:.4f}  (MAE/mean={test_metrics['mae_over_mean']:.4f})")
    print(f"  Test RMSE:           {test_metrics['rmse']:.4f}")
    print("=" * 60)
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
