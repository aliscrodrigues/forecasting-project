import logging

import torch

from config import PROJECT_ROOT, default_config
from data.reader import load_sales_data, validate_sales_schema
from evaluation.metrics import evaluate
from inference.predict import predict
from models.neural import NeuralModel
from preprocess.scaler import scale_features, scale_target, unscale_target
from preprocess.split import time_based_split
from preprocess.transform import (
    MONTH_COLUMN,
    TARGET_AHEAD_COLUMN,
    build_dataset,
    preprocess_series,
)
from train.dataloader import make_loader
from train.trainer import train_model
from utils import get_device

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure standard logging for pipeline execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def main() -> None:
    """Run the monthly SKU demand forecasting pipeline."""
    configure_logging()
    config = default_config()
    model_path = PROJECT_ROOT / "demand_model.pth"
    device = get_device()

    logger.info("M5 Monthly SKU Demand Forecasting Pipeline")
    logger.info(
        "Config: reference=%s, validation=%s months, horizon=%s, device=%s",
        config.reference_month,
        config.validation_months,
        config.forecast_horizon,
        device,
    )

    # Step 1: load and validate raw sales data
    logger.info("[1/6] Load and validate data")
    raw_data = load_sales_data(str(config.data_path))
    raw_data = validate_sales_schema(raw_data)
    logger.info("Loaded %s rows from %s", len(raw_data), config.data_path.name)

    # Step 2: aggregate monthly series and build lag features
    logger.info("[2/6] Build monthly features and supervised dataset")
    monthly = preprocess_series(raw_data)
    supervised, lag_columns = build_dataset(
        monthly,
        lags=config.lags,
        horizon=config.forecast_horizon,
    )
    logger.info(
        "Monthly records: %s | Supervised samples: %s | Features: %s",
        len(monthly),
        len(supervised),
        len(lag_columns),
    )

    # Step 3: split by reference forecast month
    logger.info("[3/6] Split data by reference month")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = time_based_split(
        supervised,
        date_column=MONTH_COLUMN,
        feature_columns=lag_columns,
        target_column=TARGET_AHEAD_COLUMN,
        reference_month=config.reference_month,
        validation_months=config.validation_months,
        forecast_horizon=config.forecast_horizon,
    )
    logger.info(
        "Partitions: train=%s, val=%s, test=%s",
        len(y_train),
        len(y_val),
        len(y_test),
    )

    # Step 4: scale features and targets using training-set statistics
    logger.info("[4/6] Scale features and targets")
    X_train_pp, X_val_pp, X_test_pp = scale_features(
        X_train, X_val, X_test, method="standardize"
    )
    y_train_s, y_val_s, y_test_s, y_loc, y_scale = scale_target(y_train, y_val, y_test)

    # Step 5: train neural network and save model
    logger.info("[5/6] Train neural network")
    train_loader = make_loader(
        X_train_pp, y_train_s, batch_size=config.batch_size, shuffle=True
    )
    val_loader = make_loader(
        X_val_pp, y_val_s, batch_size=config.batch_size, shuffle=False
    )
    model = NeuralModel(
        input_dim=X_train_pp.shape[1],
        hidden_dim=64,
        output_dim=1,
    )
    model = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
    )
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)

    # Step 6: evaluate validation and test partitions
    logger.info("[6/6] Evaluate validation and test partitions")
    val_predictions = unscale_target(predict(model, X_val_pp, device), y_loc, y_scale)
    test_predictions = unscale_target(predict(model, X_test_pp, device), y_loc, y_scale)
    val_metrics = evaluate(y_val, val_predictions, split="validation")
    test_metrics = evaluate(y_test, test_predictions, split="test")

    logger.info(
        "Validation: MAE=%.2f, RMSE=%.2f, MAE/mean=%.4f",
        val_metrics["mae"],
        val_metrics["rmse"],
        val_metrics["mae_over_mean"],
    )
    logger.info(
        "Test: MAE=%.2f, RMSE=%.2f, MAE/mean=%.4f",
        test_metrics["mae"],
        test_metrics["rmse"],
        test_metrics["mae_over_mean"],
    )
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
