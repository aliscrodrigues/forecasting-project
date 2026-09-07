import logging

import torch

from baselines import forecast_baselines
from config import PROJECT_ROOT, default_config
from data.reader import load_sales_data, validate_sales_schema
from evaluation.metrics import evaluate_predictions, format_metrics_table
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
from utils import get_device, set_seed

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
        "Config: reference=%s, validation=%s months, horizon=%s, loss=%s, "
        "hidden=%s, dropout=%s, lr=%s, wd=%s, seed=%s, device=%s",
        config.reference_month,
        config.validation_months,
        config.forecast_horizon,
        config.loss,
        config.hidden_dims,
        config.dropout,
        config.learning_rate,
        config.weight_decay,
        config.seed,
        device,
    )

    # Step 1: load and validate raw sales data
    logger.info("[1/7] Load and validate data")
    raw_data = load_sales_data(str(config.data_path))
    raw_data = validate_sales_schema(raw_data)
    logger.info("Loaded %s rows from %s", len(raw_data), config.data_path.name)

    # Step 2: aggregate monthly series and build lag features
    logger.info("[2/7] Build monthly features and supervised dataset")
    monthly = preprocess_series(raw_data)
    supervised, feature_columns = build_dataset(
        monthly,
        lags=config.lags,
        rolling_windows=config.rolling_windows,
        horizon=config.forecast_horizon,
    )
    logger.info(
        "Monthly records: %s | Supervised samples: %s | Features: %s",
        len(monthly),
        len(supervised),
        len(feature_columns),
    )

    # Step 3: split by reference forecast month
    logger.info("[3/7] Split data by reference month")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), test_target_dates = (
        time_based_split(
            supervised,
            date_column=MONTH_COLUMN,
            feature_columns=feature_columns,
            target_column=TARGET_AHEAD_COLUMN,
            reference_month=config.reference_month,
            validation_months=config.validation_months,
            forecast_horizon=config.forecast_horizon,
        )
    )
    logger.info(
        "Partitions: train=%s, val=%s, test=%s",
        len(y_train),
        len(y_val),
        len(y_test),
    )

    # Step 4: scale features and targets using training-set statistics
    logger.info("[4/7] Scale features and targets")
    X_train_pp, X_val_pp, X_test_pp = scale_features(
        X_train, X_val, X_test, method="standardize"
    )
    y_train_s, y_val_s, _, y_loc, y_scale = scale_target(y_train, y_val, y_test)

    # Step 5: train neural network and save model
    logger.info("[5/7] Train neural network")
    set_seed(config.seed)
    train_loader = make_loader(
        X_train_pp,
        y_train_s,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = make_loader(
        X_val_pp, y_val_s, batch_size=config.batch_size, shuffle=False
    )
    model = NeuralModel(
        input_dim=X_train_pp.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    model = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        patience=config.patience,
        loss=config.loss,
    )
    torch.save(model.state_dict(), model_path)
    logger.info("Model saved to %s", model_path)

    # Step 6: generate baseline forecasts for the test period
    logger.info("[6/7] Generate baseline forecasts")
    baseline_predictions = forecast_baselines(
        monthly,
        supervised,
        config.reference_month,
        config.forecast_horizon,
    )
    logger.info("Generated %s baseline models", len(baseline_predictions))

    # Step 7: evaluate neural network and baselines on the test period
    logger.info("[7/7] Evaluate models and compare test results")
    test_months = test_target_dates.dt.to_period("M")
    logger.info(
        "Test scope: %s rows across %s target months (%s to %s)",
        len(y_test),
        test_months.nunique(),
        test_months.min() if len(test_months) else None,
        test_months.max() if len(test_months) else None,
    )
    test_predictions = unscale_target(predict(model, X_test_pp, device), y_loc, y_scale)
    metrics_df = evaluate_predictions(
        y_test,
        {"MLP": test_predictions, **baseline_predictions},
    )
    logger.info("Test comparison:\n%s", format_metrics_table(metrics_df))
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
