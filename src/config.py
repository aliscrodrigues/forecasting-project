from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "fake_sales.csv"
M5_DATA_PATH = PROJECT_ROOT / "data" / "m5" / "processed" / "m5_daily_sales.parquet"

DEFAULT_LAGS = tuple(range(0, 13))
DEFAULT_ROLLING_WINDOWS = (3, 6, 12)
DEFAULT_FORECAST_HORIZON = 1
# M5 monthly range: 2011-01 → 2016-06 (top 50 SKUs, see scripts/extract_m5.py)
DEFAULT_REFERENCE_MONTH = "2015-06"
DEFAULT_VALIDATION_MONTHS = 6

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 500
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 0.00001
DEFAULT_PATIENCE = 50
DEFAULT_HIDDEN_DIMS = (256, 128, 64, 32)
DEFAULT_DROPOUT = 0.20
DEFAULT_LOSS = "mae"
DEFAULT_SEED = 42


@dataclass(frozen=True)
class ProjectConfig:
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    forecast_horizon: int
    reference_month: str
    validation_months: int
    target_column: str
    sku_column: str
    date_column: str
    data_path: Path
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    hidden_dims: tuple[int, ...]
    dropout: float
    loss: str
    seed: int


def _default_data_path() -> Path:
    """Prefer the M5 parquet when present; otherwise fall back to the sample CSV."""
    if M5_DATA_PATH.exists():
        return M5_DATA_PATH
    return SAMPLE_DATA_PATH


def default_config() -> ProjectConfig:
    """Return the default project configuration."""
    return ProjectConfig(
        lags=DEFAULT_LAGS,
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
        forecast_horizon=DEFAULT_FORECAST_HORIZON,
        reference_month=DEFAULT_REFERENCE_MONTH,
        validation_months=DEFAULT_VALIDATION_MONTHS,
        target_column="sales",
        sku_column="item_id",
        date_column="date",
        data_path=_default_data_path(),
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        patience=DEFAULT_PATIENCE,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        dropout=DEFAULT_DROPOUT,
        loss=DEFAULT_LOSS,
        seed=DEFAULT_SEED,
    )
