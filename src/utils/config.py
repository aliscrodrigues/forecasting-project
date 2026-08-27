from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "fake_sales.csv"
M5_DATA_PATH = PROJECT_ROOT / "data" / "m5" / "processed" / "m5_daily_sales.parquet"

DEFAULT_LAGS = (1, 2, 3, 6, 12)
DEFAULT_FORECAST_HORIZON = 1
# M5 monthly range: 2011-01 → 2016-06 (top 50 SKUs, see scripts/extract_m5.py)
DEFAULT_TRAIN_END = "2015-06"
DEFAULT_VAL_END = "2015-12"
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.001


@dataclass(frozen=True)
class ProjectConfig:
    lags: tuple[int, ...]
    forecast_horizon: int
    train_end: str
    val_end: str
    target_column: str
    sku_column: str
    date_column: str
    data_path: Path
    batch_size: int
    epochs: int
    learning_rate: float


def default_config() -> ProjectConfig:
    """Return the default project configuration."""
    print("Loading default project configuration...")
    return ProjectConfig(
        lags=DEFAULT_LAGS,
        forecast_horizon=DEFAULT_FORECAST_HORIZON,
        train_end=DEFAULT_TRAIN_END,
        val_end=DEFAULT_VAL_END,
        target_column="sales",
        sku_column="item_id",
        date_column="date",
        data_path=M5_DATA_PATH,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
    )
