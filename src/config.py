import os
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
# fake_sales.csv spans 2014-01 → 2015-12 (see data/sample/fake_sales.csv)
SAMPLE_REFERENCE_MONTH = "2015-10"
SAMPLE_VALIDATION_MONTHS = 2

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


def _resolve_data_path() -> Path:
    """Resolve dataset path from env override or filesystem defaults.

    Set ``FORECASTING_DATA=sample`` or ``FORECASTING_DATA=m5`` to force a source.
    Otherwise prefer the M5 parquet when present, else the sample CSV.
    """
    override = os.environ.get("FORECASTING_DATA", "").strip().lower()
    if override == "sample":
        return SAMPLE_DATA_PATH
    if override == "m5":
        return M5_DATA_PATH
    if M5_DATA_PATH.exists():
        return M5_DATA_PATH
    return SAMPLE_DATA_PATH


def _is_sample_data_path(data_path: Path) -> bool:
    return data_path.resolve() == SAMPLE_DATA_PATH.resolve()


def default_config(data_path: Path | None = None) -> ProjectConfig:
    """Return the default project configuration."""
    resolved_path = data_path or _resolve_data_path()
    use_sample_split = _is_sample_data_path(resolved_path)
    return ProjectConfig(
        lags=DEFAULT_LAGS,
        rolling_windows=DEFAULT_ROLLING_WINDOWS,
        forecast_horizon=DEFAULT_FORECAST_HORIZON,
        reference_month=(
            SAMPLE_REFERENCE_MONTH if use_sample_split else DEFAULT_REFERENCE_MONTH
        ),
        validation_months=(
            SAMPLE_VALIDATION_MONTHS if use_sample_split else DEFAULT_VALIDATION_MONTHS
        ),
        target_column="sales",
        sku_column="item_id",
        date_column="date",
        data_path=resolved_path,
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


def sample_config() -> ProjectConfig:
    """Return configuration for the development sample CSV."""
    return default_config(SAMPLE_DATA_PATH)
