from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"date", "item_id", "sales"}
OPTIONAL_COLUMNS = {"store_id"}


def load_sales_data(path: str | Path) -> pd.DataFrame:
    """Load raw sales data from a CSV or Parquet file.

    Expected columns: ``date``, ``item_id``, ``sales``, and optionally ``store_id``.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Sales data file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=["date"])
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Expected .csv or .parquet."
        )

    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    return df


def validate_sales_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Validate that the raw sales data contains the required columns."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(data["date"]):
        data["date"] = pd.to_datetime(data["date"])
    if not pd.api.types.is_numeric_dtype(data["sales"]):
        raise ValueError("Column 'sales' must be numeric")
    return data
