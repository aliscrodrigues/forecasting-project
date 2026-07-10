import pandas as pd

REQUIRED_COLUMNS = {"date", "item_id", "sales"}
OPTIONAL_COLUMNS = {"store_id"}


def load_sales_data(path: str) -> pd.DataFrame:
    """Load raw sales data from a CSV file."""
    print(f"Loading sales data: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"  → Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def validate_sales_schema(data: pd.DataFrame) -> pd.DataFrame:
    """Validate that the raw sales data contains the required columns."""
    print("Validating sales data schema...")
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(data["date"]):
        data["date"] = pd.to_datetime(data["date"])
    # Ensure sales is numeric
    if not pd.api.types.is_numeric_dtype(data["sales"]):
        raise ValueError("Column 'sales' must be numeric")
    print("  → Schema validated successfully")
    return data
