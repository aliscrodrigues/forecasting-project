"""Download M5 via Nixtla and prepare daily sales for the forecasting pipeline.

Run from project root:

    uv run python scripts/extract_m5.py

Output: data/m5/processed/m5_daily_sales.parquet
Columns: date, item_id, store_id, sales

If automatic download fails (SSL/proxy), download manually in your browser:
  https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip
Save the file as: data/m5/datasets/m5.zip
Then run this script again (it will extract the zip locally).
"""

import ssl
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd
from datasetsforecast.m5 import M5

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "m5" / "datasets"
OUTPUT_PATH = DATA_DIR / "m5" / "processed" / "m5_daily_sales.parquet"
CALENDAR_FILE = RAW_PATH / "calendar.csv"
ZIP_PATH = RAW_PATH / "m5.zip"
SOURCE_URL = M5.source_url
TOP_N_SKUS = 50


def download_m5_zip(url: str, dest: Path) -> None:
    """Download m5.zip (urllib + certifi, then curl fallback on macOS/proxy setups)."""
    print(f"Downloading {url} ...")

    try:
        context = ssl.create_default_context(cafile=certifi.where())
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, context=context, timeout=120) as response:
            with dest.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        print(f"  → Saved to {dest}")
        return
    except Exception as urllib_error:
        print(f"  urllib download failed ({urllib_error}); trying curl...")

    result = subprocess.run(
        ["curl", "-fL", "--retry", "3", "-o", str(dest), url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    print(f"  → Saved to {dest} (via curl)")


def extract_m5_zip(zip_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dest_dir)
    print(f"  → Extracted to {dest_dir}")


def download_datasets() -> Path:
    """Download and extract raw M5 CSV files into data/m5/datasets/."""
    print("─── Step 1: Download raw M5 datasets ───")

    if CALENDAR_FILE.exists():
        print(f"Raw files already present at {RAW_PATH}")
        return RAW_PATH

    RAW_PATH.mkdir(parents=True, exist_ok=True)

    if not ZIP_PATH.exists():
        try:
            download_m5_zip(SOURCE_URL, ZIP_PATH)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not download M5 ({exc}).\n"
                f"Download manually in your browser:\n  {SOURCE_URL}\n"
                f"Save as: {ZIP_PATH}\n"
                "Then run this script again."
            ) from exc

    if not CALENDAR_FILE.exists():
        extract_m5_zip(ZIP_PATH, RAW_PATH)

    if not CALENDAR_FILE.exists():
        raise FileNotFoundError(
            f"Expected {CALENDAR_FILE} after extraction. "
            f"Check that {ZIP_PATH} is a valid M5 archive."
        )

    print(f"  → Raw datasets ready at {RAW_PATH}")
    return RAW_PATH


def to_daily_sales(Y_df: pd.DataFrame, S_df: pd.DataFrame) -> pd.DataFrame:
    """Map Nixtla tables to project schema: date, item_id, store_id, sales."""
    static = S_df[["unique_id", "item_id", "store_id"]].copy()
    static["item_id"] = static["item_id"].astype(str)
    static["store_id"] = static["store_id"].astype(str)

    daily = Y_df.merge(static, on="unique_id", how="inner")
    daily = daily.rename(columns={"ds": "date", "y": "sales"})
    daily = daily[["date", "item_id", "store_id", "sales"]]
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)


def select_top_skus(daily: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep only the top-N item_ids by total demand across all stores and dates."""
    demand_by_sku = daily.groupby("item_id")["sales"].sum()
    top_skus = demand_by_sku.nlargest(top_n).index
    filtered = daily[daily["item_id"].isin(top_skus)].reset_index(drop=True)
    print(f"  → Selected top {top_n} SKUs (of {demand_by_sku.shape[0]} total)")
    return filtered


def prepare_project_dataset(
    data_dir: Path = DATA_DIR,
    output_path: Path = OUTPUT_PATH,
    top_n_skus: int = TOP_N_SKUS,
) -> pd.DataFrame:
    """Load M5, filter to top SKUs, and save project-format daily sales."""
    print("─── Step 2: Prepare dataset for project pipeline ───")
    print("Loading M5 tables...")
    Y_df, _, S_df = M5.load(str(data_dir))

    daily = to_daily_sales(Y_df, S_df)
    print(f"  → Loaded {len(daily):,} daily rows ({daily['item_id'].nunique()} SKUs)")

    daily = select_top_skus(daily, top_n_skus)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_path, index=False)

    print(f"Saved {len(daily):,} rows to {output_path}")
    print(
        f"  {daily['item_id'].nunique()} SKUs, {daily['store_id'].nunique()} stores, "
        f"{daily['date'].min().date()} → {daily['date'].max().date()}"
    )
    return daily


def main() -> None:
    download_datasets()
    prepare_project_dataset()


if __name__ == "__main__":
    main()
