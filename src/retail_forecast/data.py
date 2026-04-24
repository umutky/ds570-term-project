import pandas as pd
from pathlib import Path
from retail_forecast.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

COMPETITION = "m5-forecasting-accuracy"

REQUIRED_FILES = [
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv",
]


def download_raw() -> None:
    """Download M5 competition files from Kaggle and unzip them into data/raw/.

    Requires a valid Kaggle API token at ~/.kaggle/kaggle.json or the
    KAGGLE_USERNAME and KAGGLE_KEY environment variables.
    """
    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "The 'kaggle' package is required for automatic download. "
            "Run: uv add kaggle"
        )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Authenticating with Kaggle API...")
    kaggle.api.authenticate()

    print(f"Downloading '{COMPETITION}' competition files to {RAW_DATA_DIR} ...")
    kaggle.api.competition_download_files(
        COMPETITION,
        path=str(RAW_DATA_DIR),
        quiet=False,
        force=False,
    )

    # The kaggle package downloads a zip; unzip it.
    zip_path = RAW_DATA_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        import zipfile
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DATA_DIR)
        zip_path.unlink()
        print("Extraction complete.")


def _ensure_raw_data() -> None:
    """Download all required M5 files if any are missing."""
    missing = [f for f in REQUIRED_FILES if not (RAW_DATA_DIR / f).exists()]
    if missing:
        print(f"Missing data files: {missing}")
        print("Starting automatic download from Kaggle...")
        download_raw()

    still_missing = [f for f in REQUIRED_FILES if not (RAW_DATA_DIR / f).exists()]
    if still_missing:
        raise FileNotFoundError(
            f"Files still missing after download attempt: {still_missing}\n"
            "Check your Kaggle credentials (~/.kaggle/kaggle.json) and "
            "that you have accepted the competition rules at:\n"
            "https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules"
        )


def load_raw(file: str) -> pd.DataFrame:
    """Load a CSV from data/raw/, downloading all M5 files automatically if missing.

    Args:
        file: Filename relative to data/raw/, e.g. 'sales_train_evaluation.csv'.

    Returns:
        DataFrame with the file contents.
    """
    path = RAW_DATA_DIR / file
    if not path.exists():
        _ensure_raw_data()
    return pd.read_csv(path)


def select_subset(
    df: pd.DataFrame,
    store_id: str,
    cat_ids: list[str],
) -> pd.DataFrame:
    """Filter the wide-format sales DataFrame to one store and one or more categories.

    Args:
        df: Full sales_train_evaluation DataFrame (wide format).
        store_id: Store identifier, e.g. 'CA_1'.
        cat_ids: Category identifiers, e.g. ['FOODS_3'].

    Returns:
        Filtered DataFrame reset to a zero-based index.
    """
    mask = (df["store_id"] == store_id) & (df["cat_id"].isin(cat_ids))
    subset = df.loc[mask].reset_index(drop=True)
    if subset.empty:
        raise ValueError(
            f"No rows found for store_id={store_id!r}, cat_ids={cat_ids}. "
            "Check your filter values against the available stores and categories."
        )
    return subset


def melt_sales_long(
    df: pd.DataFrame,
    calendar_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert wide-format sales data to long format.

    Input columns:  id, item_id, dept_id, cat_id, store_id, state_id, d_1 ... d_1941
    Output columns: id, item_id, dept_id, cat_id, store_id, state_id, d, sales
                    plus calendar columns when calendar_df is provided.

    Args:
        df: Wide-format sales DataFrame (one row per item, one column per day).
        calendar_df: Optional calendar DataFrame to join date and event metadata.

    Returns:
        Long-format DataFrame sorted by (id, d).
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]

    long = df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )

    if calendar_df is not None:
        cal_cols = [
            "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
            "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ]
        keep = [c for c in cal_cols if c in calendar_df.columns]
        long = long.merge(calendar_df[keep], on="d", how="left")
        long["date"] = pd.to_datetime(long["date"])

    long = long.sort_values(["id", "d"]).reset_index(drop=True)
    return long


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Save a DataFrame to data/processed/<name>.parquet.

    Args:
        df: DataFrame to persist.
        name: Base filename without extension, e.g. 'sales_long'.

    Returns:
        Path to the written parquet file.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df):,} rows -> {path}")
    return path
