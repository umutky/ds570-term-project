from retail_forecast import config


def train():
    print("hello from rf-train! Pipeline is starting...")
    print(f"Models will be saved to: {config.MODELS_DIR}")


def predict():
    print("hello from rf-predict! Making predictions...")


def process_data():
    """Load the full M5 dataset, melt to long format, and save as a processed parquet."""
    from retail_forecast.data import load_raw, melt_sales_long, save_processed

    print(f"Loading sales data from {config.RAW_DATA_DIR} ...")
    sales = load_raw("sales_train_evaluation.csv")
    print(f"  Raw shape: {sales.shape}")

    print("Loading calendar data...")
    calendar = load_raw("calendar.csv")

    print("Melting to long format and joining calendar...")
    long_df = melt_sales_long(sales, calendar_df=calendar)
    print(f"  Long shape: {long_df.shape}")

    save_processed(long_df, "sales_long")
    print("Done. Run the EDA notebook to explore the data.")
