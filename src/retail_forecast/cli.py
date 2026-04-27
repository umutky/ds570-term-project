from retail_forecast import config


def fetch_data():
    """Download the CA_1 subset parquet from GitHub Release (cached after first run)."""
    from retail_forecast.data.fetch import fetch
    fetch()


def process_data():
    """Fetch the CA_1 subset, load it, and save as processed parquet."""
    from retail_forecast.data.fetch import fetch
    from retail_forecast.data.load import load_subset, save_processed

    path = fetch()
    print(f"Loading subset from {path} ...")
    df = load_subset(path)
    print(f"  Shape: {df.shape}")
    save_processed(df, "sales_long")
    print("Done. Run the EDA notebook to explore the data.")


def train():
    print("hello from rf-train! Pipeline is starting...")
    print(f"Models will be saved to: {config.MODELS_DIR}")


def predict():
    print("hello from rf-predict! Making predictions...")
