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
    """Load processed data, build features, train Tweedie + Gaussian models, save both."""
    import pandas as pd
    from retail_forecast.features import build_feature_matrix
    from retail_forecast.models.lgbm import LGBMGaussian, LGBMTweedie
    from retail_forecast.tracker import ModelTracker

    processed = config.PROCESSED_DATA_DIR / "sales_long.parquet"
    if not processed.exists():
        print("Processed data not found — running rf-process first...")
        process_data()

    print(f"Loading {processed} ...")
    df = pd.read_parquet(processed)
    print(f"  Shape: {df.shape}")

    feat_cache = config.PROCESSED_DATA_DIR / "feature_matrix.parquet"
    if feat_cache.exists():
        print(f"Loading cached feature matrix ...")
        fm = pd.read_parquet(feat_cache)
        for col in ["cat_id", "dept_id"]:
            if col in fm.columns:
                fm[col] = fm[col].astype("category")
    else:
        print("Building feature matrix (this may take a few minutes)...")
        fm = build_feature_matrix(df)
        fm.to_parquet(feat_cache, index=False)
    print(f"  Feature matrix shape: {fm.shape}")

    # Chronological 80/10/10 split
    dates = fm["date"].sort_values().unique()
    n = len(dates)
    train_end = dates[int(n * 0.80)]
    val_end   = dates[int(n * 0.90)]

    train_df = fm[fm["date"] <= train_end]
    val_df   = fm[(fm["date"] > train_end) & (fm["date"] <= val_end)]
    test_df  = fm[fm["date"] > val_end]
    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} | Test: {len(test_df):,}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for ModelClass, name in [(LGBMTweedie, "tweedie"), (LGBMGaussian, "gaussian")]:
        print(f"\nTraining LightGBM ({name}) ...")
        model = ModelClass(num_boost_round=1000, early_stopping_rounds=50)
        model.fit(train_df, val_df)
        model.save(config.MODELS_DIR / f"lgbm_{name}.pkl")

        tracker = ModelTracker(f"LGBM_{name.capitalize()}")
        tracker.log_data(fm, train_end, val_end)
        tracker.log_params(model.params,
                           num_boost_round=model.num_boost_round,
                           early_stopping_rounds=model.early_stopping_rounds)
        tracker.log_training(model)
        tracker.log_metrics(model, val_df, test_df)
        tracker.log_feature_importance(model.feature_importance)
        tracker.save()
        tracker.print_report()

    print(f"\nDone. Models saved to {config.MODELS_DIR}")


def predict():
    """Load the Tweedie model and run inference on the test set, save predictions."""
    import pandas as pd
    from retail_forecast.features import build_feature_matrix
    from retail_forecast.models.lgbm import LGBMForecast
    from retail_forecast.predict import predict as run_predict

    model_path = config.MODELS_DIR / "lgbm_tweedie.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run rf-train first.")

    processed = config.PROCESSED_DATA_DIR / "sales_long.parquet"
    if not processed.exists():
        raise FileNotFoundError(f"Processed data not found. Run rf-process first.")

    print(f"Loading model from {model_path} ...")
    model = LGBMForecast.load(model_path)

    print("Loading and featurising data ...")
    df = pd.read_parquet(processed)
    fm = build_feature_matrix(df)

    dates = fm["date"].sort_values().unique()
    test_df = fm[fm["date"] > dates[int(len(dates) * 0.90)]]
    print(f"  Running inference on {len(test_df):,} test rows ...")

    result = run_predict(model, test_df)

    out_path = config.REPORTS_DIR / "predictions.parquet"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)
    print(f"Predictions saved → {out_path}")
