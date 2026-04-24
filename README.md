# Retail Demand Forecasting

End-to-end retail demand forecasting pipeline using the M5 Accuracy Competition dataset (Walmart).
Built as the DS570 final project.

**Model:** LightGBM regression with lag, rolling, calendar, and price features.
**Scope:** Full M5 dataset -- 30,490 items x 10 stores x 1,941 days.

---

## Data Download

The M5 dataset is **not included** in this repository. Download it from Kaggle:

1. Install the Kaggle CLI: `pip install kaggle` (or `uv tool install kaggle`)
2. Set up your Kaggle API key (`~/.kaggle/kaggle.json`)
3. Download and unzip into `data/raw/`:

```bash
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d data/raw/
```

Required files in `data/raw/`:
- `sales_train_evaluation.csv` — 30,490 items × 1,941 days
- `calendar.csv` — daily date/event metadata
- `sell_prices.csv` — weekly item prices per store

---

## Quickstart (Local)

```bash
# 1. Install dependencies (requires uv)
uv sync --all-extras

# 2. Download M5 data (see above), then process it
uv run rf-process

# 3. Train the model
uv run rf-train

# 4. Run the dashboard
uv run streamlit run app/streamlit_app.py
```

---

## Quickstart (Docker)

```bash
# Build the image
docker build -t retail-forecast .

# Process data (bind-mount your local data/ folder)
docker run --rm -v $(pwd)/data:/app/data retail-forecast rf-process

# Train
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs retail-forecast rf-train

# Serve dashboard
docker run --rm -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  retail-forecast streamlit run app/streamlit_app.py --server.address=0.0.0.0
```

---

## Project Structure

```
retail-demand-forecasting/
├── data/
│   ├── raw/          # M5 CSVs — not committed (gitignored)
│   └── processed/    # Feature-engineered parquets — not committed
├── notebooks/        # EDA and experiment notebooks
├── src/
│   └── retail_forecast/
│       ├── config.py    # Path constants
│       ├── data.py      # Raw loading, Kaggle auto-download, wide-to-long transform
│       ├── features.py  # Lag / rolling / calendar / price features
│       ├── model.py     # LightGBM wrapper
│       ├── evaluate.py  # Metrics & backtesting
│       └── cli.py       # Entry points: rf-train, rf-predict, rf-process
├── app/
│   └── streamlit_app.py  # Dashboard (Week 4)
├── outputs/
│   ├── models/       # Saved models
│   └── reports/      # Metrics and figures
├── tests/
├── Dockerfile
└── pyproject.toml
```

---