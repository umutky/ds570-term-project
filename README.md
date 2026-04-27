# Retail Demand Forecasting

End-to-end retail demand forecasting pipeline using the M5 Accuracy Competition dataset.
Built as the DS570 final project.

**Scope:** CA_1 store × 3,049 products × 3 categories (HOBBIES, HOUSEHOLD, FOODS)
**Model:** LightGBM regression with Tweedie objective (intermittent / zero-inflated demand)
**Novelty:** Tweedie vs. Gaussian loss comparison on intermittent demand + interactive what-if simulator

---

## Quickstart (Docker)

No local files needed. The subset data is fetched automatically from GitHub Releases.

```bash
# Build the image (~2-3 min first time)
docker build -t retail-forecast .

# Fetch data + launch dashboard
docker run --rm -p 8501:8501 retail-forecast
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Quickstart (Local)

```bash
# 1. Install dependencies (requires uv)
uv sync --all-extras

# 2. Fetch the CA_1 subset from GitHub Releases
uv run rf-fetch

# 3. Process into parquet
uv run rf-process

# 4. Train the model
uv run rf-train

# 5. Launch the dashboard
uv run streamlit run app/streamlit_app.py
```

---

## Data

**Source:** [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy), Makridakis Open Forecasting Center, 2020.

**Subset:** `store_id == 'CA_1'` × all products × all dates
- 3,049 time series (items)
- 3 categories: HOBBIES, HOUSEHOLD, FOODS (7 departments)
- 1,941 days (2011-01-29 to 2016-06-19)
- ~5.9M rows in long format

**Why CA_1?** See `notebooks/2026-05-04-ko-01-eda-global.ipynb` — comparative analysis across all 10 M5 stores justifies this selection (data integrity, category diversity, zero-inflation range, timing budget).

**Runtime fetch:** The subset is hosted as a [GitHub Release asset](https://github.com/umutky/retail-demand-forecasting/releases/tag/v0.1.0-data).
`rf-fetch` downloads it automatically on first run and caches it at `data/raw/m5_ca1_subset.parquet`.
No Kaggle account needed.

**Reproducibility:** To rebuild the subset from the full M5 CSVs:
```bash
# Download full M5 data into data/raw/ from Kaggle first, then:
uv run python scripts/build_subset.py
```

---

## Project Structure

```
retail-demand-forecasting/
├── data/
│   ├── raw/          # m5_ca1_subset.parquet — fetched at runtime (.gitignore'd)
│   └── processed/    # feature-engineered parquets (.gitignore'd)
├── notebooks/
│   ├── 2026-05-04-ko-01-eda-global.ipynb     # All M5: store comparison → CA_1 selection
│   └── 2026-05-04-ko-02-eda-ca1-focused.ipynb # CA_1 deep-dive: intermittent demand
├── scripts/
│   └── build_subset.py   # One-time: builds CA_1 subset from full M5 CSVs
├── src/
│   └── retail_forecast/
│       ├── config.py     # Paths, DATA_URL, DATA_SHA256
│       ├── data/
│       │   ├── fetch.py  # HTTP fetch + SHA256 verification + caching
│       │   └── load.py   # Parquet loading, wide-to-long, save_processed
│       ├── features.py   # Lag / rolling / calendar / price features (Week 3)
│       ├── models/       # LightGBM + baseline models (Week 3)
│       ├── evaluate.py   # RMSE, MAE, WMAPE, backtest (Week 3)
│       └── cli.py        # Entry points: rf-fetch / rf-process / rf-train / rf-predict
├── app/
│   ├── streamlit_app.py  # Main dashboard
│   └── pages/            # Multi-page Streamlit (Week 4)
├── tests/
│   └── test_fetch.py     # Fetch logic smoke tests
├── outputs/
│   ├── models/           # Saved models (.gitignore'd)
│   └── reports/          # Metrics and figures (.gitignore'd)
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

---

## Design Decisions

- **LightGBM Tweedie:** M5 competition winners used tree-based ensembles with Tweedie loss. Tweedie is mathematically suited for zero-inflated count data; standard L2 (Gaussian) produces biased predictions when zeros dominate.
- **CA_1 subset:** Single-store focus removes store-level heterogeneity, making the category-level intermittency comparison (the novelty claim) cleaner. Subset size (~50-80 MB) fits the <5 min Docker build+run budget.
- **GitHub Release for data:** Public, versioned, no auth required — satisfies the "no local file / no account" reproducibility requirement.
- **uv over pip:** Deterministic installs via `uv.lock`, faster resolution, same lock file used in Dockerfile and local dev.

---

## Limitations & Future Work

- **CA_1 only:** Cross-store generalization is not evaluated; model may not transfer to TX/WI stores directly.
- **No hierarchical reconciliation:** Item-level forecasts are independent; dept/category-level aggregations may be inconsistent.
- **Cold-start items:** Items with very few non-zero observations (extreme sparsity) are modeled but forecasts are unreliable.
- **Rare events:** Super Bowl, Christmas — very few samples in training data, model is uncertain.
- **Future:** Quantile regression (p10/p50/p90), multi-horizon evaluation, online re-training.
