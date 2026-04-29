# Retail Demand Forecasting

End-to-end retail demand forecasting pipeline using the M5 Accuracy Competition dataset.
Built as the DS570 final project.

**Scope:** CA_1 store × 3,049 products × 3 categories (HOBBIES, HOUSEHOLD, FOODS)
**Model:** LightGBM regression

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

## Design Decisions

- **LightGBM Tweedie:** M5 competition winners used tree-based ensembles with Tweedie loss. Tweedie is mathematically suited for zero-inflated count data; standard L2 (Gaussian) produces biased predictions when zeros dominate.
- **CA_1 subset:** Single-store focus removes store-level heterogeneity, making the category-level intermittency comparison (the novelty claim) cleaner. Subset size (~50-80 MB) fits the <5 min Docker build+run budget.
- **GitHub Release for data:** Public, versioned, no auth required — satisfies the "no local file / no account" reproducibility requirement.
- **uv over pip:** Deterministic installs via `uv.lock`, faster resolution, same lock file used in Dockerfile and local dev.

---

## Limitations & Future Work

- **CA_1 only:** Cross-store generalization is not evaluated; model may not transfer to TX/WI stores directly.