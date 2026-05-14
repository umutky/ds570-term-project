"""Recursive 28-day ahead demand forecast.

For a 28-day horizon the recursion boundary is:
  - sales_lag_28  on forecast day N: always from historical data (N-28 <= last_date).
  - sales_lag_365 on any forecast day: always from historical data.
  - sales_lag_7   days 1-7: historical; days 8-28: prior predictions.
  - sales_lag_14  days 1-14: historical; days 15-28: prior predictions.

Rolling features are recomputed each step from a growing pivot table that
accumulates the predictions from previous steps alongside the historical data.

Calendar features for future dates:
  - Arithmetic fields (wday, month, doy_*) are derived directly from the date.
  - SNAP and event flags are approximated from the same calendar day one year prior.

Prices are carried forward from the last known value per item.
"""

import numpy as np
import pandas as pd

from retail_forecast.features import (
    DATE_COL,
    FEATURE_COLS,
    ID_COL,
    TARGET_COL,
    _EVENT_TYPE_MAP,
)

# M5 wday convention: Sat=1, Sun=2, Mon=3, Tue=4, Wed=5, Thu=6, Fri=7
# Python dayofweek:   Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
_M5_WDAY = [3, 4, 5, 6, 7, 1, 2]

# Derive lag numbers directly from FEATURE_COLS so forecast.py stays in sync
# with features.py automatically when lags are added or removed.
_LAG_NUMS: list[int] = sorted(
    int(c.replace("sales_lag_", ""))
    for c in FEATURE_COLS
    if c.startswith("sales_lag_")
)


def forecast_future(
    model,
    df: pd.DataFrame,
    horizon: int = 28,
) -> pd.DataFrame:
    """Generate recursive demand forecasts for the next `horizon` days.

    Args:
        model: Fitted LGBMForecast (or any object with .predict(df)).
        df: Processed sales DataFrame in long format (output of rf-process).
            Required columns: id, date, sales, wday, month, year, dept_id,
            cat_id, event_type_1, snap_CA, sell_price.
        horizon: Number of calendar days to forecast ahead (default 28).

    Returns:
        DataFrame with columns [id, date, y_pred] sorted by (id, date).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    last_date = df["date"].max()

    # Keep enough history for lag_365, rolling_60, dept rolling, and buffer.
    hist_start = last_date - pd.Timedelta(days=440)
    pivot = (
        df[df["date"] >= hist_start]
        .pivot_table(index=ID_COL, columns="date", values=TARGET_COL, fill_value=0.0)
    )
    item_ids: list[str] = pivot.index.tolist()
    n_items = int(len(item_ids))

    # Item metadata: dept_id, cat_id, last known price
    meta = df.drop_duplicates(ID_COL).set_index(ID_COL).reindex(item_ids)

    # Carry forward the last non-null sell_price per item
    price_df = df.dropna(subset=["sell_price"]).sort_values("date")
    last_price_series = (
        price_df.groupby(ID_COL)["sell_price"]
        .last()
        .reindex(item_ids)
        .fillna(1.0)
    )
    last_price: np.ndarray = last_price_series.values.astype(float)

    # Trailing 365-day price mean per item (mirrors the feature engineering logic)
    trailing_mean_series = (
        price_df.groupby(ID_COL)["sell_price"]
        .apply(lambda s: s.tail(365).mean())
        .reindex(item_ids)
        .fillna(1.0)
    )
    price_rel_year: np.ndarray = (last_price / trailing_mean_series.values).astype(float)

    # Preserve category encoding from training data
    dept_cats = df["dept_id"].astype("category").cat.categories
    cat_cats = df["cat_id"].astype("category").cat.categories

    dept_values = pd.Categorical(meta["dept_id"].values, categories=dept_cats)
    cat_values = pd.Categorical(meta["cat_id"].values, categories=cat_cats)

    # Precompute dept integer codes for vectorized bincount (dept rolling means)
    dept_codes: np.ndarray = pd.Categorical(
        meta["dept_id"].values, categories=dept_cats
    ).codes.astype(int)
    n_depts = len(dept_cats)

    # Calendar approximation for future dates
    cal = (
        df[["date", "snap_CA", "event_type_1"]]
        .drop_duplicates("date")
        .set_index("date")
    )

    def _snap(date: pd.Timestamp) -> int:
        proxy = date - pd.Timedelta(days=364)
        return int(cal["snap_CA"].get(proxy, 0))

    def _has_event(date: pd.Timestamp) -> int:
        proxy = date - pd.Timedelta(days=364)
        val = cal["event_type_1"].get(proxy, None)
        return 0 if (val is None or (isinstance(val, float) and np.isnan(val))) else 1

    def _event_type_enc(date: pd.Timestamp) -> int:
        proxy = date - pd.Timedelta(days=364)
        val = cal["event_type_1"].get(proxy, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0
        return _EVENT_TYPE_MAP.get(val, 0)

    # Recursive forecasting loop
    all_steps: list[pd.DataFrame] = []

    for step in range(1, horizon + 1):
        fdate: pd.Timestamp = last_date + pd.Timedelta(days=step)
        doy = int(fdate.day_of_year)

        # Vectorised lag lookups (one pivot column per lag)
        def _lag(days: int) -> np.ndarray:
            d = fdate - pd.Timedelta(days=days)
            return pivot[d].values if d in pivot.columns else np.zeros(n_items)

        lag_arrays = {f"sales_lag_{n}": _lag(n) for n in _LAG_NUMS}

        # Rolling window: 60 lagged dates, vectorised via column_stack
        roll_cols = [fdate - pd.Timedelta(days=i) for i in range(1, 61)]
        roll_matrix = np.column_stack([
            pivot[d].values if d in pivot.columns else np.zeros(n_items)
            for d in roll_cols
        ])  # shape: (n_items, 60)

        rm7  = roll_matrix[:, :7].mean(axis=1)
        rm14 = roll_matrix[:, :14].mean(axis=1)
        rm28 = roll_matrix[:, :28].mean(axis=1)
        rm30 = roll_matrix[:, :30].mean(axis=1)
        rm60 = roll_matrix.mean(axis=1)
        rs7  = roll_matrix[:, :7].std(axis=1)
        rs28 = roll_matrix[:, :28].std(axis=1)

        # zero_streak: consecutive zero-sales days ending at fdate-1
        zero_streak_vals = np.zeros(n_items)
        still_counting = np.ones(n_items, dtype=bool)
        for back in range(1, 92):
            d = fdate - pd.Timedelta(days=back)
            if d not in pivot.columns:
                break
            col = pivot[d].values
            is_zero = col == 0
            zero_streak_vals += (still_counting & is_zero).astype(float)
            still_counting = still_counting & is_zero
            if not still_counting.any():
                break

        # days_since_last_sale: calendar days since last nonzero sale before fdate
        days_since_vals = np.full(n_items, 365.0)
        found = np.zeros(n_items, dtype=bool)
        for back in range(1, 366):
            d = fdate - pd.Timedelta(days=back)
            if d not in pivot.columns:
                break
            col = pivot[d].values
            newly_found = ~found & (col > 0)
            days_since_vals[newly_found] = float(back)
            found |= newly_found
            if found.all():
                break

        # Department rolling mean: sum of dept daily sales, vectorised via bincount
        dept_d7  = np.zeros(n_depts)
        dept_d28 = np.zeros(n_depts)
        count7, count28 = 0, 0
        for d_back in range(1, 29):
            d = fdate - pd.Timedelta(days=d_back)
            if d in pivot.columns:
                dept_sum = np.bincount(dept_codes, weights=pivot[d].values, minlength=n_depts)
                if d_back <= 7:
                    dept_d7 += dept_sum
                    count7 += 1
                dept_d28 += dept_sum
                count28 += 1
        dept_rm7  = (dept_d7  / max(count7,  1))[dept_codes]
        dept_rm28 = (dept_d28 / max(count28, 1))[dept_codes]

        # Calendar scalars (broadcast across all items in the DataFrame constructor)
        step_df = pd.DataFrame(
            {
                ID_COL:   item_ids,
                DATE_COL: fdate,
                "dept_id": dept_values,
                "cat_id":  cat_values,
                **lag_arrays,
                # Rolling means
                "sales_rolling_mean_7":  rm7,
                "sales_rolling_mean_14": rm14,
                "sales_rolling_mean_28": rm28,
                "sales_rolling_mean_30": rm30,
                "sales_rolling_mean_60": rm60,
                # Rolling standard deviations
                "sales_rolling_std_7":   rs7,
                "sales_rolling_std_28":  rs28,
                # Intermittency
                "zero_streak":           zero_streak_vals,
                "days_since_last_sale":  days_since_vals,
                # Hierarchical
                "dept_rolling_mean_7":   dept_rm7,
                "dept_rolling_mean_28":  dept_rm28,
                # Calendar
                "wday":               _M5_WDAY[fdate.dayofweek],
                "month":              fdate.month,
                "is_weekend":         int(fdate.dayofweek >= 5),
                "is_month_start":     int(fdate.day <= 3),
                "is_month_end":       int(fdate.day >= 28),
                "has_event":          _has_event(fdate),
                "event_type_encoded": _event_type_enc(fdate),
                "has_snap":           _snap(fdate),
                "doy_sin":            float(np.sin(2 * np.pi * doy / 365.25)),
                "doy_cos":            float(np.cos(2 * np.pi * doy / 365.25)),
                "week_of_year":       int(fdate.isocalendar().week),
                # Price (carry forward, no change assumed)
                "sell_price":       last_price,
                "price_change_pct": 0.0,
                "price_rel_year":   price_rel_year,
            }
        )

        preds: np.ndarray = np.maximum(model.predict(step_df), 0.0)

        # Add predictions to pivot so subsequent steps can use them as lags
        pivot[fdate] = preds

        all_steps.append(
            pd.DataFrame(
                {ID_COL: item_ids, DATE_COL: fdate, "y_pred": preds}
            )
        )

    result = pd.concat(all_steps, ignore_index=True)
    return result.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
