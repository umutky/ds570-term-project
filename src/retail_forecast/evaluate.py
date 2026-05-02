from typing import Any, Callable

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error.

    denom = sum(|y_true|) so the metric is well-defined even when individual
    actuals are zero — unlike standard MAPE which divides per-row.
    """
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE, MAE, and WMAPE after clipping predictions to >= 0."""
    y_pred_clipped = np.maximum(y_pred, 0)
    return {
        "rmse": rmse(y_true, y_pred_clipped),
        "mae": mae(y_true, y_pred_clipped),
        "wmape": wmape(y_true, y_pred_clipped),
    }


def time_series_backtest(
    df: pd.DataFrame,
    model_factory: Callable[[pd.DataFrame, pd.DataFrame], Any],
    date_col: str = "date",
    n_folds: int = 3,
    horizon: int = 28,
) -> pd.DataFrame:
    """Expanding-window time-series cross-validation.

    For each fold:
      - Train on all data up to the cutoff date (expanding window).
      - Evaluate on the next `horizon` days.
      - model_factory(train_df, val_df) must return a fitted model with .predict(df).

    Cutoffs are spaced evenly over the last 60% of available dates, leaving
    the first 40% always in training (ensures lag_365 features are populated).

    Args:
        df: Feature matrix from build_feature_matrix(), sorted by (id, date).
        model_factory: callable(train_df, val_df) -> fitted model.
        date_col: Name of the date column.
        n_folds: Number of expanding-window folds.
        horizon: Forecast horizon in days per fold.

    Returns:
        DataFrame with columns [fold, cutoff, rmse, mae, wmape, n_rows].
    """
    dates = np.sort(df[date_col].unique())
    n_dates = len(dates)

    # Reserve the first 40% strictly for training history to ensure lag features exist
    eval_start_idx = int(n_dates * 0.40)
    eval_dates = dates[eval_start_idx:]

    if len(eval_dates) < n_folds * horizon:
        raise ValueError(
            f"Not enough dates for {n_folds} folds × {horizon}-day horizon. "
            f"Available eval window: {len(eval_dates)} days."
        )

    step = len(eval_dates) // (n_folds + 1)
    cutoffs = [eval_dates[i * step] for i in range(1, n_folds + 1)]

    results = []
    for fold, cutoff in enumerate(cutoffs, start=1):
        cutoff_ts = pd.Timestamp(cutoff)
        val_start = cutoff_ts + pd.Timedelta(days=1)
        val_end = val_start + pd.Timedelta(days=horizon - 1)

        train_df = df[df[date_col] <= cutoff_ts]
        val_df = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)]

        if len(val_df) == 0:
            continue

        model = model_factory(train_df, val_df)
        preds = model.predict(val_df)
        metrics = evaluate(val_df["sales"].values, preds)

        row = {
            "fold": fold,
            "cutoff": cutoff_ts.date(),
            "n_train": len(train_df),
            "n_val": len(val_df),
            **metrics,
        }
        results.append(row)
        print(
            f"  Fold {fold} | cutoff={cutoff_ts.date()} | "
            f"RMSE={metrics['rmse']:.3f} | MAE={metrics['mae']:.3f} | WMAPE={metrics['wmape']:.4f}"
        )

    return pd.DataFrame(results)
