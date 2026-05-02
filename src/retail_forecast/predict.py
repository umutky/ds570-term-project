"""Inference and what-if scenario generation.

predict()         — standard inference, returns actual vs. predicted DataFrame.
what_if_price()   — simulate a uniform price change across all items.
what_if_event()   — simulate adding or removing an event on all days.
"""

import numpy as np
import pandas as pd

from retail_forecast.features import DATE_COL, ID_COL, TARGET_COL


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run model inference and return actual vs. predicted values.

    Args:
        model: Fitted LGBMForecast (or any object with .predict(df)).
        df: Feature matrix from build_feature_matrix().

    Returns:
        DataFrame with [id, date, sales, y_pred] — predictions clipped to >= 0.
    """
    preds = np.maximum(model.predict(df), 0)
    result = df[[ID_COL, DATE_COL, TARGET_COL]].copy()
    result["y_pred"] = preds
    return result


def what_if_price(
    model,
    df: pd.DataFrame,
    price_multiplier: float,
) -> pd.DataFrame:
    """Simulate a uniform price change and compare forecasts to baseline.

    Scales sell_price by price_multiplier and recomputes price features
    in-place (approximation — does not re-derive lag/rolling features).

    Args:
        model: Fitted model.
        df: Feature matrix (must include sell_price, price_change_pct, price_rel_year).
        price_multiplier: e.g. 1.1 for +10% price increase, 0.9 for -10%.

    Returns:
        DataFrame with [id, date, sales, y_pred_baseline, y_pred_whatif, price_multiplier].
    """
    baseline_preds = np.maximum(model.predict(df), 0)

    modified = df.copy()
    if "sell_price" in modified.columns:
        modified["sell_price"] = modified["sell_price"] * price_multiplier
    if "price_rel_year" in modified.columns:
        modified["price_rel_year"] = modified["price_rel_year"] * price_multiplier
    if "price_change_pct" in modified.columns:
        # Approximate the incremental price change from the shift
        modified["price_change_pct"] = price_multiplier - 1.0

    whatif_preds = np.maximum(model.predict(modified), 0)

    result = df[[ID_COL, DATE_COL, TARGET_COL]].copy()
    result["y_pred_baseline"] = baseline_preds
    result["y_pred_whatif"] = whatif_preds
    result["price_multiplier"] = price_multiplier
    return result


def what_if_event(
    model,
    df: pd.DataFrame,
    add_event: bool = True,
    event_label: str = "Sporting",
) -> pd.DataFrame:
    """Simulate adding or removing an event flag on all rows.

    Flips has_event to 1 (add) or 0 (remove) across the entire dataset.
    Useful for visualising event sensitivity in the Streamlit what-if page.

    Args:
        model: Fitted model.
        df: Feature matrix (must include has_event).
        add_event: True → set has_event=1; False → set has_event=0.
        event_label: Human-readable label for the scenario (not used by model).

    Returns:
        DataFrame with [id, date, sales, y_pred_baseline, y_pred_whatif, event_scenario].
    """
    baseline_preds = np.maximum(model.predict(df), 0)

    modified = df.copy()
    if "has_event" in modified.columns:
        modified["has_event"] = int(add_event)

    whatif_preds = np.maximum(model.predict(modified), 0)

    result = df[[ID_COL, DATE_COL, TARGET_COL]].copy()
    result["y_pred_baseline"] = baseline_preds
    result["y_pred_whatif"] = whatif_preds
    result["event_scenario"] = f"{'add' if add_event else 'remove'} {event_label}"
    return result
