"""Test-set inference helper.

predict()  - run the fitted model on a feature matrix and return a
             DataFrame with actual vs. predicted values. Used for
             backtest evaluation and model-insight visualisations.

For generating the real 28-day future forecast, see forecast.forecast_future().
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
        DataFrame with [id, date, sales, y_pred], predictions clipped to >= 0.
    """
    preds = np.maximum(model.predict(df), 0)
    result = df[[ID_COL, DATE_COL, TARGET_COL]].copy()
    result["y_pred"] = preds
    return result
