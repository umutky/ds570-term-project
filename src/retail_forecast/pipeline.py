"""sklearn Pipeline for demand forecasting.

Feature engineering (lag/rolling/calendar/price) is performed before this
Pipeline, because those transformations require temporal ordering and per-item
grouping that sklearn's column-wise API cannot express without custom code.

This module wraps the downstream preprocessing and modelling step:
  - ColumnTransformer: numerical features (passthrough) + categorical features (OrdinalEncoder)
  - LGBMRegressor with Tweedie or Gaussian objective

Usage:
    from retail_forecast.pipeline import build_pipeline
    from retail_forecast.features import FEATURE_COLS, TARGET_COL

    pipe = build_pipeline(objective="tweedie")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].values

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test[FEATURE_COLS])

Design notes:
- LightGBM is a gradient-boosted tree model; feature scaling has no effect on
  split-finding and would only waste compute. Numerical columns are passed
  through unchanged to make this explicit.
- Categorical columns (dept_id, cat_id) are ordinally encoded so sklearn's
  Pipeline can handle them as a homogeneous array. LightGBM treats the
  resulting integer codes as ordered values, which is acceptable for tree
  models since the optimal thresholds are learned from data.
- Early stopping is not available through the sklearn LGBMRegressor interface
  in the same way as the native lgb.train() API. For training with
  validation-set early stopping, use LGBMForecast in lgbm.py.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from retail_forecast.features import FEATURE_COLS, TARGET_COL

NUMERIC_FEATURE_COLS: list[str] = [c for c in FEATURE_COLS if c not in ("dept_id", "cat_id")]
CATEGORICAL_FEATURE_COLS: list[str] = ["dept_id", "cat_id"]

_DEFAULT_LGBM_PARAMS: dict = {
    "learning_rate": 0.05,
    "num_leaves": 128,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "n_estimators": 500,
    "verbosity": -1,
    "n_jobs": -1,
}


def build_pipeline(objective: str = "tweedie") -> Pipeline:
    """Build an unfitted sklearn Pipeline: ColumnTransformer then LGBMRegressor.

    Args:
        objective: LightGBM objective. Use 'tweedie' (default) for zero-inflated
            demand or 'regression' for Gaussian/L2 loss as a comparison.

    Returns:
        An unfitted sklearn Pipeline ready for pipe.fit(X, y).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                NUMERIC_FEATURE_COLS,
            ),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURE_COLS,
            ),
        ],
        remainder="drop",
    ).set_output(transform="pandas")  # preserve column names so LGBMRegressor receives a DataFrame

    params = {**_DEFAULT_LGBM_PARAMS, "objective": objective}
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.1

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMRegressor(**params)),
        ]
    )


def fit_pipeline(
    feature_matrix: pd.DataFrame,
    objective: str = "tweedie",
    train_frac: float = 0.80,
) -> Pipeline:
    """Convenience wrapper: build and fit a Pipeline on the feature matrix.

    Performs a chronological train split (first train_frac of dates for
    training). No validation-set early stopping; n_estimators is fixed.

    Args:
        feature_matrix: Output of retail_forecast.features.build_feature_matrix().
        objective: 'tweedie' or 'regression'.
        train_frac: Fraction of unique dates used for training (default 0.80).

    Returns:
        Fitted sklearn Pipeline.
    """
    dates = np.sort(feature_matrix["date"].unique())
    cutoff = dates[int(len(dates) * train_frac)]

    train_df = feature_matrix[feature_matrix["date"] <= cutoff]
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].values

    pipe = build_pipeline(objective)
    pipe.fit(X_train, y_train)
    return pipe
