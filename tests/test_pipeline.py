"""Smoke tests for the sklearn Pipeline (pipeline.py)."""

import numpy as np
import pandas as pd
import pytest

from retail_forecast.features import FEATURE_COLS, TARGET_COL, build_feature_matrix
from retail_forecast.pipeline import (
    CATEGORICAL_FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
    build_pipeline,
    fit_pipeline,
)


@pytest.fixture()
def feature_matrix_400d() -> pd.DataFrame:
    """400-day, 2-item feature matrix, long enough for lag_365 features."""
    dates = pd.date_range("2019-01-01", periods=400, freq="D")
    rng = np.random.default_rng(42)
    rows = []
    for item in ["ITEM_A", "ITEM_B"]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "id": item, "item_id": item, "dept_id": "FOODS_1", "cat_id": "FOODS",
                    "store_id": "CA_1", "state_id": "CA",
                    "sales": int(rng.integers(0, 8)),
                    "date": date, "wday": (i % 7) + 1, "month": date.month, "year": date.year,
                    "event_type_1": None, "snap_CA": 0,
                    "sell_price": 2.5 + rng.uniform(-0.1, 0.1),
                    "wm_yr_wk": 11101 + i // 7,
                }
            )
    df = pd.DataFrame(rows)
    fm = build_feature_matrix(df)
    if len(fm) == 0:
        pytest.skip("Feature matrix empty. Dataset too short for lag_365.")
    return fm


class TestFeatureColSplit:
    def test_no_overlap(self) -> None:
        """NUMERIC and CATEGORICAL feature lists must be disjoint."""
        assert set(NUMERIC_FEATURE_COLS).isdisjoint(set(CATEGORICAL_FEATURE_COLS))

    def test_union_equals_feature_cols(self) -> None:
        """Together they must cover exactly FEATURE_COLS."""
        assert set(NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS) == set(FEATURE_COLS)


class TestBuildPipeline:
    def test_returns_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline
        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_preprocessor_and_model(self) -> None:
        pipe = build_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert "preprocessor" in step_names
        assert "model" in step_names

    def test_tweedie_objective(self) -> None:
        pipe = build_pipeline(objective="tweedie")
        assert pipe.named_steps["model"].objective == "tweedie"

    def test_gaussian_objective(self) -> None:
        pipe = build_pipeline(objective="regression")
        assert pipe.named_steps["model"].objective == "regression"


class TestFitPipeline:
    def test_fits_and_predicts(self, feature_matrix_400d: pd.DataFrame) -> None:
        pipe = fit_pipeline(feature_matrix_400d, objective="tweedie")
        X_test = feature_matrix_400d[FEATURE_COLS]
        preds = pipe.predict(X_test)
        assert len(preds) == len(X_test)
        assert np.isfinite(preds).all()

    def test_tweedie_predictions_nonnegative(self, feature_matrix_400d: pd.DataFrame) -> None:
        pipe = fit_pipeline(feature_matrix_400d, objective="tweedie")
        preds = pipe.predict(feature_matrix_400d[FEATURE_COLS])
        assert (preds >= 0).all(), "Tweedie output must be non-negative"

    def test_gaussian_fits(self, feature_matrix_400d: pd.DataFrame) -> None:
        pipe = fit_pipeline(feature_matrix_400d, objective="regression")
        preds = pipe.predict(feature_matrix_400d[FEATURE_COLS])
        assert len(preds) == len(feature_matrix_400d)
        assert np.isfinite(preds).all()

    def test_column_transformer_present(self, feature_matrix_400d: pd.DataFrame) -> None:
        from sklearn.compose import ColumnTransformer
        pipe = fit_pipeline(feature_matrix_400d)
        assert isinstance(pipe.named_steps["preprocessor"], ColumnTransformer)
