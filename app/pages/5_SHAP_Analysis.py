import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from retail_forecast.config import MODELS_DIR, PROCESSED_DATA_DIR
from retail_forecast.features import FEATURE_COLS
from retail_forecast.models.lgbm import LGBMForecast

st.title("SHAP Feature Impact Analysis")
st.caption(
    "SHapley Additive exPlanations — direction-aware, per-prediction feature attributions. "
    "Unlike gain-based importance, SHAP values show the marginal contribution of each feature "
    "to each individual prediction with sign and magnitude.  \n"
    "Computed on a **3,000-row random sample** of the held-out test set."
)

N_SHAP = 3_000
CAT_COLORS = {"FOODS": "#636EFA", "HOBBIES": "#EF553B", "HOUSEHOLD": "#00CC96"}
_NUMERIC_FEATURES = [c for c in FEATURE_COLS if c not in ("dept_id", "cat_id")]

feat_path = PROCESSED_DATA_DIR / "feature_matrix.parquet"


@st.cache_resource
def load_model(name: str) -> LGBMForecast | None:
    path = MODELS_DIR / f"lgbm_{name}.pkl"
    return LGBMForecast.load(path) if path.exists() else None


@st.cache_resource
def get_explainer(name: str):
    mdl = load_model(name)
    return shap.TreeExplainer(mdl._booster) if mdl else None


@st.cache_data(show_spinner="Computing SHAP values (first load ~30 s)…")
def compute_shap(n_sample: int = N_SHAP):
    feat = pd.read_parquet(feat_path, columns=FEATURE_COLS + ["date"])
    dates = feat["date"].sort_values().unique()
    val_end = dates[int(len(dates) * 0.90)]
    test_feat = feat[feat["date"] > val_end].drop(columns=["date"])
    for col in ("cat_id", "dept_id"):
        if col in test_feat.columns:
            test_feat[col] = test_feat[col].astype("category")

    sample = test_feat.sample(n=min(n_sample, len(test_feat)), random_state=42)
    cat_ids = sample["cat_id"].astype(str).values
    X_sample = sample[FEATURE_COLS].copy()

    sv: dict[str, np.ndarray] = {}
    for name in ("tweedie", "gaussian"):
        exp = get_explainer(name)
        if exp is not None:
            sv[name] = exp.shap_values(X_sample)

    return sv, X_sample, cat_ids


@st.cache_data(show_spinner="Loading test set rows…")
def load_test_rows() -> pd.DataFrame:
    feat = pd.read_parquet(feat_path, columns=FEATURE_COLS + ["date", "id"])
    dates = feat["date"].sort_values().unique()
    val_end = dates[int(len(dates) * 0.90)]
    test = feat[feat["date"] > val_end].copy()
    for col in ("cat_id", "dept_id"):
        if col in test.columns:
            test[col] = test[col].astype("category")
    return test


# Guards
if load_model("tweedie") is None:
    st.error("Tweedie model not found. Run `rf-train` first.")
    st.stop()

if not feat_path.exists():
    st.error("Feature matrix not found. Run `rf-process` and `rf-train` first.")
    st.stop()

gaussian_ok = load_model("gaussian") is not None
available_models = ["tweedie"] + (["gaussian"] if gaussian_ok else [])

shap_vals, X_sample, cat_ids = compute_shap()


# 1 & 2  Global feature importance
st.subheader("1 — Global Feature Importance")
st.caption(
    "**Bar:** mean |SHAP| — ranks features by average impact magnitude, both models side by side.  \n"
    "**Beeswarm:** each dot is one prediction; color = feature value (red = high, blue = low)."
)

tab_bar, tab_bee = st.tabs(["Mean |SHAP| — Bar", "SHAP Distribution — Beeswarm"])


def _render_bar(key: str, color: str) -> None:
    if key not in shap_vals:
        st.info(f"{key.capitalize()} model not available.")
        return
    fig, _ = plt.subplots(figsize=(5, 5))
    shap.summary_plot(
        shap_vals[key], X_sample,
        plot_type="bar", max_display=15, show=False, color=color,
    )
    plt.tight_layout()
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)


def _render_bee(key: str) -> None:
    if key not in shap_vals:
        st.info(f"{key.capitalize()} model not available.")
        return
    fig, _ = plt.subplots(figsize=(5, 5))
    shap.summary_plot(shap_vals[key], X_sample, max_display=15, show=False)
    plt.tight_layout()
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)


with tab_bar:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tweedie**")
        _render_bar("tweedie", "#00CC96")
    with c2:
        st.markdown("**Gaussian**")
        _render_bar("gaussian", "#FFA15A")

with tab_bee:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Tweedie**")
        _render_bee("tweedie")
    with c2:
        st.markdown("**Gaussian**")
        _render_bee("gaussian")


# 3  Dependence plot
st.divider()
st.subheader("2 — Feature Dependence Plot")
st.caption(
    "How a feature's value drives its SHAP contribution across all sampled predictions. "
    "Color = product category. LOWESS trend line per category."
)

dep_c1, dep_c2 = st.columns([2, 1])
with dep_c1:
    _default = "zero_streak" if "zero_streak" in _NUMERIC_FEATURES else _NUMERIC_FEATURES[0]
    dep_feature = st.selectbox(
        "Feature", _NUMERIC_FEATURES,
        index=_NUMERIC_FEATURES.index(_default),
        key="dep_feature",
    )
with dep_c2:
    dep_model = st.radio(
        "Model", available_models,
        horizontal=True, format_func=str.capitalize, key="dep_model",
    )

dep_idx = FEATURE_COLS.index(dep_feature)
dep_df = pd.DataFrame({
    "x":      X_sample[dep_feature].values,
    "shap":   shap_vals[dep_model][:, dep_idx],
    "cat_id": cat_ids,
})

fig_dep = px.scatter(
    dep_df, x="x", y="shap", color="cat_id",
    color_discrete_map=CAT_COLORS,
    opacity=0.35,
    trendline="lowess",
    labels={
        "x":      dep_feature,
        "shap":   f"SHAP value  ({dep_feature})",
        "cat_id": "Category",
    },
    category_orders={"cat_id": ["FOODS", "HOBBIES", "HOUSEHOLD"]},
    height=440,
)
fig_dep.add_hline(y=0, line_dash="dash", line_color="black", line_width=1,
                  annotation_text="No impact", annotation_position="top right")
fig_dep.update_layout(template="plotly_white")
st.plotly_chart(fig_dep, use_container_width=True)


#  4  Waterfall — single prediction
st.divider()
st.subheader("3 — Single Prediction Waterfall")
st.caption(
    "Which features pushed this specific prediction up or down from the model's expected baseline? "
    "Red bars = push prediction higher. Blue bars = push it lower."
)

test_rows = load_test_rows()

wf_c1, wf_c2 = st.columns([1, 3])
with wf_c1:
    wf_cats = sorted(test_rows["cat_id"].astype(str).unique())
    wf_cat = st.selectbox("Category", wf_cats, key="wf_cat")

    wf_items = sorted(
        test_rows[test_rows["cat_id"].astype(str) == wf_cat]["id"].unique()
    )
    wf_item = st.selectbox("Item", wf_items, key="wf_item")

    item_dates = sorted(pd.to_datetime(
        test_rows[test_rows["id"] == wf_item]["date"].unique()
    ))
    wf_date = st.selectbox(
        "Date", [d.date() for d in item_dates], key="wf_date"
    )

    wf_model = st.radio(
        "Model", available_models, key="wf_model", format_func=str.capitalize,
    )

with wf_c2:
    row_mask = (
        (test_rows["id"] == wf_item) &
        (test_rows["date"] == pd.Timestamp(str(wf_date)))
    )
    if not row_mask.any():
        st.warning("No data found for this selection.")
    else:
        X_wf = test_rows.loc[row_mask, FEATURE_COLS].iloc[[0]]
        exp = get_explainer(wf_model)
        sv_wf = exp(X_wf)
        plt.figure()
        shap.plots.waterfall(sv_wf[0], max_display=15, show=False)
        st.pyplot(plt.gcf(), bbox_inches="tight")
        plt.close("all")


# 6  Category SHAP box plot
st.divider()
st.subheader("4 — Feature Impact by Category")
st.caption(
    "Distribution of SHAP values for one feature, split by product category. "
    "Reveals whether a feature matters more for intermittent (HOBBIES, ~72% zeros) "
    "vs regular demand (FOODS, ~56% zeros)."
)

box_c1, box_c2 = st.columns([2, 1])
with box_c1:
    box_feature = st.selectbox(
        "Feature", _NUMERIC_FEATURES,
        index=_NUMERIC_FEATURES.index(_default),
        key="box_feature",
    )
with box_c2:
    box_model = st.radio(
        "Model", available_models,
        horizontal=True, format_func=str.capitalize, key="box_model",
    )

box_idx = FEATURE_COLS.index(box_feature)
box_df = pd.DataFrame({
    "shap":   shap_vals[box_model][:, box_idx],
    "cat_id": cat_ids,
})

fig_box = px.box(
    box_df, x="cat_id", y="shap",
    color="cat_id",
    color_discrete_map=CAT_COLORS,
    points="outliers",
    labels={"cat_id": "Category", "shap": f"SHAP value  ({box_feature})"},
    category_orders={"cat_id": ["FOODS", "HOBBIES", "HOUSEHOLD"]},
    height=400,
)
fig_box.add_hline(y=0, line_dash="dash", line_color="black", line_width=1,
                  annotation_text="No impact", annotation_position="top right")
fig_box.update_layout(showlegend=False, template="plotly_white")
st.plotly_chart(fig_box, use_container_width=True)
