import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from retail_forecast.config import MODELS_DIR, REPORTS_DIR
from retail_forecast.models.lgbm import LGBMForecast

st.set_page_config(page_title="Model Insights", layout="wide")
st.title("Model Insights")
st.caption(
    "Feature importance, Tweedie vs Gaussian evaluation, "
    "and actual vs predicted analysis on the held-out test set."
)


# ---- Load helpers -----------------------------------------------------------

@st.cache_resource
def load_model(name: str) -> LGBMForecast | None:
    path = MODELS_DIR / f"lgbm_{name}.pkl"
    if not path.exists():
        return None
    return LGBMForecast.load(path)


@st.cache_data
def load_test_predictions(name: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / f"test_predictions_{name}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_latest_tracker(model_label: str) -> dict | None:
    runs_dir = REPORTS_DIR / "model_runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(runs_dir.glob(f"*_{model_label}.json"), reverse=True)
    if not candidates:
        return None
    with open(candidates[0]) as f:
        return json.load(f)


# ---- Check availability ------------------------------------------------------
tweedie_model = load_model("tweedie")
gaussian_model = load_model("gaussian")

if tweedie_model is None:
    st.error("Model not found. Run `rf-train` first.")
    st.stop()


# ---- Feature importance -----------------------------------------------------
st.subheader("Feature importance (Tweedie model — gain)")
st.caption(
    "Gain measures the total improvement in the loss function brought "
    "by features at split points across all trees."
)

fi = tweedie_model.feature_importance
fi_df = fi.reset_index().rename(columns={"index": "feature", 0: "gain"})
fi_df.columns = ["feature", "gain"]
fi_df = fi_df.head(15)
fi_df["gain_pct"] = fi_df["gain"] / fi_df["gain"].sum() * 100

fig_fi = px.bar(
    fi_df,
    x="gain_pct",
    y="feature",
    orientation="h",
    labels={"feature": "Feature", "gain_pct": "Importance (% of total gain)"},
    text_auto=".1f",
)
fig_fi.update_layout(
    yaxis=dict(autorange="reversed"),
    height=420,
    margin=dict(t=20, b=40),
)
st.plotly_chart(fig_fi, use_container_width=True)


# ---- Tweedie vs Gaussian metrics --------------------------------------------
st.divider()
st.subheader("Tweedie vs Gaussian — test set metrics")
st.caption(
    "Both models use identical features and train/val/test splits. "
    "Differences in RMSE, MAE, and WMAPE reflect the benefit of the Tweedie "
    "objective for zero-inflated demand data."
)

tweedie_data = load_test_predictions("tweedie")
gaussian_data = load_test_predictions("gaussian")

if tweedie_data is None or gaussian_data is None:
    st.warning("Test predictions not found. Re-run `rf-train` to generate them.")
else:
    def compute_metrics(df: pd.DataFrame, pred_col: str = "y_pred") -> dict:
        y = df["sales"].values
        p = np.maximum(df[pred_col].values, 0)
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        mae = float(np.mean(np.abs(y - p)))
        denom = float(np.sum(np.abs(y)))
        wmape = float(np.sum(np.abs(y - p)) / denom) if denom > 0 else 0.0
        return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "WMAPE": round(wmape, 4)}

    rows = []
    for model_name, pred_df in [("Tweedie", tweedie_data), ("Gaussian", gaussian_data)]:
        overall = compute_metrics(pred_df)
        overall["Model"] = model_name
        overall["Category"] = "Overall"
        rows.append(overall)
        for cat, grp in pred_df.groupby("cat_id"):
            m = compute_metrics(grp)
            m["Model"] = model_name
            m["Category"] = str(cat)
            rows.append(m)

    metrics_df = pd.DataFrame(rows)[["Model", "Category", "RMSE", "MAE", "WMAPE"]]
    overall_df = metrics_df[metrics_df["Category"] == "Overall"].set_index("Model")
    by_cat_df = metrics_df[metrics_df["Category"] != "Overall"]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Overall test set**")
        st.dataframe(overall_df[["RMSE", "MAE", "WMAPE"]], use_container_width=True)
    with col2:
        st.markdown("**By category**")
        st.dataframe(
            by_cat_df.pivot(index="Category", columns="Model", values="WMAPE").round(4),
            use_container_width=True,
        )
        st.caption("Values shown: WMAPE (lower is better)")

    # WMAPE comparison bar chart
    fig_metrics = px.bar(
        by_cat_df,
        x="Category",
        y="WMAPE",
        color="Model",
        barmode="group",
        labels={"WMAPE": "WMAPE (lower is better)", "Category": "Category"},
        color_discrete_map={"Tweedie": "#1f77b4", "Gaussian": "#ff7f0e"},
        text_auto=".3f",
    )
    fig_metrics.update_layout(height=320, margin=dict(t=20, b=40))
    st.plotly_chart(fig_metrics, use_container_width=True)


# ---- Actual vs Predicted ----------------------------------------------------
st.divider()
st.subheader("Actual vs predicted — Tweedie model on test set")

if tweedie_data is not None:
    col1, col2 = st.columns([1, 3])
    with col1:
        cats = sorted(tweedie_data["cat_id"].unique().tolist())
        selected_cat = st.selectbox("Category", cats)

        depts = sorted(
            tweedie_data[tweedie_data["cat_id"] == selected_cat]["dept_id"].unique().tolist()
        ) if "dept_id" in tweedie_data.columns else []
        selected_dept = st.selectbox("Department", depts)

        items = sorted(
            tweedie_data[tweedie_data["dept_id"] == selected_dept]["id"].unique().tolist()
        ) if depts else []
        selected_item = st.selectbox("Item", items)

    with col2:
        item_test = (
            tweedie_data[tweedie_data["id"] == selected_item]
            .sort_values("date")
        )

        fig_avp = go.Figure()
        fig_avp.add_trace(
            go.Scatter(
                x=item_test["date"],
                y=item_test["sales"],
                mode="lines",
                name="Actual sales",
                line=dict(color="#1f77b4", width=1.5),
            )
        )
        fig_avp.add_trace(
            go.Scatter(
                x=item_test["date"],
                y=item_test["y_pred"],
                mode="lines",
                name="Predicted (Tweedie)",
                line=dict(color="#d62728", width=1.5, dash="dot"),
            )
        )
        fig_avp.update_layout(
            xaxis_title="Date",
            yaxis_title="Units sold",
            legend=dict(orientation="h", y=1.05),
            height=340,
            margin=dict(t=30, b=40),
        )
        st.plotly_chart(fig_avp, use_container_width=True)

    # Aggregated daily actual vs predicted over entire test set
    st.markdown("**Daily total demand — actual vs predicted (all items)**")
    agg = (
        tweedie_data.groupby("date")[["sales", "y_pred"]]
        .sum()
        .reset_index()
    )
    fig_agg = go.Figure()
    fig_agg.add_trace(
        go.Scatter(
            x=agg["date"], y=agg["sales"],
            mode="lines", name="Actual",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig_agg.add_trace(
        go.Scatter(
            x=agg["date"], y=agg["y_pred"],
            mode="lines", name="Predicted (Tweedie)",
            line=dict(color="#d62728", width=1.5, dash="dot"),
        )
    )
    fig_agg.update_layout(
        xaxis_title="Date",
        yaxis_title="Total units sold (all items)",
        legend=dict(orientation="h", y=1.05),
        height=300,
        margin=dict(t=30, b=40),
    )
    st.plotly_chart(fig_agg, use_container_width=True)
