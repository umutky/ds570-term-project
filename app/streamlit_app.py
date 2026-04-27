import streamlit as st

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Retail Demand Forecasting")
st.markdown(
    """
    **DS570 Final Project** — End-to-end retail demand forecasting on the M5 dataset.

    **Scope:** CA_1 store × 3,049 products × 3 categories (HOBBIES, HOUSEHOLD, FOODS)

    **Model:** LightGBM with Tweedie objective (intermittent demand)

    Use the sidebar to navigate between pages.
    """
)
