import pandas as pd
import pytest
from retail_forecast.data import select_subset, melt_sales_long


def _make_wide_df():
    """Minimal wide-format sales DataFrame for testing."""
    return pd.DataFrame({
        "id":       ["FOODS_3_001_CA_1_evaluation", "FOODS_3_002_CA_1_evaluation",
                     "HOBBIES_1_001_CA_1_evaluation"],
        "item_id":  ["FOODS_3_001",  "FOODS_3_002",  "HOBBIES_1_001"],
        "dept_id":  ["FOODS_3",      "FOODS_3",      "HOBBIES_1"],
        "cat_id":   ["FOODS",        "FOODS",        "HOBBIES"],
        "store_id": ["CA_1",         "CA_1",         "CA_1"],
        "state_id": ["CA",           "CA",           "CA"],
        "d_1":      [5,              0,              2],
        "d_2":      [3,              1,              0],
    })


def test_select_subset_filters_correctly():
    df = _make_wide_df()
    result = select_subset(df, store_id="CA_1", cat_ids=["FOODS"])
    assert len(result) == 2
    assert set(result["cat_id"]) == {"FOODS"}


def test_select_subset_raises_on_empty():
    df = _make_wide_df()
    with pytest.raises(ValueError, match="No rows found"):
        select_subset(df, store_id="TX_1", cat_ids=["FOODS"])


def test_melt_sales_long_shape():
    df = _make_wide_df()
    subset = select_subset(df, store_id="CA_1", cat_ids=["FOODS"])
    long = melt_sales_long(subset)
    # 2 items x 2 days = 4 rows
    assert len(long) == 4
    assert "sales" in long.columns
    assert "d" in long.columns


def test_melt_sales_long_no_negative_sales():
    df = _make_wide_df()
    long = melt_sales_long(df)
    assert (long["sales"] >= 0).all()


def test_melt_sales_long_with_calendar():
    df = _make_wide_df()
    calendar = pd.DataFrame({
        "d":    ["d_1", "d_2"],
        "date": ["2011-01-29", "2011-01-30"],
        "wm_yr_wk": [11101, 11101],
    })
    long = melt_sales_long(df, calendar_df=calendar)
    assert "date" in long.columns
    assert pd.api.types.is_datetime64_any_dtype(long["date"])
