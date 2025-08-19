# app.py
"""
Streamlit Open Order Report Dashboard
Requirements:
 - Save as app.py and run: streamlit run app.py
 - Libraries: streamlit, pandas, numpy, plotly
 - This single file implements:
    * File upload
    * Data parsing + graceful handling of missing columns
    * KPI calculations (operational, financial, risk) optimized for apparel and footwear
    * Audience-specific views & insights
    * Interactive filters & charts
    * Clean, presentation-ready layout for management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io  # For PDF export
import logging  # For error logging
import re  # For regex in payment days
import math  # For additional calculations
import statistics  # For stats like median
from collections import defaultdict  # For grouping
from typing import Optional, List, Dict, Any  # For type hints

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="Apparel & Footwear Open Order Dashboard", initial_sidebar_state="expanded")

# ---------------------------
# Utility helpers - Expanded with more functions
# ---------------------------
def safe_parse_dates(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a datetime series even if the column missing; else parse existing column."""
    if col not in df.columns:
        logger.warning(f"Column '{col}' not found, returning NaT series.")
        return pd.Series([pd.NaT] * len(df), index=df.index)
    parsed = pd.to_datetime(df[col], errors='coerce')
    if parsed.isna().all():
        logger.warning(f"All values in '{col}' are invalid dates.")
    return parsed

def safe_col(df: pd.DataFrame, col: str, default: Any = 0) -> pd.Series:
    """Return column if present, else a series with default values."""
    if col in df.columns:
        return df[col]
    logger.warning(f"Column '{col}' not found, using default {default}.")
    return pd.Series([default] * len(df), index=df.index)

def safe_sum(series: pd.Series) -> float:
    """Sum while ignoring non-numeric gracefully."""
    try:
        numeric = pd.to_numeric(series, errors='coerce')
        return numeric.sum(skipna=True)
    except Exception as e:
        logger.error(f"Error summing series: {e}")
        return 0.0

def safe_median(series: pd.Series) -> float:
    """Median while ignoring non-numeric gracefully."""
    try:
        numeric = pd.to_numeric(series, errors='coerce').dropna()
        if len(numeric) == 0:
            return 0.0
        return statistics.median(numeric)
    except Exception as e:
        logger.error(f"Error calculating median: {e}")
        return 0.0

def safe_std(series: pd.Series) -> float:
    """Standard deviation while ignoring non-numeric gracefully."""
    try:
        numeric = pd.to_numeric(series, errors='coerce')
        return numeric.std(skipna=True)
    except Exception as e:
        logger.error(f"Error calculating std: {e}")
        return 0.0

def human_currency(x: float) -> str:
    """Format currency simple."""
    try:
        if pd.isna(x):
            return "$0"
        x = float(x)
        if abs(x) >= 1_000_000_000:
            return f"${x/1_000_000_000:,.2f}B"
        if abs(x) >= 1_000_000:
            return f"${x/1_000_000:,.2f}M"
        if abs(x) >= 1_000:
            return f"${x/1_000:,.0f}K"
        return f"${x:,.2f}"
    except Exception as e:
        logger.error(f"Error formatting currency: {e}")
        return str(x)

def human_percentage(x: float) -> str:
    """Format percentage."""
    try:
        if pd.isna(x):
            return "0%"
        return f"{x:.1f}%"
    except Exception as e:
        logger.error(f"Error formatting percentage: {e}")
        return str(x)

def human_number(x: float) -> str:
    """Format large numbers."""
    try:
        if pd.isna(x):
            return "0"
        x = float(x)
        if abs(x) >= 1_000_000:
            return f"{x/1_000_000:,.2f}M"
        if abs(x) >= 1_000:
            return f"{x/1_000:,.0f}K"
        return f"{x:,.0f}"
    except Exception as e:
        logger.error(f"Error formatting number: {e}")
        return str(x)

def compute_payment_days_from_terms(terms_descr: str) -> Optional[int]:
    """Try to extract numeric days from TERMS_DESCR (e.g., '30 days', 'NET45'). Returns int or None"""
    if pd.isna(terms_descr):
        return None
    s = str(terms_descr).upper()
    m = re.search(r'NET\s*?(\d{1,3})', s)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d{1,3})\s*DAYS', s)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d{1,3})\b', s)
    if m:
        val = int(m.group(1))
        if 0 < val <= 180:
            return val
    logger.warning(f"Could not parse payment days from '{terms_descr}'")
    return None

def compute_risk_score(row: pd.Series) -> float:
    """Compute a simple risk score for each order."""
    score = 0.0
    if row["IS_PAST_EVENT"]:
        score += 50
    if row["ON_HOLD"]:
        score += 30
    if row["DAYS_OPEN"] > 90:
        score += 20
    return score

def categorize_risk(score: float) -> str:
    """Categorize risk based on score."""
    if score > 70:
        return "High"
    elif score > 30:
        return "Medium"
    else:
        return "Low"

# ---------------------------
# Caching for performance
# ---------------------------
@st.cache_data
def load_and_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Take raw DataFrame and produce cleaned df with required calculated fields."""
    df = df_raw.copy()

    # Normalize column names to expected (strip spaces, uppercase)
    df.columns = [c.strip().upper() if isinstance(c, str) else c for c in df.columns]

    # Ensure numeric columns exist
    num_cols_init = ["OPEN_QTY", "ORDER_QTY", "SHIP_QTY", "CANCEL_QTY", "PRICE", "OPEN_VALUE", "ORDER_VALUE", "DISCOUNT", "DISC_RATE", "EXCHANGE_RATE"]
    for col in num_cols_init:
        if col not in df.columns:
            df[col] = 0.0
            logger.info(f"Created missing column '{col}' with default 0.")

    # Convert numeric safely
    for c in num_cols_init:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    # Dates
    date_cols = ["START_DATE", "EVENT_DATE", "CANCEL_DATE", "BOOK_DATE"]
    for col in date_cols:
        df[col] = safe_parse_dates(df, col)

    # Derived calculations - Expanded
    df["COMPUTED_OPEN_VALUE"] = np.where(df["OPEN_VALUE"] > 0, df["OPEN_VALUE"], df["OPEN_QTY"] * df["PRICE"])
    df["COMPUTED_ORDER_VALUE"] = df["ORDER_QTY"] * df["PRICE"]
    df["COMPUTED_SHIP_VALUE"] = df["SHIP_QTY"] * df["PRICE"]
    df["COMPUTED_CANCEL_VALUE"] = df["CANCEL_QTY"] * df["PRICE"]

    today = pd.Timestamp.now().normalize()
    df["_DASHBOARD_DATE"] = today

    df["DAYS_OPEN"] = (today - df["START_DATE"]).dt.days.clip(lower=0)
    df["DAYS_TO_EVENT"] = (df["EVENT_DATE"] - today).dt.days.clip(lower=0)
    df["DAYS_PAST_EVENT"] = (today - df["EVENT_DATE"]).dt.days.clip(lower=0)
    df["IS_PAST_EVENT"] = (df["EVENT_DATE"].notna()) & (df["EVENT_DATE"] < today) & (df["OPEN_QTY"] > 0)
    df["IS_UPCOMING"] = (df["EVENT_DATE"] > today)
    df["IS_OPEN"] = (df["OPEN_QTY"] > 0) | (df["ORDER_QTY"] > (df["SHIP_QTY"] + df["CANCEL_QTY"]))

    # On hold - More robust check
    hold_cols = ["ORDER_HOLD", "CREDIT_HOLD", "HOLD_STATUS"]
    df["ON_HOLD"] = False
    for col in hold_cols:
        if col in df.columns:
            df["ON_HOLD"] = df["ON_HOLD"] | df[col].astype(str).str.upper().isin(["Y", "YES", "TRUE", "ON", "1", "HOLD", "H"])
            break

    df["CANCEL_LOSS"] = df["CANCEL_QTY"] * df["PRICE"]

    df["EXCHANGE_RATE"] = pd.to_numeric(df.get("EXCHANGE_RATE", 1), errors='coerce').fillna(1.0)
    if "CURRENCY" not in df.columns:
        df["CURRENCY"] = "USD"
    else:
        df["CURRENCY"] = df["CURRENCY"].fillna("USD").str.upper()

    df["OPEN_VALUE_BASE"] = df["COMPUTED_OPEN_VALUE"] * df["EXCHANGE_RATE"]
    df["ORDER_VALUE_BASE"] = df["COMPUTED_ORDER_VALUE"] * df["EXCHANGE_RATE"]

    df["TERMS_DESCR"] = safe_col(df, "TERMS_DESCR", None)
    df["PAYMENT_DAYS"] = df["TERMS_DESCR"].apply(compute_payment_days_from_terms)
    df["EST_PAYMENT_DATE"] = df["BOOK_DATE"] + pd.to_timedelta(df["PAYMENT_DAYS"].fillna(30), unit='D')  # Default 30 days

    df["LEAD_TIME"] = (df["EVENT_DATE"] - df["START_DATE"]).dt.days.clip(lower=0)

    if "COUNTRY" not in df.columns:
        df["COUNTRY"] = "Unknown"
    else:
        df["COUNTRY"] = df["COUNTRY"].fillna("Unknown")

    if "CUST_NAME" not in df.columns:
        if "CUSTOMER" in df.columns:
            df["CUST_NAME"] = df["CUSTOMER"]
        else:
            df["CUST_NAME"] = "Unknown"
    df["CUST_NAME"] = df["CUST_NAME"].fillna("Unknown")

    if "SKU" not in df.columns:
        df["SKU"] = df.get("STYLE", "").astype(str) + "-" + df.get("CLR", "").astype(str) + "-" + df.get("SIZE", "").astype(str)
    df["SKU"] = df["SKU"].str.strip("-").replace("", "UNKNOWN_SKU")

    if "STYLE" not in df.columns:
        df["STYLE"] = df["SKU"].str.split("-").str[0]

    if "ORDER_NO" not in df.columns:
        df["ORDER_NO"] = pd.Series([f"ORDER_{i+1}" for i in range(len(df))])  # Fallback unique IDs

    if "BOOK_SEASON" in df.columns:
        df["BOOK_SEASON"] = df["BOOK_SEASON"].fillna("Unknown")
    else:
        df["BOOK_SEASON"] = "Unknown"

    if "CATEGORY" not in df.columns:
        df["CATEGORY"] = "Unknown"

    # Add risk score
    df["RISK_SCORE"] = df.apply(compute_risk_score, axis=1)
    df["RISK_CATEGORY"] = df["RISK_SCORE"].apply(categorize_risk)

    return df

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all KPIs using sums and medians where appropriate."""
    kpis = {}

    # Core Totals - All sums
    kpis["total_orders"] = df["ORDER_NO"].nunique()
    kpis["total_open_units"] = safe_sum(df["OPEN_QTY"])
    kpis["total_open_value"] = safe_sum(df["COMPUTED_OPEN_VALUE"])
    kpis["total_order_value"] = safe_sum(df["COMPUTED_ORDER_VALUE"])
    kpis["total_shipped_units"] = safe_sum(df["SHIP_QTY"])
    kpis["total_shipped_value"] = safe_sum(df["COMPUTED_SHIP_VALUE"])
    kpis["total_cancelled_units"] = safe_sum(df["CANCEL_QTY"])
    kpis["total_cancelled_value"] = safe_sum(df["COMPUTED_CANCEL_VALUE"])
    kpis["total_discount"] = safe_sum(df["DISCOUNT"])
    kpis["total_on_hold_value"] = safe_sum(df.loc[df["ON_HOLD"], "COMPUTED_OPEN_VALUE"])
    kpis["total_on_hold_units"] = safe_sum(df.loc[df["ON_HOLD"], "OPEN_QTY"])
    kpis["total_past_due_value"] = safe_sum(df.loc[df["IS_PAST_EVENT"], "COMPUTED_OPEN_VALUE"])
    kpis["total_past_due_units"] = safe_sum(df.loc[df["IS_PAST_EVENT"], "OPEN_QTY"])
    kpis["total_fx_exposure"] = safe_sum(df["OPEN_VALUE_BASE"]) if safe_sum(df["EXCHANGE_RATE"]) > len(df) else 0

    # Rates - Using totals for calculations
    kpis["unit_fill_rate"] = (kpis["total_shipped_units"] / kpis["total_open_units"] * 100) if kpis["total_open_units"] > 0 else 0
    kpis["value_fill_rate"] = (kpis["total_shipped_value"] / kpis["total_open_value"] * 100) if kpis["total_open_value"] > 0 else 0
    kpis["backorder_rate"] = (kpis["total_open_units"] / (kpis["total_shipped_units"] + kpis["total_open_units"] + kpis["total_cancelled_units"]) * 100) if (kpis["total_shipped_units"] + kpis["total_open_units"] + kpis["total_cancelled_units"]) > 0 else 0
    kpis["cancellation_rate"] = (kpis["total_cancelled_units"] / (kpis["total_shipped_units"] + kpis["total_open_units"] + kpis["total_cancelled_units"]) * 100) if (kpis["total_shipped_units"] + kpis["total_open_units"] + kpis["total_cancelled_units"]) > 0 else 0
    kpis["discount_rate"] = (kpis["total_discount"] / kpis["total_order_value"] * 100) if kpis["total_order_value"] > 0 else 0
    kpis["inventory_turnover"] = kpis["total_shipped_units"] / kpis["total_open_units"] if kpis["total_open_units"] > 0 else 0
    kpis["otd_rate"] = ((kpis["total_orders"] - df[df["IS_PAST_EVENT"]]["ORDER_NO"].nunique()) / kpis["total_orders"] * 100) if kpis["total_orders"] > 0 else 0

    # Medians for days - Using median instead of mean
    kpis["median_lead_time"] = safe_median(df["LEAD_TIME"])
    kpis["median_payment_days"] = safe_median(df["PAYMENT_DAYS"])
    kpis["median_days_open"] = safe_median(df["DAYS_OPEN"])
    kpis["std_lead_time"] = safe_std(df["LEAD_TIME"])

    # Additional industry KPIs
    kpis["total_backorder_value"] = safe_sum(df[df["IS_PAST_EVENT"]]["COMPUTED_OPEN_VALUE"])
    kpis["total_cancellation_loss"] = kpis["total_cancelled_value"]
    kpis["total_high_risk_value"] = safe_sum(df[df["RISK_CATEGORY"] == "High"]["COMPUTED_OPEN_VALUE"])
    kpis["high_risk_orders"] = df[df["RISK_CATEGORY"] == "High"]["ORDER_NO"].nunique()
    kpis["total_open_per_sku"] = kpis["total_open_units"] / df["SKU"].nunique() if df["SKU"].nunique() > 0 else 0
    kpis["max_open_sku_units"] = df.groupby("SKU")["OPEN_QTY"].sum().max() if not df.empty else 0
    kpis["total_production_backlog_units"] = kpis["total_past_due_units"]
    kpis["cash_conversion_cycle"] = kpis["median_payment_days"] - kpis["median_lead_time"]  # Using medians

    return kpis

@st.cache_data
def compute_groupings(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute groupings for charts."""
    groupings = {}

    groupings["top_customers_value"] = df.groupby("CUST_NAME")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False).head(10)
    groupings["top_skus_value"] = df.groupby("SKU")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False).head(15)
    groupings["top_skus_units"] = df.groupby("SKU")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False).head(15)
    groupings["customer_open_value"] = df.groupby("CUST_NAME")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)
    groupings["customer_open_units"] = df.groupby("CUST_NAME")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False)
    groupings["style_open_value"] = df.groupby("STYLE")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)
    groupings["style_open_units"] = df.groupby("STYLE")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False)
    groupings["country_open_value"] = df.groupby("COUNTRY")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False) if "COUNTRY" in df.columns and df["COUNTRY"].notna().any() else pd.DataFrame()
    groupings["country_open_units"] = df.groupby("COUNTRY")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False) if "COUNTRY" in df.columns and df["COUNTRY"].notna().any() else pd.DataFrame()
    if "BOOK_SEASON" in df.columns:
        groupings["season_open_value"] = df.groupby("BOOK_SEASON")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)
        groupings["season_open_units"] = df.groupby("BOOK_SEASON")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False)
    if "CATEGORY" in df.columns:
        groupings["category_open_value"] = df.groupby("CATEGORY")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)
        groupings["category_open_units"] = df.groupby("CATEGORY")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False)

    aging_bins = [0, 30, 60, 90, 120, 180, np.inf]
    aging_labels = ["<30 days", "30-60", "60-90", "90-120", "120-180", ">180"]
    df["AGING_BUCKET"] = pd.cut(df["DAYS_OPEN"], bins=aging_bins, labels=aging_labels)
    groupings["aging_open_value"] = df.groupby("AGING_BUCKET", observed=True)["COMPUTED_OPEN_VALUE"].sum().reset_index()
    groupings["aging_open_units"] = df.groupby("AGING_BUCKET", observed=True)["OPEN_QTY"].sum().reset_index()

    df["_MONTH_START"] = df["START_DATE"].dt.to_period("M").dt.to_timestamp()
    groupings["monthly_open_value"] = df.groupby("_MONTH_START")["COMPUTED_OPEN_VALUE"].sum().reset_index().dropna().sort_values("_MONTH_START")
    groupings["monthly_open_units"] = df.groupby("_MONTH_START")["OPEN_QTY"].sum().reset_index().dropna().sort_values("_MONTH_START")
    groupings["monthly_shipped_value"] = df.groupby("_MONTH_START")["COMPUTED_SHIP_VALUE"].sum().reset_index().dropna().sort_values("_MONTH_START")

    groupings["payment_projection_value"] = df.groupby(pd.Grouper(key="EST_PAYMENT_DATE", freq="M"))["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("EST_PAYMENT_DATE")

    groupings["fx_exposure_by_currency"] = df.groupby("CURRENCY")["OPEN_VALUE_BASE"].sum().reset_index(name="CONVERTED_TO_BASE")

    # Additional groupings
    groupings["risk_by_category_value"] = df.groupby("RISK_CATEGORY")["COMPUTED_OPEN_VALUE"].sum().reset_index()
    groupings["risk_by_category_units"] = df.groupby("RISK_CATEGORY")["OPEN_QTY"].sum().reset_index()
    groupings["open_by_style_units"] = df.groupby("STYLE")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False).head(10)
    groupings["open_by_style_value"] = df.groupby("STYLE")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False).head(10)
    groupings["cancel_by_reason_value"] = df.groupby("CANCEL_CODE")["COMPUTED_CANCEL_VALUE"].sum().reset_index().sort_values("COMPUTED_CANCEL_VALUE", ascending=False) if "CANCEL_CODE" in df.columns else pd.DataFrame()
    if "SHIP_VIA" in df.columns:
        groupings["ship_via_value"] = df.groupby("SHIP_VIA")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)

    # For heatmaps
    if "CUST_NAME" in df.columns and "STYLE" in df.columns:
        groupings["customer_style_heatmap"] = df.pivot_table(index="CUST_NAME", columns="STYLE", values="COMPUTED_OPEN_VALUE", aggfunc="sum", fill_value=0)
    if "COUNTRY" in df.columns and "STYLE" in df.columns:
        groupings["country_style_heatmap"] = df.pivot_table(index="COUNTRY", columns="STYLE", values="COMPUTED_OPEN_VALUE", aggfunc="sum", fill_value=0)

    return groupings

# ---------------------------
# UI Sidebar: Upload + Filters - Enhanced
# ---------------------------
st.title("Full-Fledged Apparel & Footwear Open Order Dashboard")
st.markdown("A comprehensive dashboard calculating industry metrics and KPIs using sums, with pie charts, line graphs, bar charts, heatmaps, and detailed analysis. Focused on Customers and Styles.")

with st.sidebar:
    st.header("Data Upload & Filters")
    uploaded_file = st.file_uploader("Upload Open Order Report (CSV or Excel)", type=["csv", "xlsx"], accept_multiple_files=False)

    # Date filters - Expanded
    date_filter_enabled = st.checkbox("Filter by Date Range", value=False)
    if date_filter_enabled:
        col1, col2 = st.columns(2)
        with col1:
            start_min = st.date_input("Start From", value=(datetime.now() - timedelta(days=365)))
        with col2:
            start_max = st.date_input("Start To", value=datetime.now())
        event_min = st.date_input("Event From", value=None)
        event_max = st.date_input("Event To", value=None)
    else:
        start_min, start_max, event_min, event_max = None, None, None, None

    # Advanced filters
    min_open_value = st.number_input("Min Open Value", value=0.0)
    max_risk_score = st.slider("Max Risk Score", 0, 100, 100)

    st.markdown("---")
    st.caption("Dynamic Filters (populate after upload)")

# Handle file upload
if uploaded_file is None:
    st.info("Upload your report to begin. Expected columns include: ORDER_NO, START_DATE, EVENT_DATE, BOOK_DATE, OPEN_QTY, ORDER_QTY, SHIP_QTY, CANCEL_QTY, PRICE, OPEN_VALUE, ORDER_VALUE, DISCOUNT, CUST_NAME, SKU, STYLE, CLR, BOOK_SEASON, COUNTRY, ORDER_HOLD, TERMS_DESCR, EXCHANGE_RATE, CURRENCY, CANCEL_CODE, SHIP_VIA, CATEGORY.")
    st.stop()

# Load data
try:
    if uploaded_file.name.endswith('.xlsx'):
        df_raw = pd.read_excel(uploaded_file, dtype=str)
    else:
        df_raw = pd.read_csv(uploaded_file, dtype=str, encoding='utf-8')
except UnicodeDecodeError:
    uploaded_file.seek(0)
    df_raw = pd.read_csv(uploaded_file, dtype=str, encoding='latin1')
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# Preprocess
df = load_and_preprocess(df_raw)

# Dynamic filters in sidebar
with st.sidebar:
    customers = sorted(df["CUST_NAME"].unique())
    customer_filter = st.multiselect("Customers", options=customers, default=[])

    styles = sorted(df["STYLE"].unique())
    style_filter = st.multiselect("Styles", options=styles, default=[])

    if "COUNTRY" in df.columns:
        countries = sorted(df["COUNTRY"].unique())
        country_filter = st.multiselect("Countries", options=countries, default=[])
    else:
        country_filter = []

    skus = sorted(df["SKU"].unique())[:5000]
    sku_filter = st.multiselect("SKUs / Style-Color-Size", options=skus, default=[])

    if "ORDER_APPR_STATUS" in df.columns:
        statuses = sorted(df["ORDER_APPR_STATUS"].fillna("Unknown").unique())
        order_status_filter = st.multiselect("Order Statuses", options=statuses, default=[])
    else:
        order_status_filter = []

    if "BOOK_SEASON" in df.columns:
        seasons = sorted(df["BOOK_SEASON"].unique())
        season_filter = st.multiselect("Book Seasons", options=seasons, default=[])
    else:
        season_filter = []

    if "CATEGORY" in df.columns:
        categories = sorted(df["CATEGORY"].unique())
        category_filter = st.multiselect("Categories", options=categories, default=[])
    else:
        category_filter = []

    risk_categories = sorted(df["RISK_CATEGORY"].unique())
    risk_filter = st.multiselect("Risk Categories", options=risk_categories, default=[])

    st.markdown("---")
    if st.button("Apply Filters & Refresh"):
        st.rerun()

# Apply filters
df_filtered = df.copy()
if date_filter_enabled:
    if start_min and start_max:
        start_min_ts = pd.to_datetime(start_min)
        start_max_ts = pd.to_datetime(start_max) + timedelta(days=1) - timedelta(seconds=1)
        df_filtered = df_filtered[(df_filtered["START_DATE"] >= start_min_ts) & (df_filtered["START_DATE"] <= start_max_ts)]
    if event_min and event_max:
        event_min_ts = pd.to_datetime(event_min)
        event_max_ts = pd.to_datetime(event_max) + timedelta(days=1) - timedelta(seconds=1)
        df_filtered = df_filtered[(df_filtered["EVENT_DATE"] >= event_min_ts) & (df_filtered["EVENT_DATE"] <= event_max_ts)]

if customer_filter:
    df_filtered = df_filtered[df_filtered["CUST_NAME"].isin(customer_filter)]
if style_filter:
    df_filtered = df_filtered[df_filtered["STYLE"].isin(style_filter)]
if country_filter:
    df_filtered = df_filtered[df_filtered["COUNTRY"].isin(country_filter)]
if sku_filter:
    df_filtered = df_filtered[df_filtered["SKU"].isin(sku_filter)]
if order_status_filter:
    df_filtered = df_filtered[df_filtered["ORDER_APPR_STATUS"].isin(order_status_filter)]
if season_filter:
    df_filtered = df_filtered[df_filtered["BOOK_SEASON"].isin(season_filter)]
if category_filter:
    df_filtered = df_filtered[df_filtered["CATEGORY"].isin(category_filter)]
if risk_filter:
    df_filtered = df_filtered[df_filtered["RISK_CATEGORY"].isin(risk_filter)]

df_filtered = df_filtered[df_filtered["COMPUTED_OPEN_VALUE"] >= min_open_value]
df_filtered = df_filtered[df_filtered["RISK_SCORE"] <= max_risk_score]

if len(df_filtered) == 0:
    st.warning("No data after filtering. Adjust filters.")
    st.stop()

# Compute KPIs and groupings
kpis = compute_kpis(df_filtered)
groupings = compute_groupings(df_filtered)

# ---------------------------
# Dashboard Layout - Enhanced with more tabs and graphs
# ---------------------------
tabs = st.tabs(["Overview", "Operations", "Finance", "Risk & Trends", "Production", "Graphs & Visuals", "Detailed Analysis & Data"])

# Common function to display metrics
def display_metrics(cols: List, metrics: List[Dict]):
    for i, metric in enumerate(metrics):
        with cols[i % len(cols)]:
            st.metric(**metric)

with tabs[0]:  # Overview
    st.subheader("Executive Overview - Key Totals and Rates")
    cols = st.columns(5)
    metrics = [
        {"label": "Total Open Value", "value": human_currency(kpis["total_open_value"]), "help": "Sum of all open order values."},
        {"label": "Total Open Units", "value": human_number(kpis["total_open_units"]), "help": "Sum of all open quantities."},
        {"label": "Total Orders", "value": human_number(kpis["total_orders"]), "help": "Count of unique orders."},
        {"label": "On-Time Delivery Rate", "value": human_percentage(kpis["otd_rate"]), "help": "Percentage of on-time orders."},
        {"label": "Median Lead Time", "value": f"{kpis['median_lead_time']:.1f} days", "help": "Median lead time in days."},
        {"label": "Total Shipped Value", "value": human_currency(kpis["total_shipped_value"]), "help": "Sum of shipped values."},
        {"label": "Total Shipped Units", "value": human_number(kpis["total_shipped_units"]), "help": "Sum of shipped quantities."}
    ]
    display_metrics(cols, metrics)

    st.subheader("Overview Visuals")
    col1, col2 = st.columns(2)
    with col1:
        fig_cust_value = px.pie(groupings["customer_open_value"], values="COMPUTED_OPEN_VALUE", names="CUST_NAME", title="Open Value by Customer (Pie)")
        st.plotly_chart(fig_cust_value, use_container_width=True)
    with col2:
        fig_monthly_value = px.line(groupings["monthly_open_value"], x="_MONTH_START", y="COMPUTED_OPEN_VALUE", title="Monthly Open Value Trend (Line)")
        st.plotly_chart(fig_monthly_value, use_container_width=True)

with tabs[1]:  # Operations
    st.subheader("Operations - Fulfillment Metrics")
    cols = st.columns(5)
    metrics = [
        {"label": "Unit Fill Rate", "value": human_percentage(kpis["unit_fill_rate"]), "help": "Shipped units / Total units."},
        {"label": "Value Fill Rate", "value": human_percentage(kpis["value_fill_rate"]), "help": "Shipped value / Total value."},
        {"label": "Backorder Rate", "value": human_percentage(kpis["backorder_rate"]), "help": "Open units / Total units."},
        {"label": "Inventory Turnover", "value": f"{kpis['inventory_turnover']:.2f}x", "help": "Shipped / Open units."},
        {"label": "Median Days Open", "value": f"{kpis['median_days_open']:.1f} days", "help": "Median days orders open."},
        {"label": "Total Past Due Units", "value": human_number(kpis["total_past_due_units"]), "help": "Sum of past due quantities."}
    ]
    display_metrics(cols, metrics)

    st.subheader("Operations Visuals")
    col1, col2 = st.columns(2)
    with col1:
        fig_topsku_value = px.bar(groupings["top_skus_value"], x="SKU", y="COMPUTED_OPEN_VALUE", title="Top SKUs by Open Value (Bar)")
        st.plotly_chart(fig_topsku_value, use_container_width=True)
    with col2:
        fig_aging_value = px.bar(groupings["aging_open_value"], x="AGING_BUCKET", y="COMPUTED_OPEN_VALUE", title="Open Value by Aging (Bar)")
        st.plotly_chart(fig_aging_value, use_container_width=True)

    fig_lead_dist = px.histogram(df_filtered, x="LEAD_TIME", nbins=20, title="Lead Time Distribution (Histogram)")
    st.plotly_chart(fig_lead_dist, use_container_width=True)

with tabs[2]:  # Finance
    st.subheader("Finance - Revenue Metrics")
    cols = st.columns(5)
    metrics = [
        {"label": "Total Open Value", "value": human_currency(kpis["total_open_value"]), "help": "Sum of open values."},
        {"label": "Total Discount", "value": human_currency(kpis["total_discount"]), "help": "Sum of discounts."},
        {"label": "Discount Rate", "value": human_percentage(kpis["discount_rate"]), "help": "Total discount / Total order value."},
        {"label": "Median Payment Days", "value": f"{kpis['median_payment_days']:.1f}", "help": "Median payment days."},
        {"label": "Cash Conversion Cycle", "value": f"{kpis['cash_conversion_cycle']:.1f} days", "help": "Medians of payments - lead time."},
        {"label": "Total FX Exposure", "value": human_currency(kpis["total_fx_exposure"]), "help": "Sum of base open values."}
    ]
    display_metrics(cols, metrics)

    st.subheader("Finance Visuals")
    col1, col2 = st.columns(2)
    with col1:
        fig_pay = px.bar(groupings["payment_projection_value"], x="EST_PAYMENT_DATE", y="COMPUTED_OPEN_VALUE", title="Projected Collections (Bar)")
        st.plotly_chart(fig_pay, use_container_width=True)
    with col2:
        if groupings["fx_exposure_by_currency"].shape[0] > 1:
            fig_fx = px.pie(groupings["fx_exposure_by_currency"], values="CONVERTED_TO_BASE", names="CURRENCY", title="FX Exposure (Pie)")
            st.plotly_chart(fig_fx, use_container_width=True)

    fig_discount_dist = px.box(df_filtered, y="DISC_RATE", x="CUST_NAME", title="Discount Rate by Customer (Box)")
    st.plotly_chart(fig_discount_dist, use_container_width=True)

with tabs[3]:  # Risk & Trends
    st.subheader("Risk & Trends Metrics")
    cols = st.columns(4)
    metrics = [
        {"label": "Cancellation Rate", "value": human_percentage(kpis["cancellation_rate"]), "help": "Cancelled / Total units."},
        {"label": "Total Cancellation Loss", "value": human_currency(kpis["total_cancellation_loss"]), "help": "Sum of cancelled values."},
        {"label": "Total On Hold Value", "value": human_currency(kpis["total_on_hold_value"]), "help": "Sum of held values."},
        {"label": "High Risk Orders", "value": human_number(kpis["high_risk_orders"]), "help": "Count of high risk orders."},
        {"label": "Total High Risk Value", "value": human_currency(kpis["total_high_risk_value"]), "help": "Sum of high risk values."}
    ]
    display_metrics(cols, metrics)

    st.subheader("Risk Visuals")
    col1, col2 = st.columns(2)
    with col1:
        fig_risk_cat = px.pie(groupings["risk_by_category_value"], values="COMPUTED_OPEN_VALUE", names="RISK_CATEGORY", title="Open Value by Risk (Pie)")
        st.plotly_chart(fig_risk_cat, use_container_width=True)
    with col2:
        if not groupings["cancel_by_reason_value"].empty:
            fig_cancel = px.bar(groupings["cancel_by_reason_value"], x="CANCEL_CODE", y="COMPUTED_CANCEL_VALUE", title="Cancellation by Reason (Bar)")
            st.plotly_chart(fig_cancel, use_container_width=True)

    fig_trend_value = px.line(groupings["monthly_open_value"], x="_MONTH_START", y="COMPUTED_OPEN_VALUE", title="Open Value Trend (Line)")
    st.plotly_chart(fig_trend_value, use_container_width=True)

with tabs[4]:  # Production
    st.subheader("Production Metrics")
    cols = st.columns(4)
    metrics = [
        {"label": "Total Open per SKU", "value": human_number(kpis["total_open_per_sku"]), "help": "Total open units / Unique SKUs."},
        {"label": "Max Open SKU Units", "value": human_number(kpis["max_open_sku_units"]), "help": "Max sum open units for a SKU."},
        {"label": "Total Production Backlog Units", "value": human_number(kpis["total_production_backlog_units"]), "help": "Sum past due units."},
        {"label": "Median Lead Time", "value": f"{kpis['median_lead_time']:.1f} days", "help": "Median lead time."},
        {"label": "Lead Time Variability", "value": f"{kpis['std_lead_time']:.1f} days", "help": "Std dev of lead times."}
    ]
    display_metrics(cols, metrics)

    st.subheader("Production Visuals")
    col1, col2 = st.columns(2)
    with col1:
        fig_style_units = px.bar(groupings["open_by_style_units"], x="STYLE", y="OPEN_QTY", title="Top Styles by Open Units (Bar)")
        st.plotly_chart(fig_style_units, use_container_width=True)
    with col2:
        if "season_open_value" in groupings:
            fig_season = px.bar(groupings["season_open_value"], x="BOOK_SEASON", y="COMPUTED_OPEN_VALUE", title="Open Value by Season (Bar)")
            st.plotly_chart(fig_season, use_container_width=True)

    if "category_open_value" in groupings:
        fig_cat = px.treemap(groupings["category_open_value"], path=["CATEGORY"], values="COMPUTED_OPEN_VALUE", title="Open Value by Category (Treemap)")
        st.plotly_chart(fig_cat, use_container_width=True)

with tabs[5]:  # Graphs & Visuals
    st.subheader("Additional Graphs & Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(df_filtered, x="DAYS_OPEN", y="COMPUTED_OPEN_VALUE", color="RISK_CATEGORY", title="Open Value vs Days Open (Scatter)")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        if "customer_style_heatmap" in groupings:
            fig_heatmap = px.imshow(groupings["customer_style_heatmap"], title="Open Value Heatmap: Customer vs Style")
            st.plotly_chart(fig_heatmap, use_container_width=True)

    fig_monthly_units = px.line(groupings["monthly_open_units"], x="_MONTH_START", y="OPEN_QTY", title="Monthly Open Units Trend (Line)")
    st.plotly_chart(fig_monthly_units, use_container_width=True)

    fig_ship_via = px.bar(groupings.get("ship_via_value", pd.DataFrame()), x="SHIP_VIA", y="COMPUTED_OPEN_VALUE", title="Open Value by Ship Via (Bar)")
    if not fig_ship_via.data:
        st.info("No ship via data.")
    else:
        st.plotly_chart(fig_ship_via, use_container_width=True)

    if "country_style_heatmap" in groupings:
        fig_country_heatmap = px.imshow(groupings["country_style_heatmap"], title="Open Value Heatmap: Country vs Style")
        st.plotly_chart(fig_country_heatmap, use_container_width=True)

with tabs[6]:  # Detailed Analysis & Data
    st.subheader("Detailed Analysis")
    st.markdown("""
    **Overall Analysis:** 
    The total open order value sums to {} with {} unique orders. This indicates a substantial backlog, potentially driven by high-demand products in key customers. The on-time delivery rate of {} suggests room for improvement in supply chain efficiency to avoid customer dissatisfaction.
    """.format(human_currency(kpis["total_open_value"]), human_number(kpis["total_orders"]), human_percentage(kpis["otd_rate"])))

    st.markdown("**Operations Analysis:**")
    st.write("- Unit fill rate at {}: If below 95%, it points to potential stockouts; recommend reviewing inventory levels for top SKUs to ensure fulfillment targets are met.".format(human_percentage(kpis["unit_fill_rate"])))
    st.write("- Backorder rate of {}: High rates may lead to lost sales; prioritize production for past-due orders totaling {} units.".format(human_percentage(kpis["backorder_rate"]), human_number(kpis["total_past_due_units"])))
    st.write("- Inventory turnover {}x: Lower turnover indicates excess stock; aim to increase shipments to balance inventory.".format(kpis["inventory_turnover"]))

    st.markdown("**Finance Analysis:**")
    st.write("- Discount rate {}: Total discounts sum to {}; analyze if discounts are yielding higher volume or eroding margins.".format(human_percentage(kpis["discount_rate"]), human_currency(kpis["total_discount"])))
    st.write("- Cash conversion cycle {} days: Positive cycle supports liquidity, but monitor payment collections projected at future months.".format(kpis["cash_conversion_cycle"]))
    st.write("- FX exposure totals {}: Diversify currencies if exposure is concentrated to mitigate risks.".format(human_currency(kpis["total_fx_exposure"])))

    st.markdown("**Risk Analysis:**")
    st.write("- Cancellation rate {} with loss {}: Investigate common reasons; high cancellations may signal product or forecasting issues.".format(human_percentage(kpis["cancellation_rate"]), human_currency(kpis["total_cancellation_loss"])))
    st.write("- High-risk orders: {} orders valued at {}; resolve holds and past-dues promptly to unlock revenue.".format(human_number(kpis["high_risk_orders"]), human_currency(kpis["total_high_risk_value"])))

    st.markdown("**Production Analysis:**")
    st.write("- Production backlog {} units: Increase capacity for affected SKUs to reduce lead times, currently median {} days.".format(human_number(kpis["total_production_backlog_units"]), kpis["median_lead_time"]))
    st.write("- Lead time variability {} days: High std dev suggests inconsistent processes; standardize to improve predictability.".format(kpis["std_lead_time"]))

    st.subheader("Raw Filtered Data Preview")
    st.dataframe(df_filtered.head(1000))

    # Exports
    st.subheader("Export Options")
    col1, col2 = st.columns(2)
    with col1:
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "filtered_open_orders.csv", "text/csv")
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df_filtered.to_excel(writer, index=False)
        excel_buffer.seek(0)
        st.download_button("Download Excel", excel_buffer, "filtered_open_orders.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------
# Final notes
# ---------------------------
st.markdown("---")
st.info("Dashboard focused on Customers and Styles, with optional Country support, using sums for quantities and values, medians for days, with enhanced graphs and detailed analysis across departments.")