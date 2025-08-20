# app.py
"""
Streamlit Open Order Report Dashboard (v8 - Brand/Div Enhanced)
- Implements user's requested changes:
  * Keep Overview.
  * Operations: show Top SKUs with Open Value broken down by Customer + other useful ops visuals.
  * Finance: remove median payment days, cash conversion cycle, total FX exposure; remove projected collections, FX pie, discount rate by customer.
           Add meaningful finance visuals (Top Customers by Order Value, Revenue by Brand/Div, Discount vs Order Value).
  * Risk & Trends: replace with useful visuals (Risk by Brand/Div, Past-Due by Customer, On-Hold by Value). If not relevant, it hides.
  * Production: remove BOOK_SEASON visuals (if empty). Show Top SKUs with Customer amounts, Category breakdown.
  * Graphs & Visuals: keep "Open Value by Ship Via". Replace scatter with more insightful plots if needed.
  * Detailed Analysis: keep and expand with brand/div insights.
- Includes DIV→Brand mapping, new DIV_BRAND column, and filter across tabs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import logging
import re
import statistics

# ---------------------------
# Setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide", page_title="Open Order Dashboard (Brand/Div Enhanced)", initial_sidebar_state="expanded")
st.title("Open Order Dashboard")
st.caption("Optimized for Operations, Finance, Production; with meaningful, data-backed visuals and Div/Brand mapping.")

# ---------------------------
# Helpers
# ---------------------------
def safe_parse_dates(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([pd.NaT] * len(df), index=df.index)
    parsed = pd.to_datetime(df[col], errors="coerce")
    return parsed

def safe_col(df: pd.DataFrame, col: str, default):
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)

def safe_sum(x: pd.Series) -> float:
    try:
        num = pd.to_numeric(x, errors="coerce")
        return float(num.sum(skipna=True))
    except Exception:
        return 0.0

def safe_median(x: pd.Series) -> float:
    try:
        num = pd.to_numeric(x, errors="coerce").dropna()
        return float(statistics.median(num)) if len(num) else 0.0
    except Exception:
        return 0.0

def human_currency(x: float) -> str:
    try:
        x = float(x)
        if abs(x) >= 1_000_000_000: return f"${x/1_000_000_000:,.2f}B"
        if abs(x) >= 1_000_000:     return f"${x/1_000_000:,.2f}M"
        if abs(x) >= 1_000:         return f"${x/1_000:,.0f}K"
        return f"${x:,.2f}"
    except Exception:
        return "$0"

def human_number(x: float) -> str:
    try:
        x = float(x)
        if abs(x) >= 1_000_000: return f"{x/1_000_000:,.2f}M"
        if abs(x) >= 1_000:     return f"{x/1_000:,.0f}K"
        return f"{x:,.0f}"
    except Exception:
        return "0"

def compute_risk_score(row: pd.Series) -> float:
    score = 0.0
    if row.get("IS_PAST_EVENT", False): score += 50
    if row.get("ON_HOLD", False):       score += 30
    if row.get("DAYS_OPEN", 0) > 90:    score += 20
    return score

def categorize_risk(score: float) -> str:
    if score > 70: return "High"
    if score > 30: return "Medium"
    return "Low"

# ---------------------------
# Data Prep
# ---------------------------
@st.cache_data
def load_and_preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip().upper() if isinstance(c, str) else c for c in df.columns]

    # Ensure numeric columns exist and convert
    num_cols = ["OPEN_QTY","ORDER_QTY","SHIP_QTY","CANCEL_QTY","PRICE","OPEN_VALUE","ORDER_VALUE","DISCOUNT","DISC_RATE","EXCHANGE_RATE"]
    for c in num_cols:
        if c not in df.columns: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Dates
    for c in ["START_DATE","EVENT_DATE","CANCEL_DATE","BOOK_DATE"]:
        df[c] = safe_parse_dates(df, c)

    # Derived
    df["COMPUTED_OPEN_VALUE"]   = np.where(df["OPEN_VALUE"]>0, df["OPEN_VALUE"], df["OPEN_QTY"]*df["PRICE"])
    df["COMPUTED_ORDER_VALUE"]  = df["ORDER_QTY"]*df["PRICE"]
    df["COMPUTED_SHIP_VALUE"]   = df["SHIP_QTY"]*df["PRICE"]
    df["COMPUTED_CANCEL_VALUE"] = df["CANCEL_QTY"]*df["PRICE"]

    today = pd.Timestamp.now().normalize()
    df["_DASHBOARD_DATE"] = today
    df["DAYS_OPEN"]       = (today - df["START_DATE"]).dt.days.clip(lower=0)
    df["IS_PAST_EVENT"]   = (df["EVENT_DATE"].notna()) & (df["EVENT_DATE"] < today) & (df["OPEN_QTY"] > 0)
    df["IS_OPEN"]         = (df["OPEN_QTY"] > 0) | (df["ORDER_QTY"] > (df["SHIP_QTY"] + df["CANCEL_QTY"]))

    # On hold
    df["ON_HOLD"] = False
    for col in ["ORDER_HOLD", "CREDIT_HOLD", "HOLD_STATUS"]:
        if col in df.columns:
            df["ON_HOLD"] = df["ON_HOLD"] | df[col].astype(str).str.upper().isin(["Y","YES","TRUE","ON","1","HOLD","H"])

    # Fallbacks
    df["CURRENCY"]  = df.get("CURRENCY", "USD")
    if "CURRENCY" in df.columns:
        df["CURRENCY"] = df["CURRENCY"].fillna("USD").astype(str).str.upper()

    df["EXCHANGE_RATE"] = pd.to_numeric(df.get("EXCHANGE_RATE", 1), errors="coerce").fillna(1.0)
    df["OPEN_VALUE_BASE"]  = df["COMPUTED_OPEN_VALUE"] * df["EXCHANGE_RATE"]
    df["ORDER_VALUE_BASE"] = df["COMPUTED_ORDER_VALUE"] * df["EXCHANGE_RATE"]

    if "CUST_NAME" not in df.columns:
        df["CUST_NAME"] = df.get("CUSTOMER", "Unknown")
    df["CUST_NAME"] = df["CUST_NAME"].fillna("Unknown").astype(str)

    if "SKU" not in df.columns:
        df["SKU"] = df.get("STYLE", "").astype(str) + "-" + df.get("CLR", "").astype(str) + "-" + df.get("SIZE", "").astype(str)
    df["SKU"] = df["SKU"].fillna("UNKNOWN_SKU").replace("", "UNKNOWN_SKU")

    if "STYLE" not in df.columns:
        df["STYLE"] = df["SKU"].astype(str).str.split("-").str[0]

    if "ORDER_NO" not in df.columns:
        df["ORDER_NO"] = pd.Series([f"ORDER_{i+1}" for i in range(len(df))])

    if "BOOK_SEASON" not in df.columns:
        df["BOOK_SEASON"] = "Unknown"
    else:
        df["BOOK_SEASON"] = df["BOOK_SEASON"].fillna("Unknown")

    if "CATEGORY" not in df.columns:
        df["CATEGORY"] = "Unknown"

    if "COUNTRY" not in df.columns:
        df["COUNTRY"] = "Unknown"
    else:
        df["COUNTRY"] = df["COUNTRY"].fillna("Unknown")

    # DIV / Brand mapping
    brand_mapping = {
        "PILOTI": "PILOTI",
        "PILOTI LLC": "PILOTI",
        "PILOTI LLC CORP": "PILOTI",
        "PODIUM I BRANDS ATLANTIS": "ATLANTIS",
        "PODIUM I BRANDS BERKLY JENSEN PODIUM I BRANDS CLOUDVEIL": "BJ CLOUDVEIL",
        "PODIUM I BRANDS CLOUDVEIL": "CLOUDVEIL",
        "PODIUM I BRANDS COMRAD SOCKS": "COMRAD",
        "PODIUM I BRANDS GAIAM": "GAIAM",
        "PODIUM I BRANDS HOLDINGS LLC": "PODIUM HOME",
        "PODIUM I BRANDS HOSIERY": "HOSIERY",
        "PODIUM I BRANDS IP": "IP",
        "PODIUM I BRANDS LLC - PODIUM HOME": "PODIUM HOME",
        "PODIUM I BRANDS LLC CORP": "PODIUM HOME",
        "PODIUM I BRANDS MISSION": "MISSION",
        "PODIUM I BRANDS RO+ME": "RO+ME",
        "PODIUM I BRANDS ROBEEZ": "ROBEEZ",
        "PODIUM I BRANDS TIU": "TIU",
        "PODIUM I BRANDS TRUMPETTE": "TRUMPETTE",
        "PODIUM I BRANDS ZANELLA": "ZANELLA",
        "PODIUM II BRANDS LLC CORP": "PODIUM II",
        "PODIUM II BRANDS PRIVATE LABEL": "PRIVATE LABEL",
        "STOCK COLLECTION": "STOCK COLLECTION",
        "SWIMS NA LLC": "SWIMS",
        # Also map shorthand codes if present
        "P": "PILOTI",
        "16": "PILOTI",
        "94": "PILOTI",
        "3": "ATLANTIS",
        "BJ": "BJ CLOUDVEIL",
        "2": "CLOUDVEIL",
        "14": "COMRAD",
        "12": "GAIAM",
        "97": "PODIUM HOME",
        "9": "HOSIERY",
        "96": "IP",
        "11": "PODIUM HOME",
        "98": "PODIUM HOME",
        "10": "MISSION",
        "5": "RO+ME",
        "4": "ROBEEZ",
        "13": "TIU",
        "7": "TRUMPETTE",
        "1": "ZANELLA",
        "99": "PODIUM II",
        "61": "PRIVATE LABEL",
        "8": "PRIVATE LABEL",
        "15": "STOCK COLLECTION",
        "6": "SWIMS",
    }
    if "DIV" in df.columns:
        df["DIV"] = df["DIV"].astype(str).str.strip()
        df["DIV_BRAND"] = df["DIV"].map(brand_mapping).fillna(df["DIV"])
    else:
        df["DIV_BRAND"] = "Unknown"

    # Risk
    df["RISK_SCORE"]    = df.apply(compute_risk_score, axis=1)
    df["RISK_CATEGORY"] = df["RISK_SCORE"].apply(categorize_risk)

    # Aging buckets
    aging_bins   = [0, 30, 60, 90, 120, 180, np.inf]
    aging_labels = ["<30", "30-60", "60-90", "90-120", "120-180", ">180"]
    df["AGING_BUCKET"] = pd.cut(df["DAYS_OPEN"], bins=aging_bins, labels=aging_labels, include_lowest=True)
    return df

def compute_kpis(df: pd.DataFrame):
    k = {}
    k["total_orders"]          = df["ORDER_NO"].nunique()
    k["total_open_units"]      = safe_sum(df["OPEN_QTY"])
    k["total_open_value"]      = safe_sum(df["COMPUTED_OPEN_VALUE"])
    k["total_order_value"]     = safe_sum(df["COMPUTED_ORDER_VALUE"])
    k["total_shipped_units"]   = safe_sum(df["SHIP_QTY"])
    k["total_shipped_value"]   = safe_sum(df["COMPUTED_SHIP_VALUE"])
    k["total_cancelled_units"] = safe_sum(df["CANCEL_QTY"])
    k["total_cancelled_value"] = safe_sum(df["COMPUTED_CANCEL_VALUE"])
    k["total_discount"]        = safe_sum(df["DISCOUNT"])

    # Rates
    denom = (k["total_shipped_units"] + k["total_open_units"] + k["total_cancelled_units"])
    k["unit_fill_rate"]  = (k["total_shipped_units"] / k["total_open_units"] * 100) if k["total_open_units"] else 0.0
    k["value_fill_rate"] = (k["total_shipped_value"] / k["total_open_value"] * 100) if k["total_open_value"] else 0.0
    k["backorder_rate"]  = (k["total_open_units"] / denom * 100) if denom else 0.0
    k["cancellation_rate"] = (k["total_cancelled_units"] / denom * 100) if denom else 0.0
    k["discount_rate"]   = (k["total_discount"] / k["total_order_value"] * 100) if k["total_order_value"] else 0.0

    # Time
    k["median_lead_time"] = safe_median((df["EVENT_DATE"] - df["START_DATE"]).dt.days.clip(lower=0) if df["EVENT_DATE"].notna().any() else pd.Series([0]*len(df)))
    k["median_days_open"] = safe_median(df["DAYS_OPEN"])

    # Risk/Backlog
    k["total_past_due_units"]  = safe_sum(df.loc[df["IS_PAST_EVENT"], "OPEN_QTY"])
    k["total_past_due_value"]  = safe_sum(df.loc[df["IS_PAST_EVENT"], "COMPUTED_OPEN_VALUE"])
    k["total_on_hold_value"]   = safe_sum(df.loc[df["ON_HOLD"], "COMPUTED_OPEN_VALUE"])
    k["total_high_risk_value"] = safe_sum(df.loc[df["RISK_CATEGORY"]=="High", "COMPUTED_OPEN_VALUE"])
    k["high_risk_orders"]      = df.loc[df["RISK_CATEGORY"]=="High","ORDER_NO"].nunique()
    return k

@st.cache_data
def compute_groupings(df: pd.DataFrame):
    g = {}
    # Customers
    g["customer_open_value"] = df.groupby("CUST_NAME")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)
    g["customer_open_units"] = df.groupby("CUST_NAME")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False)

    # Styles / SKU
    g["top_skus_value"] = df.groupby("SKU")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False).head(10)
    top_skus_list = g["top_skus_value"]["SKU"].tolist()
    df_top = df[df["SKU"].isin(top_skus_list)]
    g["top_sku_customer_value"] = df_top.groupby(["SKU","CUST_NAME"])["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values(["SKU","COMPUTED_OPEN_VALUE"], ascending=[True,False])

    g["style_open_units"] = df.groupby("STYLE")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False)
    g["style_open_value"] = df.groupby("STYLE")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)

    # Brand / Div
    g["brand_open_value"] = df.groupby("DIV_BRAND")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)
    g["brand_open_units"] = df.groupby("DIV_BRAND")["OPEN_QTY"].sum().reset_index().sort_values("OPEN_QTY", ascending=False)

    # Country & Ship Via (optional)
    if "COUNTRY" in df.columns:
        g["country_open_value"] = df.groupby("COUNTRY")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)
    if "SHIP_VIA" in df.columns:
        g["ship_via_value"] = df.groupby("SHIP_VIA")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)

    # Monthly trends
    df["_MONTH_START"] = df["START_DATE"].dt.to_period("M").dt.to_timestamp()
    g["monthly_open_value"] = df.groupby("_MONTH_START")["COMPUTED_OPEN_VALUE"].sum().reset_index().dropna().sort_values("_MONTH_START")
    g["monthly_open_units"] = df.groupby("_MONTH_START")["OPEN_QTY"].sum().reset_index().dropna().sort_values("_MONTH_START")

    # Risk
    g["risk_by_category_value"] = df.groupby("RISK_CATEGORY")["COMPUTED_OPEN_VALUE"].sum().reset_index()
    g["past_due_by_customer"]   = df[df["IS_PAST_EVENT"]].groupby("CUST_NAME")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)
    g["on_hold_by_brand"]       = df[df["ON_HOLD"]].groupby("DIV_BRAND")["COMPUTED_OPEN_VALUE"].sum().reset_index().sort_values("COMPUTED_OPEN_VALUE", ascending=False)

    return g

# ---------------------------
# Sidebar - Upload & Filters
# ---------------------------
with st.sidebar:
    st.header("1) Upload Open Order Report")
    uploaded_file = st.file_uploader("CSV or Excel", type=["csv","xlsx"], accept_multiple_files=False)

    st.header("2) Filters")
    date_filter_enabled = st.checkbox("Filter by Start Date", value=False)
    if date_filter_enabled:
        col1, col2 = st.columns(2)
        with col1:
            start_from = st.date_input("Start From", value=(datetime.now() - timedelta(days=365)))
        with col2:
            start_to   = st.date_input("Start To", value=datetime.now())
    else:
        start_from, start_to = None, None

    min_open_value = st.number_input("Min Open Value", value=0.0, step=100.0)

# Load
if not uploaded_file:
    st.info("Upload your report to begin. Expected columns include: ORDER_NO, START_DATE, EVENT_DATE, OPEN_QTY, ORDER_QTY, SHIP_QTY, CANCEL_QTY, PRICE, OPEN_VALUE, DISCOUNT, CUST_NAME, SKU, STYLE, DIV, BOOK_SEASON, COUNTRY, SHIP_VIA, CATEGORY.")
    st.stop()

try:
    if uploaded_file.name.endswith(".xlsx"):
        df_raw = pd.read_excel(uploaded_file, dtype=str)
    else:
        try:
            df_raw = pd.read_csv(uploaded_file, dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, dtype=str, encoding="latin1")
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

df = load_and_preprocess(df_raw)

# Dynamic filters
with st.sidebar:
    customers  = sorted(df["CUST_NAME"].unique().tolist())
    styles     = sorted(df["STYLE"].unique().tolist())
    brands     = sorted(df["DIV_BRAND"].unique().tolist())
    countries  = sorted(df["COUNTRY"].unique().tolist()) if "COUNTRY" in df.columns else []
    skus       = sorted(df["SKU"].unique().tolist())[:5000]
    risks      = sorted(df["RISK_CATEGORY"].unique().tolist())

    st.markdown("---")
    customer_filter = st.multiselect("Customers", options=customers, default=[])
    style_filter    = st.multiselect("Styles", options=styles, default=[])
    brand_filter    = st.multiselect("Brands / Div", options=brands, default=[])
    country_filter  = st.multiselect("Countries", options=countries, default=[])
    sku_filter      = st.multiselect("SKUs", options=skus, default=[])
    risk_filter     = st.multiselect("Risk Categories", options=risks, default=[])

    if st.button("Apply Filters"):
        st.rerun()

# Apply filters
df_filtered = df.copy()
if date_filter_enabled and start_from and start_to:
    min_ts = pd.to_datetime(start_from)
    max_ts = pd.to_datetime(start_to) + timedelta(days=1) - timedelta(seconds=1)
    df_filtered = df_filtered[(df_filtered["START_DATE"] >= min_ts) & (df_filtered["START_DATE"] <= max_ts)]
if customer_filter:
    df_filtered = df_filtered[df_filtered["CUST_NAME"].isin(customer_filter)]
if style_filter:
    df_filtered = df_filtered[df_filtered["STYLE"].isin(style_filter)]
if brand_filter:
    df_filtered = df_filtered[df_filtered["DIV_BRAND"].isin(brand_filter)]
if country_filter:
    df_filtered = df_filtered[df_filtered["COUNTRY"].isin(country_filter)]
if sku_filter:
    df_filtered = df_filtered[df_filtered["SKU"].isin(sku_filter)]
if risk_filter:
    df_filtered = df_filtered[df_filtered["RISK_CATEGORY"].isin(risk_filter)]

df_filtered = df_filtered[df_filtered["COMPUTED_OPEN_VALUE"] >= min_open_value]

if df_filtered.empty:
    st.warning("No data after filtering. Adjust filters.")
    st.stop()

# Compute KPIs & groupings
kpis = compute_kpis(df_filtered)
group = compute_groupings(df_filtered)

# ---------------------------
# Tabs
# ---------------------------
tabs = st.tabs(["Overview", "Operations", "Finance", "Risk & Trends", "Production", "Graphs & Visuals", "Detailed Analysis & Data"])

# Overview
with tabs[0]:
    st.subheader("Executive Overview — Key Totals and Rates")
    cols = st.columns(5)
    metrics = [
        {"label": "Total Open Value", "value": human_currency(kpis["total_open_value"]), "help": "Sum of all open order values."},
        {"label": "Total Open Units", "value": human_number(kpis["total_open_units"]), "help": "Sum of open quantities."},
        {"label": "Total Orders", "value": human_number(kpis["total_orders"]), "help": "Unique order count."},
        {"label": "Unit Fill Rate", "value": f"{kpis['unit_fill_rate']:.1f}%", "help": "Shipped units / Open units."},
        {"label": "Median Days Open", "value": f"{kpis['median_days_open']:.1f} days", "help": "Median days orders stay open."},
        {"label": "Total Shipped Value", "value": human_currency(kpis["total_shipped_value"]), "help": "Sum of shipped values."},
        {"label": "Cancellation Rate", "value": f"{kpis['cancellation_rate']:.1f}%", "help": "Cancelled units / Total units."},
    ]
    for i, m in enumerate(metrics):
        with cols[i % len(cols)]:
            st.metric(**m)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(group["customer_open_value"].head(15), values="COMPUTED_OPEN_VALUE", names="CUST_NAME", title="Open Value by Customer (Top 15)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(group["monthly_open_value"], x="_MONTH_START", y="COMPUTED_OPEN_VALUE", title="Monthly Open Value Trend")
        st.plotly_chart(fig, use_container_width=True)

# Operations
with tabs[1]:
    st.subheader("Operations — Fulfillment & Priorities")
    cols = st.columns(5)
    metrics = [
        {"label": "Value Fill Rate", "value": f"{kpis['value_fill_rate']:.1f}%", "help": "Shipped value / Open value."},
        {"label": "Backorder Rate", "value": f"{kpis['backorder_rate']:.1f}%", "help": "Open units / Total."},
        {"label": "Total Past Due Units", "value": human_number(kpis["total_past_due_units"]), "help": "Past-due open units."},
        {"label": "On Hold Value", "value": human_currency(kpis["total_on_hold_value"]), "help": "Open value currently on hold."},
        {"label": "Median Lead Time", "value": f"{kpis['median_lead_time']:.1f} days", "help": "Median days from start to event."},
    ]
    for i, m in enumerate(metrics):
        with cols[i % len(cols)]:
            st.metric(**m)

    st.markdown("**Top SKUs — Open Value broken down by Customer**")
    if not group["top_sku_customer_value"].empty:
        fig = px.bar(group["top_sku_customer_value"], x="SKU", y="COMPUTED_OPEN_VALUE", color="CUST_NAME", title="Top SKUs by Open Value (Stacked by Customer)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No SKU/Customer data to show.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(group["customer_open_units"].head(15), x="CUST_NAME", y="OPEN_QTY", title="Top Customers by Open Units")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(group["brand_open_units"].head(15), x="DIV_BRAND", y="OPEN_QTY", title="Open Units by Brand/Div")
        st.plotly_chart(fig, use_container_width=True)

# Finance
with tabs[2]:
    st.subheader("Finance — Revenue & Margins (Meaningful KPIs)")
    cols = st.columns(5)
    metrics = [
        {"label": "Total Open Value", "value": human_currency(kpis["total_open_value"]), "help": "Sum of open values."},
        {"label": "Total Order Value", "value": human_currency(kpis["total_order_value"]), "help": "Sum of order values."},
        {"label": "Total Discount", "value": human_currency(kpis["total_discount"]), "help": "Sum of discount amounts."},
        {"label": "Discount Rate", "value": f"{kpis['discount_rate']:.1f}%", "help": "Discount / Order value."},
        {"label": "Shipped Value", "value": human_currency(kpis["total_shipped_value"]), "help": "Sum of shipped value."},
    ]
    for i, m in enumerate(metrics):
        with cols[i % len(cols)]:
            st.metric(**m)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(group["customer_open_value"].head(15), x="CUST_NAME", y="COMPUTED_OPEN_VALUE", title="Top Customers by Open Value")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(group["brand_open_value"].head(15), x="DIV_BRAND", y="COMPUTED_OPEN_VALUE", title="Open Value by Brand/Div")
        st.plotly_chart(fig, use_container_width=True)

    # Discount vs Order Value (scatter)
    st.markdown("**Discount % vs Order Value (size = Open Value)**")
    df_scatter = df_filtered.copy()
    df_scatter["DISC_RATE_SAFE"] = pd.to_numeric(df_scatter.get("DISC_RATE", 0), errors="coerce").fillna(0.0)
    fig = px.scatter(df_scatter, x="COMPUTED_ORDER_VALUE", y="DISC_RATE_SAFE", size="COMPUTED_OPEN_VALUE",
                     hover_data=["CUST_NAME","DIV_BRAND","STYLE","SKU"],
                     title="Discount Rate vs Order Value")
    st.plotly_chart(fig, use_container_width=True)

# Risk & Trends
with tabs[3]:
    st.subheader("Risk & Trends — Actionable Views")
    any_visual = False

    if not group["risk_by_category_value"].empty:
        any_visual = True
        fig = px.pie(group["risk_by_category_value"], values="COMPUTED_OPEN_VALUE", names="RISK_CATEGORY", title="Open Value by Risk Category")
        st.plotly_chart(fig, use_container_width=True)

    if not group["past_due_by_customer"].empty:
        any_visual = True
        fig = px.bar(group["past_due_by_customer"].head(20), x="CUST_NAME", y="COMPUTED_OPEN_VALUE", title="Past-Due Open Value by Customer")
        st.plotly_chart(fig, use_container_width=True)

    if "on_hold_by_brand" in group and not group["on_hold_by_brand"].empty:
        any_visual = True
        fig = px.bar(group["on_hold_by_brand"], x="DIV_BRAND", y="COMPUTED_OPEN_VALUE", title="On-Hold Open Value by Brand/Div")
        st.plotly_chart(fig, use_container_width=True)

    if not any_visual:
        st.info("This section is hidden because there are no meaningful Risk/Trend visuals for the current filters.")

# Production
with tabs[4]:
    st.subheader("Production — Backlog & Mix")
    cols = st.columns(4)
    metrics = [
        {"label": "Total Past Due Value", "value": human_currency(kpis["total_past_due_value"]), "help": "Open value past the event date."},
        {"label": "Total Past Due Units", "value": human_number(kpis["total_past_due_units"]), "help": "Open units past due."},
        {"label": "High-Risk Orders", "value": human_number(kpis["high_risk_orders"]), "help": "Count of high risk orders."},
        {"label": "High-Risk Value", "value": human_currency(kpis["total_high_risk_value"]), "help": "Open value flagged as high risk."},
    ]
    for i, m in enumerate(metrics):
        with cols[i % len(cols)]:
            st.metric(**m)

    st.markdown("**Top SKUs with Customer Amounts (Production Priority)**")
    if not group["top_sku_customer_value"].empty:
        fig = px.bar(group["top_sku_customer_value"], x="SKU", y="COMPUTED_OPEN_VALUE", color="CUST_NAME",
                     title="Top SKUs by Open Value — Customer Breakdown")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No SKU/Customer data to show.")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(group["style_open_units"].head(15), x="STYLE", y="OPEN_QTY", title="Top Styles by Open Units")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.treemap(group["brand_open_value"].head(25), path=["DIV_BRAND"], values="COMPUTED_OPEN_VALUE", title="Open Value by Brand/Div (Treemap)")
        st.plotly_chart(fig, use_container_width=True)

# Graphs & Visuals
with tabs[5]:
    st.subheader("Additional Visuals")
    col1, col2 = st.columns(2)
    with col1:
        # Replace generic scatter with more revealing bubble
        fig = px.scatter(df_filtered, x="DAYS_OPEN", y="COMPUTED_OPEN_VALUE", size="OPEN_QTY", color="DIV_BRAND",
                         hover_data=["CUST_NAME","STYLE","SKU"],
                         title="Open Value vs Days Open (size=units, color=Brand)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "ship_via_value" in group and not group["ship_via_value"].empty:
            fig = px.bar(group["ship_via_value"], x="SHIP_VIA", y="COMPUTED_OPEN_VALUE", title="Open Value by Ship Via")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Ship Via data.")

    # Brand vs Customer heatmap-like matrix (pivot then imshow)
    mat = df_filtered.pivot_table(index="DIV_BRAND", columns="CUST_NAME", values="COMPUTED_OPEN_VALUE", aggfunc="sum", fill_value=0)
    if not mat.empty:
        fig = px.imshow(mat, aspect="auto", title="Heatmap: Brand/Div vs Customer (Open Value)")
        st.plotly_chart(fig, use_container_width=True)

# Detailed
with tabs[6]:
    st.subheader("Detailed Analysis")
    st.markdown(f"""
- **Total open value:** {human_currency(kpis["total_open_value"])} across **{human_number(kpis["total_orders"])}** orders.
- **Fill rates:** Units {kpis['unit_fill_rate']:.1f}% • Value {kpis['value_fill_rate']:.1f}%.
- **Discounts:** {human_currency(kpis["total_discount"])} total • Rate {kpis['discount_rate']:.1f}%.
- **Backlog/Risk:** Past-due value {human_currency(kpis["total_past_due_value"])} • On-hold {human_currency(kpis["total_on_hold_value"])} • High-risk {human_currency(kpis["total_high_risk_value"])}.
""")

    # Brand/Div insights
    st.subheader("Brand / Division Insights")
    top_brands = group["brand_open_value"].head(10)
    if not top_brands.empty:
        st.dataframe(top_brands.rename(columns={"COMPUTED_OPEN_VALUE":"OPEN_VALUE"}), use_container_width=True)
    else:
        st.info("No Brand/Division aggregation available.")

    st.subheader("Raw Filtered Data (first 1000 rows)")
    st.dataframe(df_filtered.head(1000), use_container_width=True)

    st.subheader("Export")
    col1, col2 = st.columns(2)
    with col1:
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered CSV", csv, "filtered_open_orders.csv", "text/csv")
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            df_filtered.to_excel(writer, index=False, sheet_name="Filtered")
        excel_buffer.seek(0)
        st.download_button("Download Filtered Excel", excel_buffer, "filtered_open_orders.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.info("This dashboard emphasizes Div/Brand across all departments, removes weak visuals, and adds SKU×Customer breakdowns to reflect the real operational and financial picture.")
