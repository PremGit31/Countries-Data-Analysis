import streamlit as st
import pandas as pd
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Country Indicator Dashboard",
    layout="wide"
)

st.title("🌍 Country Indicator Explorer")
st.caption("Filter countries by multiple socio-economic indicators")

# -------------------- DATA LOADING --------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/df.csv")

df = load_data()

# -------------------- BASIC VALIDATION --------------------
if "country_or_territory" not in df.columns:
    st.error("CSV must contain a 'country_or_territory' column")
    st.stop()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) == 0:
    st.error("No numeric columns found for filtering")
    st.stop()

# -------------------- SIDEBAR FILTERS --------------------
st.sidebar.header("🔧 Filter Settings")

selected_metrics = st.sidebar.multiselect(
    "Select metrics to filter",
    numeric_cols,
    default=numeric_cols[:3]
)

filters = {}

for col in selected_metrics:
    min_val = float(df[col].min())
    max_val = float(df[col].max())

    filters[col] = st.sidebar.slider(
        f"{col} range",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )

# -------------------- APPLY FILTERS --------------------
filtered_df = df.copy()

for col, (low, high) in filters.items():
    filtered_df = filtered_df[
        (filtered_df[col] >= low) & (filtered_df[col] <= high)
    ]

# -------------------- RESULTS --------------------
st.subheader("📊 Filtered Results")

col1, col2, col3 = st.columns(3)

col1.metric("Total Countries", len(df))
col2.metric("Matching Countries", len(filtered_df))
col3.metric("Active Filters", len(filters))

st.divider()

st.dataframe(
    filtered_df.sort_values("country_or_territory "),
    use_container_width=True
)

# -------------------- OPTIONAL DOWNLOAD --------------------
st.download_button(
    "⬇️ Download Filtered Data",
    filtered_df.to_csv(index=False),
    "filtered_countries.csv",
    "text/csv"
)

# -------------------- RAW DATA VIEW --------------------
with st.expander("🔍 View Raw Data"):
    st.dataframe(df, use_container_width=True)
