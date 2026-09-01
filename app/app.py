import streamlit as st
import pandas as pd

st.set_page_config(page_title="CSV Viewer", layout="wide")

st.title("CSV Dataset Viewer")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read the CSV file: {e}")
        st.stop()

    st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    st.subheader("Preview")
    st.dataframe(df)

    st.subheader("Column Info")
    st.dataframe(df.dtypes.rename("dtype"))
