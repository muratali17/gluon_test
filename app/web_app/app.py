import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import AutoGluonML

st.set_page_config(page_title="AutoGluon Trainer", layout="wide")

st.title("AutoGluon ML Trainer")

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

    st.subheader("Training Configuration")
    label = st.selectbox("Target label", df.columns)
    task_name = st.text_input("Task name", value="task_finding_best_pipe")
    time_limit = st.number_input("Time limit (seconds)", min_value=10, value=300, step=10)

    if st.button("Train", type="primary"):
        try:
            ag = AutoGluonML()
            with st.spinner("Training in progress..."):
                ag.train(df, label=label, time_limit=time_limit)
                model_path = ag.save_model(task_name)

            st.success(f"Model saved to: {model_path}")

            st.subheader("Leaderboard")
            leaderboard = ag.predictor.leaderboard(ag.test_data)
            st.dataframe(leaderboard)
        except Exception as e:
            st.error(f"Training failed: {e}")
