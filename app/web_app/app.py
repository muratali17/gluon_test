import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import AutoGluonML
from ml.utilities import list_task_names

st.set_page_config(page_title="AutoGluon Trainer", layout="wide")

st.title("AutoGluon ML Trainer")

tab_upload, tab_train, tab_saved = st.tabs(["Upload Data", "Train Model", "Saved Models"])

with tab_upload:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read the CSV file: {e}")
            st.stop()

        st.session_state.df = df

    if "df" in st.session_state:
        df = st.session_state.df
        st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        st.subheader("Preview")
        st.dataframe(df)

        st.subheader("Column Info")
        st.dataframe(df.dtypes.rename("dtype"))

with tab_train:
    if "df" not in st.session_state:
        st.info("Please upload a CSV file in the Upload Data tab first.")
    else:
        df = st.session_state.df

        st.subheader("Training Configuration")

        label = st.selectbox("Target label", df.columns)

        task_name = st.text_input("Task name", value="task_finding_best_pipe")

        time_limit = st.number_input("Time limit (seconds)", min_value=10, value=300, step=10)

        preset = st.selectbox("Preset", ["medium", "best"])

        if st.button("Train", type="primary"):
            try:
                ag = AutoGluonML()
                with st.spinner("Training in progress..."):
                    ag.train(df, label=label, time_limit=time_limit, task_name=task_name, presets=preset)

                st.success(f"Model saved ")

                st.subheader("Leaderboard")
                leaderboard = ag.predictor.leaderboard(ag.test_data)
                st.dataframe(leaderboard)

                st.subheader("Feature Importance")
                st.dataframe(ag.feature_importance(ag.test_data))
            except Exception as e:
                st.error(f"Training failed: {e}")

with tab_saved:
    st.subheader("Saved Models")

    task_names = list_task_names()

    if not task_names:
        st.info("No saved models found.")
    else:
        st.selectbox("Select a saved model", task_names)
