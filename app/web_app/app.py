import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml import AutoGluonML
from ml.utilities import list_task_names

st.set_page_config(page_title="AutoGluon Trainer", layout="wide")

st.title("AutoGluon ML Trainer")

tab_upload, tab_train, tab_predict = st.tabs(["Upload Data", "Train Model", "Predict"])

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

with tab_predict:
    st.subheader("Saved Models")

    task_names = list_task_names()

    if not task_names:
        st.info("No saved models found.")
    else:
        selected_model = st.selectbox("Select a saved model", task_names)

        if st.button("Load Model", type="primary"):
            try:
                ag = AutoGluonML()
                with st.spinner("Loading model..."):
                    ag.load(selected_model)
                st.session_state.loaded_model = selected_model
                st.session_state.ag = ag
                st.success(f"Model '{selected_model}' loaded.")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    if "ag" in st.session_state:
        ag = st.session_state.ag
        st.subheader(f"Predict with '{st.session_state.loaded_model}'")

        label = ag.predictor.label
        features = ag.predictor.features()
        type_map = ag.predictor.feature_metadata_in.get_type_group_map_raw()

        int_cols = type_map.get("int", [])
        float_cols = type_map.get("float", [])

        mode = st.radio(
            "Prediction mode",
            ["Single prediction", "Batch prediction (upload CSV)"],
        )

        if mode == "Single prediction":
            st.markdown(f"Enter one row of data to predict **{label}**.")

            sample = ag.sample_row
            if sample is not None:
                st.caption("Prefilled with a sample row from the training data.")

            with st.form("single_prediction_form"):
                row = {}
                for col in features:
                    if sample is not None and col in sample.index and not pd.isna(sample[col]):
                        default = sample[col]
                    else:
                        default = None

                    if col in int_cols:
                        row[col] = st.number_input(col, step=1, value=int(default) if default is not None else 0)
                    elif col in float_cols:
                        row[col] = st.number_input(col, value=float(default) if default is not None else 0.0)
                    else:
                        row[col] = st.text_input(col, value=str(default) if default is not None else "")

                if st.form_submit_button("Predict"):
                    try:
                        input_df = pd.DataFrame([row], columns=features)
                        prediction = ag.predict(input_df)
                        st.success(f"Predicted {label}: {prediction.iloc[0]}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

        else:
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="predict_csv")

            if uploaded_file is not None:
                try:
                    predict_df = pd.read_csv(uploaded_file)
                except Exception as e:
                    st.error(f"Failed to read the CSV file: {e}")
                    st.stop()

                st.subheader("Uploaded Data")
                st.dataframe(predict_df)

                if st.button("Predict"):
                    try:
                        input_df = (
                            predict_df.drop(columns=[label])
                            if label in predict_df.columns
                            else predict_df
                        )
                        predictions = ag.predict(input_df)

                        result = input_df.copy()
                        result[label] = predictions

                        st.subheader("Predictions")
                        st.dataframe(result)

                        st.download_button(
                            "Download predictions",
                            result.to_csv(index=False).encode("utf-8"),
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
