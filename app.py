
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from predict import predict_batch
FEATURE_NAMES = joblib.load("feature_names.pkl")
NUM_FEATURES = len(FEATURE_NAMES)

# --------------------------------------------------
# Streamlit Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Accident Prediction System",
    layout="wide"
)

st.title("🚦 Accident Prediction – Batch Inference")
st.write(
    "Upload a CSV file containing multiple records. "
    "The system will predict accident risk for each record."
)

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Column Validation
    # --------------------------------------------------
    missing_cols = set(FEATURE_NAMES) - set(df.columns)
    extra_cols = set(df.columns) - set(FEATURE_NAMES)

    if missing_cols:
        st.error(f"❌ Missing required columns: {missing_cols}")
        st.stop()

    if extra_cols:
        st.warning(f"⚠️ Extra columns will be ignored: {extra_cols}")
        df = df[FEATURE_NAMES]

    # Ensure correct order
    df = df[FEATURE_NAMES]

    # --------------------------------------------------
    # Threshold Control
    # --------------------------------------------------
    threshold = st.slider(
        "Accident Risk Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.35,
        step=0.05
    )

    # --------------------------------------------------
    # Run Prediction
    # --------------------------------------------------
    if st.button("Run Prediction"):
        with st.spinner("Predicting accidents..."):
            X = df.values.astype(np.float32)
            labels, probs = predict_batch(X, threshold=threshold)

        # Add outputs
        df["Accident_Probability"] = probs.round(3)
        df["Accident_Prediction"] = pd.Series(labels).map(
            {0: "No Accident", 1: "Accident Likely"}
        )

        # --------------------------------------------------
        # Display Results
        # --------------------------------------------------
        st.subheader("Prediction Results")
        st.dataframe(df)

        # --------------------------------------------------
        # Download Results
        # --------------------------------------------------
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Predictions",
            data=csv,
            file_name="accident_predictions.csv",
            mime="text/csv"
        )
