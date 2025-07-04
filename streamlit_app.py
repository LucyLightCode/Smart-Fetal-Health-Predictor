import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix

# Load model
model = joblib.load("random_forest_model.pkl")

# --- Title Section with Style ---
st.markdown(
    """
    <div style="background-color:#fce4ec;padding:20px;border-radius:10px;">
        <h1 style="color:#d81b60;text-align:center;">👶 Smart Fetal Health Predictor</h1>
        <p style="text-align:center; font-size:18px; color:#4a148c;">
            Predict fetal health status using CTG input features.
        </p>
    </div>
    """, unsafe_allow_html=True
)

# --- Feature Names ---
feature_names = [
    "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "histogram_mean",
    "histogram_mode",
    "histogram_median",
    "prolongued_decelerations",
    "mean_value_of_long_term_variability",
    "accelerations",
    "histogram_variance"
]

# --- Input Form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        abnormal_short_term_variability = st.slider("Abnormal Short Term Variability", 0.0, 100.0, 20.0)
        mean_value_of_short_term_variability = st.slider("Mean Value of Short Term Variability", 0.0, 10.0, 2.0)
        percentage_of_time_with_abnormal_long_term_variability = st.slider("% of Time with Abnormal Long Term Variability", 0.0, 100.0, 10.0)
        histogram_mean = st.slider("Histogram Mean", 80.0, 160.0, 130.0)
        histogram_mode = st.slider("Histogram Mode", 80.0, 160.0, 120.0)

    with col2:
        histogram_median = st.slider("Histogram Median", 80.0, 160.0, 125.0)
        prolongued_decelerations = st.number_input("Prolongued Decelerations", min_value=0.0, value=0.0)
        mean_value_of_long_term_variability = st.slider("Mean Value of Long Term Variability", 0.0, 100.0, 10.0)
        accelerations = st.number_input("Accelerations", min_value=0.0, value=0.003)
        histogram_variance = st.slider("Histogram Variance", 0.0, 100.0, 10.0)

    colA, colB = st.columns(2)
    with colA:
        submitted = st.form_submit_button("🚀 Predict", type="primary")
    with colB:
        clear = st.form_submit_button("🧹 Clear", type="secondary")

# --- Single Prediction ---
if submitted:
    input_df = pd.DataFrame([[abnormal_short_term_variability,
                              mean_value_of_short_term_variability,
                              percentage_of_time_with_abnormal_long_term_variability,
                              histogram_mean,
                              histogram_mode,
                              histogram_median,
                              prolongued_decelerations,
                              mean_value_of_long_term_variability,
                              accelerations,
                              histogram_variance]], columns=feature_names)

    prediction = model.predict(input_df)[0]
    label = {1: "🟢 Normal", 2: "🟠 Suspect", 3: "🔴 Pathological"}.get(prediction, "Unknown")

    st.markdown(
        f"""
        <div style="padding:20px;background-color:#e8f5e9;border-radius:10px;">
            <h3 style="color:#1b5e20;">🔍 Prediction Result</h3>
            <h2 style="color:#1b5e20;">Fetal Health Status: <b>{label}</b></h2>
        </div>
        """, unsafe_allow_html=True
    )

elif clear:
    st.rerun()

# --- Dataset Upload Section ---
st.markdown("---")
st.header("📁 Test with Dataset")
uploaded_file = st.file_uploader("Upload a CSV file with the required features", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        # ✅ Check if all required features exist
        if all(col in data.columns for col in feature_names):
            predictions = model.predict(data[feature_names])
            data["Predicted"] = predictions
            data["Predicted_Label"] = data["Predicted"].map({1: "Normal", 2: "Suspect", 3: "Pathological"})

            st.success("✅ Predictions completed!")
            st.dataframe(data)

            # Optional: check accuracy if 'target' exists
            if "target" in data.columns:
                acc = accuracy_score(data["target"], data["Predicted"])
                st.markdown(f"**📊 Accuracy vs. Target: `{acc:.2%}`**")
                st.text(confusion_matrix(data["target"], data["Predicted"]))

            # Download results
            st.download_button("📥 Download Results as CSV",
                               data=data.to_csv(index=False),
                               file_name="fetal_predictions.csv",
                               mime="text/csv")
        else:
            st.error("❌ Your CSV is missing one or more required features.")
    except Exception as e:
        st.error(f"⚠️ Error reading CSV: {e}")
