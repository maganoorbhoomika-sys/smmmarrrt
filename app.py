import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# ── Load model & scaler ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("logistic_regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ── UI ────────────────────────────────────────────────────────
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient's clinical measurements to predict diabetes risk.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.number_input("Pregnancies",               min_value=0,   max_value=17,  value=1,   step=1)
    glucose        = st.number_input("Glucose (mg/dL)",           min_value=0,   max_value=200, value=120, step=1)
    blood_pressure = st.number_input("Blood Pressure (mmHg)",     min_value=0,   max_value=122, value=70,  step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)",       min_value=0,   max_value=99,  value=20,  step=1)

with col2:
    insulin = st.number_input("Insulin (muU/ml)",                 min_value=0,   max_value=846, value=80,  step=1)
    bmi     = st.number_input("BMI",                              min_value=0.0, max_value=67.1,value=25.0,step=0.1, format="%.1f")
    dpf     = st.number_input("Diabetes Pedigree Function",       min_value=0.0, max_value=2.5, value=0.4, step=0.01,format="%.2f")
    age     = st.number_input("Age (years)",                      min_value=1,   max_value=100, value=30,  step=1)

st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    input_df = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, dpf, age]],
        columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age']
    )

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes**\n\nPredicted probability: **{probability:.1%}**")
    else:
        st.success(f"✅ **Low Risk of Diabetes**\n\nPredicted probability: **{probability:.1%}**")

    st.caption("Risk probability")
    st.progress(float(probability))

st.divider()
st.caption("Model: Logistic Regression · Dataset: Pima Indians Diabetes · For educational purposes only.")
