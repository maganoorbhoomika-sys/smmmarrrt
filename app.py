import streamlit as st
import numpy as np

# Model parameters hardcoded — no sklearn/joblib needed
SCALER_MEAN  = [3.845052, 121.656250, 72.386719, 29.108073,
                140.671875, 32.455208, 0.471876, 33.240885]
SCALER_SCALE = [3.367384, 30.418463, 12.088764, 8.785496,
                86.326802, 6.870699, 0.331113, 11.752573]
COEF         = [0.412782, 1.132359, -0.106985, 0.034343,
               -0.094010, 0.634458, 0.286307, 0.153491]
INTERCEPT    = -0.849830

def predict(values):
    x = np.array(values, dtype=float)
    x_scaled = (x - SCALER_MEAN) / SCALER_SCALE
    log_odds  = np.dot(COEF, x_scaled) + INTERCEPT
    prob      = 1 / (1 + np.exp(-log_odds))
    return int(prob >= 0.5), float(prob)

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="Diabetes Prediction App",
                   page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient's clinical measurements to predict diabetes risk.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    pregnancies    = st.number_input("Pregnancies",           min_value=0,   max_value=17,  value=1)
    glucose        = st.number_input("Glucose (mg/dL)",       min_value=0,   max_value=200, value=120)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0,   max_value=122, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)",   min_value=0,   max_value=99,  value=20)
with col2:
    insulin = st.number_input("Insulin (muU/ml)",             min_value=0,   max_value=846, value=80)
    bmi     = st.number_input("BMI",                          min_value=0.0, max_value=67.1, value=25.0)
    dpf     = st.number_input("Diabetes Pedigree Function",   min_value=0.0, max_value=2.5,  value=0.4)
    age     = st.number_input("Age (years)",                  min_value=1,   max_value=100, value=30)

st.divider()

if st.button("Predict", use_container_width=True, type="primary"):
    pred, prob = predict([pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age])
    if pred == 1:
        st.error(f"⚠️ **High Risk of Diabetes** — probability: **{prob:.1%}**")
    else:
        st.success(f"✅ **Low Risk of Diabetes** — probability: **{prob:.1%}**")
    st.progress(prob)

st.divider()
st.caption("Logistic Regression · Pima Indians Diabetes Dataset · Educational use only.")
