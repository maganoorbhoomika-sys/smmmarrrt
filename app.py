import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

@st.cache_resource
def load_model():
    df = pd.read_csv("diabetes.csv")
    for col in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = load_model()

st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
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
    bmi     = st.number_input("BMI",                          min_value=0.0, max_value=67.1,value=25.0)
    dpf     = st.number_input("Diabetes Pedigree Function",   min_value=0.0, max_value=2.5, value=0.4)
    age     = st.number_input("Age (years)",                  min_value=1,   max_value=100, value=30)

st.divider()

if st.button("Predict", use_container_width=True, type="primary"):
    input_df = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=['Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age']
    )
    scaled      = scaler.transform(input_df)
    prediction  = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes — probability: {probability:.1%}")
    else:
        st.success(f"✅ Low Risk of Diabetes — probability: {probability:.1%}")
    st.progress(float(probability))
