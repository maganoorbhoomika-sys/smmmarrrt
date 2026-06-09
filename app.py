import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("Diabetes Prediction App")
st.write("Enter the patient's information to predict whether they have diabetes.")

# Input fields for patient data

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1)
glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=122, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20)
insulin = st.number_input("Insulin (muU/ml)", min_value=0, max_value=846, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=25.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.4)
age = st.number_input("Age (years)", min_value=1, max_value=100, value=30)


# Create a DataFrame from the input
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                          columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

# Preprocess the input data
input_data_scaled = scaler.transform(input_data)

if st.button("Predict Diabetes"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)[:, 1]

    if prediction[0] == 1:
        st.error(f"The model predicts that the patient has Diabetes with a probability of {prediction_proba[0]:.2f}.")
    else:
        st.success(f"The model predicts that the patient does NOT have Diabetes with a probability of {prediction_proba[0]:.2f}.")

st.markdown("""
--- 
**How to run this app locally:**
1. Save the model and scaler files (`logistic_regression_model.pkl` and `scaler.pkl`) along with `app.py` and `requirements.txt` in the same directory.
2. Open your terminal or command prompt.
3. Navigate to that directory.
4. Run `pip install -r requirements.txt` to install dependencies.
5. Run `streamlit run app.py`.
""")