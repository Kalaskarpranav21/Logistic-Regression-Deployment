import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("Diabetes Prediction System")
st.write("Predict whether a patient is diabetic using Logistic Regression")

features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

inputs = []
for feature in features:
    value = st.number_input(feature, min_value=0.0)
    inputs.append(value)

if st.button("Predict"):
    scaled_data = scaler.transform([inputs])
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    if prediction == 1:
        st.error(f"Prediction: Diabetic (Probability: {probability:.2f})")
    else:
        st.success(f"Prediction: Non-Diabetic (Probability: {1-probability:.2f})")
