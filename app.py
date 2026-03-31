import streamlit as st
import numpy as np
import joblib

# Load saved files
model = joblib.load("Manufacturing_Quality_prediction.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Quality Prediction", layout="centered")

st.title("🏭 Manufacturing Quality Prediction")

st.write("Fill in the feature values:")

# Dynamic inputs
input_data = []

for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

# Prediction
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)

    st.success(f"Prediction Result: {prediction[0]}")
