import streamlit as st
import pickle
import numpy as np

# Load trained model
import os
import pickle

model_path = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
model = pickle.load(open(model_path, "rb"))

# Title
st.set_page_config(page_title="Crop Prediction", layout="centered")
st.title("🌱 Crop Prediction App")
st.write("Enter soil and environmental details to predict the best crop.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0)
    K = st.number_input("Potassium (K)", min_value=0.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    ph = st.number_input("pH value", min_value=0.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# Prediction
if st.button("Predict Crop"):
    try:
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = model.predict(features)

        st.success(f"🌾 Recommended Crop: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")
