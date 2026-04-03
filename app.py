import streamlit as st
import numpy as np
import torch
import datetime

# Clean import from our modularized code!
from src.model_network import DailyEnergyForecaster

@st.cache_resource
def load_pytorch_model():
    checkpoint = torch.load('models/pytorch_daily_energy_model.pt', weights_only=False)
    model = DailyEnergyForecaster(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval() 
    return model, checkpoint['X_mean'], checkpoint['X_std']

# 👉 YOU WERE MISSING THIS LINE! It actually loads the model and scalers.
model, X_mean, X_std = load_pytorch_model()

st.title("🌍 Global Daily Energy Consumption Forecaster")
st.write("Predict future daily energy demand based on global weather averages.")

# Sidebar for User Inputs
st.sidebar.header("Input Conditions")
date = st.sidebar.date_input("Select Date for Prediction", datetime.date.today())

st.sidebar.subheader("Weather Forecast")
avg_temp = st.sidebar.slider("Average Global Temp (°C)", -10.0, 40.0, 20.0)
humidity = st.sidebar.slider("Average Humidity (%)", 0, 100, 50)

st.sidebar.subheader("Historical Context")
lag_1 = st.sidebar.number_input("Energy Consumed Yesterday", value=50000.0)
lag_7 = st.sidebar.number_input("Energy Consumed 1 Week Ago", value=49500.0)

if st.button("Predict Daily Energy Consumption"):
    # Calculate features
    heating_demand = max(0, 18 - avg_temp)
    cooling_demand = max(0, avg_temp - 18)
    
    # [avg_temperature, humidity, month, day, weekday, is_weekend, lag_1, lag_7, heating_demand, cooling_demand]
    raw_features = np.array([[
        avg_temp, 
        humidity, 
        date.month, 
        date.day, 
        date.weekday(), 
        1 if date.weekday() in [5, 6] else 0, 
        lag_1, 
        lag_7, 
        heating_demand, 
        cooling_demand
    ]])

    scaled_features = (raw_features - X_mean) / X_std
    
    # Convert to Tensor and Predict
    tensor_features = torch.tensor(scaled_features, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(tensor_features).item()
    
    st.success(f"### Predicted Total Global Demand: {prediction:,.2f} Units")