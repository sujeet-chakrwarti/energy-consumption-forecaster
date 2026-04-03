# 🌍 Global Daily Energy Consumption Forecaster

## Overview
This project is an end-to-end Deep Learning pipeline designed to forecast global daily energy consumption. It uses historical energy usage and global weather averages (temperature and humidity) to predict future demand. 

The project features a custom Feedforward Neural Network built with **PyTorch** and an interactive web dashboard built with **Streamlit**.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Deep Learning Framework:** PyTorch
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Web Deployment:** Streamlit

## 📂 Project Structure
```text
├── data/
│   └── energy_2020_2024.csv           # Raw dataset (Aggregated daily)
├── models/
│   └── pytorch_daily_energy_model.pt  # Saved PyTorch model weights & scalers
├── src/
│   └── model_network.py
│   └── train.py                       # Script to preprocess data, train, and save the PyTorch model
│   └── data_pipeline.py                    
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
