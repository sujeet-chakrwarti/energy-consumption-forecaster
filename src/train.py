import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Import our custom modules from the src/ folder
from src.data_pipeline import load_and_engineer_features
from src.model_network import DailyEnergyForecaster

print("--- Step 1: Loading & Preprocessing Data ---")
df, features_list = load_and_engineer_features('data/energy_2020_2024.csv')
print(f"Data ready. Shape: {df.shape}")

# Prepare Arrays
X = df[features_list].values
y = df['energy_consumption'].values.reshape(-1, 1)

# Time Series Split (80% Train, 20% Test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Data Standardization
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Convert to Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

print("--- Step 2: Training PyTorch Model ---")
# Initialize Model from our src module
model = DailyEnergyForecaster(input_dim=len(features_list))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train_tensor), y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("--- Step 3: Saving Model ---")
os.makedirs('models', exist_ok=True)
save_dict = {
    'model_state': model.state_dict(),
    'X_mean': X_mean,
    'X_std': X_std,
    'input_dim': len(features_list)
}
torch.save(save_dict, 'models/pytorch_daily_energy_model.pt')
print("✅ Model saved to models/pytorch_daily_energy_model.pt")