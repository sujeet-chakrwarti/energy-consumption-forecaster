import pandas as pd
import numpy as np

def load_and_engineer_features(file_path):
    """Loads raw data, aggregates it, and engineers time/weather features."""
    
    # 1. Load and aggregate
    df_raw = pd.read_csv(file_path)
    df_raw['date'] = pd.to_datetime(df_raw['date'])
    
    df = df_raw.groupby('date').agg({
        'energy_consumption': 'sum',
        'avg_temperature': 'mean',
        'humidity': 'mean'
    }).reset_index()
    
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    # 2. Feature Engineering
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    df['energy_lag_1'] = df['energy_consumption'].shift(1)
    df['energy_lag_7'] = df['energy_consumption'].shift(7)
    
    df['heating_demand'] = np.maximum(0, 18 - df['avg_temperature'])
    df['cooling_demand'] = np.maximum(0, df['avg_temperature'] - 18)
    
    df.dropna(inplace=True) 
    
    # 3. Define features list
    features = ['avg_temperature', 'humidity', 'month', 'day', 'weekday', 
                'is_weekend', 'energy_lag_1', 'energy_lag_7', 'heating_demand', 'cooling_demand']
                
    return df, features