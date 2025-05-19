import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Load CSV and Prepare Data
df = pd.read_csv("glucose_data.csv")

# Separate Timestamp Column
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp',inplace=True)
elif 'time' in df.columns:
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time',inplace=True)
else:
    raise ValueError("CSV must contain a 'timestamp' or 'time' column.")

# Rename glucose_level to glucose
df.rename(columns={'glucose_level': 'glucose'}, inplace=True)

# Feature Engineering
# Calculate Rate of Change
df['delta'] = df['glucose'].diff().fillna(0)

# Scale glucose and data
scaler = StandardScaler()
df[['glucose_scaled', 'delta_scaled']] = scaler.fit_transform(df[['glucose','delta']])

# Anomaly Detection using Isolation Forest
model = IsolationForest(contamination=0.3, random_state=42)
df['anomaly'] = model.fit_predict(df[['glucose_scaled', 'delta_scaled']])

# Meal Detection using Peak Analysis
peaks, _ = find_peaks(df['glucose'], prominence=15, distance=20)
df['meal_detected'] = 0
df.iloc[peaks, df.columns.get_loc('meal_detected')] = 1

# Visualization
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['glucose'], label='Glucose')
plt.scatter(df.index[df['meal_detected'] == 1], df['glucose'][df['meal_detected'] == 1],
            color='orange', label='Meal Detected', zorder=5)
plt.scatter(df.index[df['anomaly'] == -1], df['glucose'][df['anomaly'] == -1],
            color='red', marker='x', label='Anomaly', zorder=5)
plt.title("Meal Detection and Anomaly Analysis from CGM Data")
plt.xlabel("Time")
plt.ylabel("Glucose Level (mg/dL)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

