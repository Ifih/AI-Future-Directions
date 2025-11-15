import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Simulate sensor data: soil_moisture, temperature, humidity, light_intensity
def generate_sensor_data(days=100):
    np.random.seed(42)
    time = np.arange(days)
    soil_moisture = 50 + 20 * np.sin(2 * np.pi * time / 30) + np.random.normal(0, 5, days)
    temperature = 20 + 10 * np.sin(2 * np.pi * time / 365) + np.random.normal(0, 2, days)
    humidity = 60 + 15 * np.cos(2 * np.pi * time / 30) + np.random.normal(0, 3, days)
    light_intensity = 500 + 200 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 50, days)
    yield_data = 100 + 0.5 * soil_moisture + 0.3 * temperature - 0.2 * humidity + 0.1 * light_intensity + np.random.normal(0, 10, days)
    return np.column_stack([soil_moisture, temperature, humidity, light_intensity]), yield_data

# Generate data
X, y = generate_sensor_data(1000)
X = X.reshape((X.shape[0], 1, X.shape[1]))  # For LSTM: (samples, timesteps, features)

# Build LSTM model
model = keras.Sequential([
    keras.layers.LSTM(50, activation='relu', input_shape=(1, 4)),
    keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train
print("Training LSTM model for yield prediction...")
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Predict
predictions = model.predict(X[:10])  # Predict first 10 days
print("Sample Predictions:")
for i, pred in enumerate(predictions):
    print(f"Day {i+1}: Predicted Yield {pred[0]:.2f}, Actual {y[i]:.2f}")

# Simulate dashboard
print("\nFarmer Dashboard Simulation:")
print("Sensors: Soil Moisture, Temperature, Humidity, Light Intensity")
print("AI Model: LSTM for time-series yield prediction")
print("Data Flow: Sensors -> Preprocessing -> LSTM -> Prediction -> Dashboard")
print("Efficiency Improvement: Optimizes irrigation, predicts yields, reduces waste.")

# Plot sample
plt.figure(figsize=(10,5))
plt.plot(y[:50], label='Actual Yield')
plt.plot(model.predict(X[:50]), label='Predicted Yield')
plt.legend()
plt.title('Yield Prediction Simulation')
plt.savefig('yield_prediction.png')
print("Plot saved as 'yield_prediction.png'")
