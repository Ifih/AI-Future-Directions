# IoT Agriculture App for Yield Prediction

## Description
This IoT Agriculture application simulates sensor data from agricultural sensors (soil moisture, temperature, humidity, light intensity) and uses a Long Short-Term Memory (LSTM) neural network to predict crop yields. It demonstrates an end-to-end pipeline from data generation to model training, prediction, and visualization.

## Requirements
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

## Installation
1. Ensure Python 3.x is installed on your system.
2. Install required packages:
   ```
   pip install tensorflow numpy matplotlib
   ```

## Usage
Run the script directly:
```
python iot_agriculture_app.py
```

The script will:
1. Generate simulated sensor data for 1000 days.
2. Train an LSTM model for yield prediction.
3. Make predictions on sample data.
4. Simulate a farmer dashboard with data flow description.
5. Generate and save a plot comparing actual vs. predicted yields.

## Output
- Console output showing training progress, sample predictions, and dashboard simulation.
- Plot image: `yield_prediction.png` (comparison of actual and predicted yields)

## Notes
- The sensor data is synthetically generated for demonstration.
- Training uses 20 epochs; adjust as needed for better performance.
- In a real deployment, connect to actual IoT sensors for live data.
