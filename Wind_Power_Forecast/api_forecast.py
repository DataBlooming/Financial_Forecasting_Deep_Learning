import os
import csv
import joblib
import numpy as np
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# Load trained model and scaler
model = load_model("lstm_wind_forecast_model.keras")
scaler = joblib.load("wind_scaler.pkl")

# Define model version
MODEL_VERSION = "LSTM_v1_2026-02-01" 

# Define input format
class InputData(BaseModel):
    recent_days: list[float]
    
# Create log file
LOG_FILE = "prediction_log.csv"
TIMESTEPS = 30  # length of input
# Write header if file doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        # header: timestamp, day_1 ... day_30, prediction, model_version
        writer.writerow(["timestamp"] + [f"day_{i+1}" for i in range(TIMESTEPS)] + ["prediction", "model_version"])

        
app = FastAPI(title="Wind Forecast API")
@app.post("/predict")
def predict(input_data: InputData):
    # 1. raw → numpy
    recent_days = np.array(input_data.recent_days).reshape(-1, 1)
    
    # 2. scale 
    recent_days_scaled = scaler.transform(recent_days)
    
    # 3. reshape for LSTM
    X = recent_days_scaled.reshape(1, -1, 1)
    
    # 4. predict (still scaled)
    y_pred_scaled = model.predict(X)

   # 5. inverse scale → real world value
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # 6. log to CSV
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        # Write one row per prediction: timestamp, input, prediction value, model version
        writer.writerow([
            datetime.now().isoformat(),
            *input_data.recent_days,  # expanding the list
            float(y_pred[0,0]),
            MODEL_VERSION
        ])

    # 7. Return prediction + model version
    return {
        "prediction": float(y_pred[0, 0]),
        "model_version": MODEL_VERSION
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)