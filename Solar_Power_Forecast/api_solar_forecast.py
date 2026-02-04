import os
import csv
import joblib
import numpy as np
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel
from tensorflow.keras.models import load_model

# ===========================
# Load solar model and scaler
# ===========================
solar_model = load_model("lstm_solar_forecast_model.keras")
solar_scaler = joblib.load("solar_scaler.pkl")
MODEL_VERSION = "LSTM_SOLAR_v1_2026-02-04"
LOG_FILE = "solar_prediction_log.csv"

# ===========================
# Settings
# ===========================
TIMESTEPS = 60  

# ===========================
# Initialize log file
# ===========================
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + [f"day_{i+1}" for i in range(TIMESTEPS)]
                        + ["prediction", "model_version"])

# ===========================
# Define input format
# ===========================
class InputData(BaseModel):
    recent_days: list[float]

# ===========================
# Initialize FastAPI
# ===========================
app = FastAPI(title="Solar Forecast API")

# ===========================
# Prediction endpoint
# ===========================
@app.post("/predict_solar")
def predict_solar(input_data: InputData):
    # 1. to numpy
    recent_days = np.array(input_data.recent_days).reshape(-1, 1)
    
    # 2. scale
    recent_days_scaled = solar_scaler.transform(recent_days)
    
    # 3. reshape for LSTM
    X = recent_days_scaled.reshape(1, -1, 1)
    
    # 4. predict
    y_pred_scaled = solar_model.predict(X)
    
    # 5. inverse scale
    y_pred = solar_scaler.inverse_transform(y_pred_scaled)
    
    # 6. log
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            *input_data.recent_days,
            float(y_pred[0,0]),
            MODEL_VERSION
        ])
        f.flush()
    
    # 7. return
    return {"prediction": float(y_pred[0,0]), "model_version": MODEL_VERSION}

# ===========================
# Run server
# ===========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
