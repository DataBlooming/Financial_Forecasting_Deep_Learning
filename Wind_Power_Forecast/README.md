# Wind Power Forecast

**LSTM + SARIMA-based Wind Power Prediction API with Monitoring**

This project demonstrates a complete workflow for predicting wind power generation using historical data. The system includes model training, API deployment, and prediction monitoring.

## Features

- Predict future wind power using past 30 days of data
- Two modeling approaches: **LSTM** (Long Short-Term Memory) neural network and **SARIMA** (Seasonal AutoRegressive Integrated Moving Average)
- **FastAPI** endpoint for real-time predictions
- Logging of predictions with timestamp, input, predicted value, and model version
- Scaler (`wind_scaler.pkl`) ensures consistent preprocessing
- Ready for visualization and monitoring in Power BI or Python

## Folder Structure

Wind_Power_Forecast/
├─ notebooks/
│ └─ wind_forecast_lstm_and_sarima.ipynb # Model training and experiments
├─ api/
│ └─ api_forecast.py # FastAPI code
├─ models/
│ ├─ lstm_wind_forecast_model.keras # Trained LSTM model
│ └─ wind_scaler.pkl # Scaler for preprocessing
├─ logs/
│ └─ sample_prediction_log.csv # Example prediction logs
├─ README.md
└─ requirements.txt

## How to Use

1. Run the API:  
   uvicorn api.api_forecast:app --reload --host 0.0.0.0 --port 8000
   Access API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

2. Make Predictions by POSTing to `/predict` with JSON payload:  
   {
  "recent_days": [7700, 7720, 7750, ..., 8400]
   }
   Response includes predicted wind power and model version.

3. Logging: Each prediction is logged to `logs/prediction_log.csv` with Timestamp, Input data, Prediction, and Model version

## Requirements

Install dependencies:  
pip install tensorflow numpy joblib fastapi uvicorn pydantic
Or with `requirements.txt`:  


## Notes

- CSV logging allows monitoring predictions over time  
- Input data in the log is stored as a list of 30 past wind power values  
- API and models can be extended to include additional features or new prediction algorithms
