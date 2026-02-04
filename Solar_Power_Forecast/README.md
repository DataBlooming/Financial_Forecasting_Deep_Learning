## Solar Power Forecast

This project provides solar power forecasting for Germany using historical data from **Open Power System Data**. The dataset covers the period from **2015 to September 2020**.

## Data

- Source: [Open Power System Data](https://data.open-power-system-data.org/time_series/)
- Region: Germany
- Data type: Daily solar power generation
- Period: 2015 – 2020-09

## Models

Two different models are used for forecasting:

### 1. LSTM (Long Short-Term Memory)
- Captures **non-linear patterns** and **long-term dependencies** in time series.
- Can automatically learn seasonal and trend components from raw data.
- Input sequence length (`TIMESTEPS`) = 60 days.
- Provides more flexibility for complex patterns in solar generation.

### 2. SARIMA (Seasonal ARIMA)
- Captures **linear trends** and **seasonal cycles** explicitly.
- Works well for strongly **seasonal time series** like solar generation.
- Advantage: interpretable seasonal parameters and trend components.

> Note: Due to limited computer memory, SARIMA was not able to find the absolute optimal model. However, the **search methodology is correct**, and the model can still capture basic seasonality and trends.

## Project Structure

 ``` 
Solar_Power_Forecast/
├─ models/ # Trained LSTM model and scaler
│ ├─ solar_forecast_lstm_and_sarima.ipynb
│ ├─ lstm_solar_forecast_model.keras
│ └─ solar_scaler.pkl
├─ api/ # FastAPI code for predictions
│ └─ api_solar_forecast.py
├─ prediction_log/ # Sample input data for testing
│ └─ solar_prediction_log.xls
└─ README.md
 ``` 

## How to Run API

1. Install dependencies:

```bash
pip install fastapi uvicorn tensorflow joblib numpy pydantic

2. Run the API:
python api/api_solar_forecast.py

3. Access the prediction endpoint:
POST http://127.0.0.1:8001/predict_solar

4. Input format: (for example)
{
  "recent_days": [5000, 5020, 4980, 5010, 5025, 4995, 5000, 5015, 4990, 5005,
                  5012, 4998, 5003, 5018, 4989, 5001, 5010, 4997, 5020, 5004,
                  5013, 4992, 5009, 5021, 4996, 5006, 5017, 4988, 5022, 5000,
                  5011, 4994, 5007, 5024, 4986, 5003, 5014, 4991, 5026, 5005,
                  5016, 4993, 5002, 5028, 4987, 5009, 5015, 4999, 5020, 5001,
                  5012, 4995, 5008, 5023, 4989, 5004, 5011, 4996, 5025, 5003]
}

5. Output example
{
  "prediction": 5025.3,
  "model_version": "LSTM_SOLAR_v1_2026-02-04"
}
 ``` 

## Notes

TIMESTEPS = 60
Logs are saved in solar_prediction_log.csv
Ensure model and scaler files are in the models/ folder
The API uses LSTM for real-time prediction.


