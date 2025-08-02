# Financial Forecasting with Deep Learning - LSTM Models

This repository contains two deep learning projects focused on financial time series forecasting using Long Short-Term Memory (LSTM) networks:

- Apple Stock Price Prediction (`apple_stock_price_prediction_lstm/`)  
- Bitcoin Price Prediction (`BTCp_price_prediction_lstm/`)

Both projects aim to predict future prices based on historical data, leveraging the strengths of LSTM in capturing temporal dependencies in sequential data.

---

## Project Overview

### 1. Apple stock price prediction   
This project uses historical Apple Inc. stock price data to train an LSTM model for price forecasting. The model incorporates features such as opening price, closing price, volume, and technical indicators to improve prediction accuracy.

### 2. Bitcoin Price Prediction  
This project applies a similar LSTM architecture to predict Bitcoin prices, using historical price and trading volume data. Due to the high volatility of cryptocurrency markets, the model includes advanced preprocessing and tuning to handle noise and non-stationarity.

---

## Common Features

- Data preprocessing including scaling and windowing of time series data  
- LSTM network architecture for sequence modeling  
- Model training with appropriate loss functions and optimizers  
- Performance evaluation using metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)  
- Visualization of predicted vs. actual price trends  

---

## Folder Structure
/Financial_Forecasting_Deep_Learning <br>
│ <br>
├── apple_stock_price_prediction_lstm <br>
└── BTCp_price_prediction_lstm 

## Requirements

- Python 3.9+  
- Common libraries: `numpy`, `pandas`, `matplotlib`, `tensorflow`, `scikit-learn`

Please ensure these libraries are installed before running the code.

---

## Notes

- Both models can be further fine-tuned for longer forecasting horizons or integrated with additional features.  
- Due to the volatile nature of financial data, especially cryptocurrencies, predictions should be used cautiously and in conjunction with other analysis methods.
