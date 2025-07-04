import yfinance as yf
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 1. Download stock data
def download_stock_data(symbol, period="1y", interval="1d"):
    data = yf.download(symbol, period=period, interval=interval)
    if data.empty:
        raise ValueError("No stock data found. Check ticker symbol or intervall.")
    data = data[['Close']]
    return data

# 2. Prepare features + labels for regression

def create_windowed_data(data, window_size=20):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# 3. Train regression model
def train_regressor(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 4. Predict future price and infer trend

def predict_and_classify(model, X_test, last_real_price, threshold=0.002):
    pred_price = model.predict(X_test[-1].reshape(1, -1))[0]
    pct_change = (pred_price - last_real_price) / last_real_price

    if pct_change > threshold:
        trend = "Up"
    elif pct_change < -threshold:
        trend = "Down"
    else:
        trend = "Neutral"
    return pred_price, trend


symbol = "TSLA"
data = download_stock_data(symbol, period="1y", interval="1d")

# Normalize prices for ML training
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['Close']])

X, y = create_windowed_data(data_scaled, window_size=20)

# Split into train/test
split = int(len(X)*0.95)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])
model = train_regressor(X_train, y_train)

# Reverse scaling
last_real_price = scaler.inverse_transform(y[-1].reshape(1, -1))[0][0]
pred_scaled, trend = predict_and_classify(model, X_test, y[-1])
pred_price = scaler.inverse_transform(np.array(pred_scaled).reshape(1, -1))[0][0]

print(f"\nPredicted closing price: ${pred_price:.2f}")
print(f"Trend direction: {trend}")

# Plot
plt.figure(figsize=(14, 6))
true_prices = scaler.inverse_transform(y.reshape(-1, 1))
X_flat = X.reshape(X.shape[0], X.shape[1])  # Flatten from 3D to 2D
pred_prices = scaler.inverse_transform(model.predict(X_flat).reshape(-1, 1))
plt.plot(true_prices, label='True Prices')
plt.plot(pred_prices, label='Predicted Prices')
plt.axvline(split, color='red', linestyle='--', label='Train/Test Split')
plt.title(f"{symbol} Closing Price Prediction with Trend Classification")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
