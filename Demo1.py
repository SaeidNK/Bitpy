import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    rolling_mean = data.rolling(window).mean()
    rolling_std = data.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

# Function to calculate Average True Range (ATR)
def calculate_atr(high, low, close, window=14):
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - close.shift()))
    tr3 = pd.Series(abs(low - close.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

# Load the preprocessed data
data = pd.read_csv("preprocessed_Bitdata.csv")

# Convert the Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Filter data for the last 3 months
end_date = data['Date'].max()
start_date = end_date - timedelta(days=90)
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Calculate additional technical indicators
filtered_data.loc[:, 'Upper_Band'], filtered_data.loc[:, 'Lower_Band'] = calculate_bollinger_bands(filtered_data['Close'])
filtered_data.loc[:, 'ATR'] = calculate_atr(filtered_data['High'], filtered_data['Low'], filtered_data['Close'])

# Separate features and date for the last 3 months
X = filtered_data.drop(columns=["Close", "Date"])  # Features excluding the target variable and Date
dates = filtered_data["Date"]
actual_prices = filtered_data["Close"]

# Load the trained Ridge Regression model
model = load('ridge_regression_model.joblib')

# Make predictions
predicted_prices = model.predict(X)

# Generate trading signals with enhanced logic
def generate_signals(actual, predicted, data):
    signals = pd.DataFrame(index=actual.index)
    signals['Actual'] = actual
    signals['Predicted'] = predicted
    signals['Signal'] = 0

    # Bollinger Band logic
    signals.loc[actual < data['Lower_Band'], 'Signal'] = 1  # Buy signal when price is below the lower Bollinger Band
    signals.loc[actual > data['Upper_Band'], 'Signal'] = -1  # Sell signal when price is above the upper Bollinger Band

    # ATR logic for sell signals (consider significant drops)
    atr_threshold = data['ATR'] * 0.5
    significant_drop = (actual - predicted) > atr_threshold
    signals.loc[significant_drop, 'Signal'] = -1  # Sell signal when predicted drop is significant

    # Combined logic (example: confirmation by both Bollinger Bands and ATR)
    signals['Combined_Signal'] = 0
    buy_signals = (signals['Signal'] == 1)
    sell_signals = (signals['Signal'] == -1)
    signals.loc[buy_signals, 'Combined_Signal'] = 1
    signals.loc[sell_signals, 'Combined_Signal'] = -1

    return signals

signals = generate_signals(actual_prices, predicted_prices, filtered_data)

# Plot the actual vs predicted prices and signals
plt.figure(figsize=(14, 7))
plt.plot(dates, actual_prices, label='Actual Prices')
plt.plot(dates, predicted_prices, label='Predicted Prices', alpha=0.7)

# Plot buy signals
buy_signals = signals[signals['Combined_Signal'] == 1]
plt.plot(buy_signals.index, actual_prices[buy_signals.index], '^', markersize=10, color='g', label='Buy Signal')

# Plot sell signals
sell_signals = signals[signals['Combined_Signal'] == -1]
plt.plot(sell_signals.index, actual_prices[sell_signals.index], 'v', markersize=10, color='r', label='Sell Signal')

plt.title('Bitcoin Price Prediction and Trading Signals (Last 3 Months)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()

# Formatting x-axis to show dates correctly and limit to last 3 months
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=7))
plt.gcf().autofmt_xdate()
plt.xlim(start_date, end_date)

plt.show()

