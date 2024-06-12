import os
import requests
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
from joblib import load

# Load environment variables
load_dotenv()

API_KEY = os.getenv('CRYPTO_API_KEY')
API_SECRET = os.getenv('CRYPTO_API_SECRET')

BASE_URL = "https://api.crypto.com/v2/"

# Load the trained Ridge Regression model
ridge_regression = load('ridge_regression_model.joblib')

# Define trading parameters
maker_fee = 0.0008  # 0.08%
taker_fee = 0.0007  # 0.07%
take_profit_threshold = 0.08  # 8% take-profit
volume_threshold = 100  # Define a threshold based on your historical volume analysis
symbol = 'BTC_USDT'
trade_quantity = 0.001  # Define the quantity you want to trade

def get_market_data():
    # Fetch ticker data
    response = requests.get(BASE_URL + 'public/get-ticker', params={'instrument_name': symbol})
    ticker_data = response.json()
    
    print("Ticker Data:", ticker_data)  # Debugging line
    
    if 'result' in ticker_data and 'data' in ticker_data['result']:
        data = ticker_data['result']['data'][0]
        price = float(data['a'])  # ask price
    else:
        print("Failed to fetch ticker data.")
        return None, None

    # Fetch candlestick data
    response = requests.get(BASE_URL + 'public/get-candlestick', params={'instrument_name': symbol, 'timeframe': '1D'})
    candlestick_data = response.json()
    
    print("Candlestick Data:", candlestick_data)  # Debugging line
    
    if 'result' in candlestick_data and 'data' in candlestick_data['result']:
        data = candlestick_data['result']['data']
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='ms', origin='unix', errors='coerce')
        df.rename(columns={'t': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        # Convert columns to numeric types
        df['Open'] = pd.to_numeric(df['Open'])
        df['High'] = pd.to_numeric(df['High'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['Close'] = pd.to_numeric(df['Close'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        print("Fetched market data")
        return df, price
    else:
        print("Failed to fetch candlestick data.")
        return None, None

def calculate_indicators(df):
    # Calculate technical indicators
    df['lag_1'] = df['Close'].shift(1)
    df['lag_2'] = df['Close'].shift(2)
    df['lag_3'] = df['Close'].shift(3)
    df['lag_4'] = df['Close'].shift(4)
    df['lag_5'] = df['Close'].shift(5)
    df['rolling_mean_5'] = df['Close'].rolling(window=5).mean()
    df['rolling_mean_10'] = df['Close'].rolling(window=10).mean()
    df['rolling_mean_20'] = df['Close'].rolling(window=20).mean()
    df['rolling_std_5'] = df['Close'].rolling(window=5).std()
    df['rolling_std_10'] = df['Close'].rolling(window=10).std()
    df['rolling_std_20'] = df['Close'].rolling(window=20).std()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    
    df['Adj Close'] = df['Close']  # Assuming 'Adj Close' is same as 'Close' for this context
    
    return df

def apply_trading_strategy(df):
    if df is not None:
        df = calculate_indicators(df)
        X_test = df.drop(columns=["Close", "Date"]).tail(300)
        df['Predictions'] = ridge_regression.predict(X_test)

        # Generate buy (1) signals based on model predictions
        df['Signal'] = 0
        df.loc[df['Predictions'] > df['Close'], 'Signal'] = 1  # Buy signal

        # Apply volume filter
        df.loc[df['Volume'] < volume_threshold, 'Signal'] = 0  # No trade if volume is below threshold

        print("Applied trading strategy")
        return df
    else:
        print("Dataframe is None. Skipping strategy application.")
        return df

def execute_trade(signal, price):
    if signal == 1:
        # Place a buy order (mock implementation)
        print(f"Executed trade: Buy {trade_quantity} of {symbol} at price {price}")
        # Here you should implement actual API call to create an order
    else:
        print("No trade executed")

def trading_bot():
    df, price = get_market_data()
    if df is not None and price is not None:
        df = apply_trading_strategy(df)
        signal = df['Signal'].iloc[-1]  # Get the latest signal
        execute_trade(signal, price)
    else:
        print("Failed to fetch market data. Skipping this iteration.")

if __name__ == "__main__":
    while True:
        try:
            trading_bot()
            time.sleep(60)  # Run the bot every minute
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)  # Wait for 1 minute before retrying
