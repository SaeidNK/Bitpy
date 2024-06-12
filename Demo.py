import os
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from joblib import load
import matplotlib.pyplot as plt

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
risk_per_trade = 0.02  # 2% of available capital
available_capital = 10000  # Total capital available for trading

def get_market_data(days=30):
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    # Fetch candlestick data for the last `days` days
    response = requests.get(BASE_URL + 'public/get-candlestick', params={'instrument_name': symbol, 'timeframe': '1D', 'start_time': start_time, 'end_time': end_time})
    candlestick_data = response.json()

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
        return df
    else:
        print("Failed to fetch candlestick data.")
        return None

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
    
    # Calculate ATR (Average True Range) for dynamic stop loss and take profit levels
    df['TR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # Calculate Bollinger Bands
    df['Upper Band'] = df['rolling_mean_20'] + (df['rolling_std_20'] * 2)
    df['Lower Band'] = df['rolling_mean_20'] - (df['rolling_std_20'] * 2)

    # Calculate ADX (Average Directional Index)
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
    df['TR'] = np.max([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1))], axis=0)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14).mean() / df['TR'].ewm(alpha=1/14).mean())
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14).mean() / df['TR'].ewm(alpha=1/14).mean())
    df['DX'] = 100 * np.abs((df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].ewm(alpha=1/14).mean()

    return df

def apply_trading_strategy(df):
    if df is not None:
        df = calculate_indicators(df)
        X_test = df.drop(columns=["Close", "Date"]).tail(300)
        df['Predictions'] = ridge_regression.predict(X_test)

        # Calculate volume threshold before using it
        volume_threshold = df['Volume'].quantile(0.20)  # Using 20th percentile as threshold

        # Generate buy (1) signals based on model predictions and Bollinger Bands confirmation
        df['Signal'] = 0
        df.loc[(df['Predictions'] > df['Close']) & (df['Close'] > df['SMA_10']) & (df['Close'] < df['Upper Band']) & (df['ADX'] > 20) & (df['RSI'] < 70) & (df['Volume'] > volume_threshold), 'Signal'] = 1  # Added RSI and volume filter

        # Apply a smoothing filter to reduce noise (e.g., exponential moving average on the signals)
        df['Signal'] = df['Signal'].ewm(span=3, adjust=False).mean().round()  # Smoothing the signal

        # Increase the stop loss multiplier
        stop_loss_multiplier = 3  # Increased stop loss multiplier to 3 times the ATR
        take_profit_multiplier = 2  # Adjust take profit multiplier if needed

        # Calculate dynamic take profit and stop loss levels using ATR
        df['Take Profit'] = df['Close'] + df['ATR'] * take_profit_multiplier
        df['Stop Loss'] = df['Close'] - df['ATR'] * stop_loss_multiplier

        # Calculate the amount to buy
        df['Amount to Buy'] = (available_capital * risk_per_trade) / (df['Close'] - df['Stop Loss'])

        # Calculate the possible profit percentage
        df['Possible Profit %'] = ((df['Take Profit'] - df['Close']) / df['Close']) * 100

        # Implement a volatility filter to avoid trading during high volatility periods
        volatility_threshold = df['ATR'].mean() * 1.5  # Example threshold; adjust as needed
        df.loc[df['ATR'] > volatility_threshold, 'Signal'] = 0

        # Identify correct and false buy signals
        df['Correct Signal'] = np.nan
        for i in range(len(df)):
            if df.loc[i, 'Signal'] == 1:
                for j in range(i + 1, len(df)):
                    if df.loc[j, 'High'] >= df.loc[i, 'Take Profit']:
                        df.loc[i, 'Correct Signal'] = True
                        break
                    if df.loc[j, 'Low'] <= df.loc[i, 'Stop Loss']:
                        df.loc[i, 'Correct Signal'] = False
                        break

        print("Applied trading strategy")
        return df
    else:
        print("Dataframe is None. Skipping strategy application.")
        return df

def display_signals(df):
    if df is not None:
        signals = df[df['Signal'] == 1]
        print("Generated Signals for Last Month:")
        signals_to_print = signals[['Date', 'Volume', 'Close', 'Take Profit', 'Stop Loss', 'Amount to Buy', 'Possible Profit %', 'Correct Signal']]
        print(signals_to_print)
        
        # Save the signals to a CSV file
        signals_to_print.to_csv('signals.csv', index=False)
        
        live_price = get_live_price()
        print(f"Live Price: {live_price}")

        # Plotting the signals on a chart
        plt.figure(figsize=(14, 7))
        plt.plot(df['Date'], df['Close'], label='Close Price')
        
        # Plot correct buy signals in green and false buy signals in red
        correct_signals = df[df['Correct Signal'] == True]
        false_signals = df[df['Correct Signal'] == False]
        
        plt.scatter(correct_signals['Date'], correct_signals['Close'], marker='^', color='g', label='Correct Buy Signal', alpha=1)
        plt.scatter(false_signals['Date'], false_signals['Close'], marker='v', color='r', label='False Buy Signal', alpha=1)
        
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Close Price with Buy Signals')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('signals_chart.png')  # Save the chart as a PNG file
        plt.show()
    else:
        print("No data to display signals.")

def get_live_price():
    response = requests.get(BASE_URL + 'public/get-ticker', params={'instrument_name': symbol})
    ticker_data = response.json()

    if 'result' in ticker_data and 'data' in ticker_data['result']:
        data = ticker_data['result']['data'][0]
        return float(data['a'])  # ask price
    else:
        print("Failed to fetch ticker data.")
        return None

def trading_bot():
    df = get_market_data(days=30)
    if df is not None:
        df = apply_trading_strategy(df)
        display_signals(df)
    else:
        print("Failed to fetch market data. Skipping this iteration.")

    while True:
        try:
            live_price = get_live_price()
            if live_price is not None:
                print(f"Live Price: {live_price}")
                # Update with new data for live trading signal generation if needed
            time.sleep(60)  # Check live price every minute
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)  # Wait for 1 minute before retrying

if __name__ == "__main__":
    trading_bot()
