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
symbol = 'BTC_USDT'
available_capital = 10000  # Total capital available for trading
risk_per_trade = 0.02  # 2% of available capital

def get_market_data(days=30):
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    response = requests.get(BASE_URL + 'public/get-candlestick', params={'instrument_name': symbol, 'timeframe': '1D', 'start_time': start_time, 'end_time': end_time})
    candlestick_data = response.json()

    if 'result' in candlestick_data and 'data' in candlestick_data['result']:
        data = candlestick_data['result']['data']
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='ms', origin='unix', errors='coerce')
        df.rename(columns={'t': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
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
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    short_ema = df['Close'].ewm(span=12, adjust=False).mean()
    long_ema = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['TR'] = np.maximum((df['High'] - df['Low']), np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['Upper_Band'], df['Lower_Band'] = df['rolling_mean_20'] + (df['rolling_std_20'] * 2), df['rolling_mean_20'] - (df['rolling_std_20'] * 2)
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14).mean() / df['TR'].ewm(alpha=1/14).mean())
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14).mean() / df['TR'].ewm(alpha=1/14).mean())
    df['DX'] = 100 * np.abs((df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].ewm(alpha=1/14).mean()
    df['Adj Close'] = df['Close']
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    X_test = df.drop(columns=["Close", "Date"])
    df['Predictions'] = np.nan
    df.loc[df.index[-len(X_test):], 'Predictions'] = ridge_regression.predict(X_test)
    volume_threshold = df['Volume'].quantile(0.20)
    df['Signal'] = 0
    df['Take Profit'] = df['Close'] + df['ATR'] * 1.5
    df['Stop Loss'] = df['Close'] - df['ATR'] * 2

    # State to track if a signal has been reached to target or stop loss
    signal_active = False
    take_profit_level = None
    stop_loss_level = None

    for i in range(len(df)):
        if signal_active:
            if df['High'].iloc[i] >= take_profit_level or df['Low'].iloc[i] <= stop_loss_level:
                signal_active = False

        if not signal_active:
            if (df['Predictions'].iloc[i] > df['Close'].iloc[i] and 
                df['Close'].iloc[i] > df['SMA_10'].iloc[i] and 
                df['RSI'].iloc[i] < 70 and 
                df['Volume'].iloc[i] > volume_threshold and 
                df['Close'].iloc[i] < df['Upper_Band'].iloc[i] and 
                df['ADX'].iloc[i] > 20 and 
                df['MACD_Hist'].iloc[i] > 0):

                df.at[df.index[i], 'Signal'] = 1
                signal_active = True
                take_profit_level = df['Take Profit'].iloc[i]
                stop_loss_level = df['Stop Loss'].iloc[i]

    df['Amount to Buy'] = (available_capital * risk_per_trade) / (df['Close'] - df['Stop Loss'])
    df['Possible Profit %'] = ((df['Take Profit'] - df['Close']) / df['Close']) * 100
    return df

def display_signals(df):
    last_month = datetime.now() - timedelta(days=30)
    signals = df[(df['Signal'] == 1) & (df['Date'] >= last_month)]
    print("Generated Signals for Last Month:")
    signals_to_print = signals[['Date', 'Volume', 'Close', 'Take Profit', 'Stop Loss', 'Amount to Buy', 'Possible Profit %', 'Predictions']]
    print(signals_to_print)
    signals_to_print.to_csv('signals.csv', index=False)
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    buy_signals = df[df['Signal'] == 1]
    ax1.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='green', label='Buy Signal', alpha=1)
    
    # Annotate the signals with details
    for idx, row in buy_signals.iterrows():
        ax1.annotate(f"Date: {row['Date'].date()}\nClose: {row['Close']}\nTP: {row['Take Profit']}\nSL: {row['Stop Loss']}",
                     (row['Date'], row['Close']),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=8,
                     color='green',
                     arrowprops=dict(facecolor='green', shrink=0.05))
    
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['Predictions'], label='Predicted Price', linestyle='--', color='orange')
    ax2.set_ylabel('Predicted Price', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    fig.suptitle('Close Price with Buy Signals and Predictions')
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('signals_chart.png')
    plt.show()

def backtest_strategy(df):
    df['Position'] = df['Signal'].shift(1)
    df['Daily Return'] = df['Close'].pct_change()
    df['Strategy Return'] = df['Daily Return'] * df['Position']
    df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod()
    df['Cumulative Market Return'] = (1 + df['Daily Return']).cumprod()

    # Calculate performance metrics
    total_return = df['Cumulative Strategy Return'].iloc[-1] - 1
    sharpe_ratio = (df['Strategy Return'].mean() / df['Strategy Return'].std()) * np.sqrt(252)
    max_drawdown = (df['Cumulative Strategy Return'].cummax() - df['Cumulative Strategy Return']).max()

    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")

    # Plot the cumulative returns
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Cumulative Strategy Return'], label='Strategy Return', color='blue')
    plt.plot(df['Date'], df['Cumulative Market Return'], label='Market Return', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Strategy and Market Returns')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cumulative_returns_chart.png')
    plt.show()

def trading_bot():
    df = get_market_data(days=365)  # Increase the historical period for backtesting
    if df is not None:
        df = calculate_indicators(df)
        df = generate_signals(df)
        display_signals(df)
        backtest_strategy(df)  # Add backtesting step
    else:
        print("Failed to fetch market data. Skipping this iteration.")

if __name__ == "__main__":
    trading_bot()
