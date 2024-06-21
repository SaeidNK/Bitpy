import os
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from joblib import load
import matplotlib.pyplot as plt
import subprocess

# Load environment variables
load_dotenv()

API_KEY = os.getenv('CRYPTO_API_KEY')
API_SECRET = os.getenv('CRYPTO_API_SECRET')

BASE_URL = "https://api.crypto.com/v2/"

# Load the trained Ridge Regression model
result = subprocess.run(['python', 'Bitprep.py'], capture_output=True, text=True)
ridge_regression = load('ridge_regression_model.joblib')

# Define trading parameters
maker_fee = 0.0008  # 0.08%
taker_fee = 0.0007  # 0.07%
take_profit_threshold = 0.08  # 8% take-profit
volume_threshold = 100  # Define a threshold based on your historical volume analysis
symbol = 'BTC_USDT'
risk_per_trade = 0.02  # 2% of available capital
available_capital = 10000  # Total capital available for trading

stop_loss_multiplier = 2
take_profit_multiplier = 1

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
    
    df['Adj Close'] = df['Close']
    
    df['TR'] = np.maximum((df['High'] - df['Low']), np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=14).mean()

    df['Upper Band'] = df['rolling_mean_20'] + (df['rolling_std_20'] * 2)
    df['Lower Band'] = df['rolling_mean_20'] - (df['rolling_std_20'] * 2)

    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), df['High'] - df['High'].shift(1), 0)
    df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), df['Low'].shift(1) - df['Low'], 0)
    df['TR'] = np.maximum((df['High'] - df['Low']), np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    df['+DI'] = 100 * (df['+DM'].ewm(alpha=1/14).mean() / df['TR'].ewm(alpha=1/14).mean())
    df['-DI'] = 100 * (df['-DM'].ewm(alpha=1/14).mean() / df['TR'].ewm(alpha=1/14).mean())
    df['DX'] = 100 * np.abs((df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].ewm(alpha=1/14).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    return df

def apply_trading_strategy(df):
    if df is not None:
        df = calculate_indicators(df)
        X_test = df.drop(columns=["Close", "Date"]).tail(300)
        df['Predictions'] = np.nan
        df.loc[df.index[-300:], 'Predictions'] = ridge_regression.predict(X_test)

        volume_threshold = df['Volume'].quantile(0.20)

        df['Signal'] = 0
        df.loc[(df['Predictions'] > df['Close']) & 
               (df['Close'] > df['SMA_10']) & 
               (df['Close'] < df['Upper Band']) & 
               (df['ADX'] > 20) & 
               (df['RSI'] < 70) & 
               (df['Volume'] > volume_threshold), 'Signal'] = 1

        df['Signal'] = df['Signal'].ewm(span=3, adjust=False).mean().round()
        
        df['Take Profit'] = df['Close'] + df['ATR'] * take_profit_multiplier
        df['Stop Loss'] = df['Close'] - df['ATR'] * stop_loss_multiplier

        df['Amount to Buy'] = (available_capital * risk_per_trade) / (df['Close'] - df['Stop Loss'])

        df['Possible Profit %'] = ((df['Take Profit'] - df['Close']) / df['Close']) * 100

        volatility_threshold = df['ATR'].mean() * 1.5
        df.loc[df['ATR'] > volatility_threshold, 'Signal'] = 0

        df['Correct Signal'] = np.nan
        df['Current Signal'] = 0
        for i in range(len(df)):
            if df.loc[i, 'Signal'] == 1:
                for j in range(i + 1, len(df)):
                    if df.loc[j, 'High'] >= df.loc[i, 'Take Profit']:
                        df.loc[i, 'Correct Signal'] = 1
                        break
                    if df.loc[j, 'Low'] <= df.loc[i, 'Stop Loss']:
                        df.loc[i, 'Correct Signal'] = 0
                        break
                if pd.isna(df.loc[i, 'Correct Signal']):
                    df.loc[i, 'Current Signal'] = 1

        # Mark the most recent signal as current
        if df['Current Signal'].sum() > 0:
            df.loc[df[df['Current Signal'] == 1].index[-1], 'Current Signal'] = 1
            df.loc[df.index[:-1], 'Current Signal'] = 0

        # Save the signals to a CSV file
        signals = df[df['Signal'] == 1]
        signals_to_save = signals[['Date', 'Volume', 'Close', 'Take Profit', 'Stop Loss', 'Amount to Buy', 'Possible Profit %', 'Correct Signal']]
        signals_to_save.to_csv('signals.csv', index=False)

        print("Applied trading strategy and saved signals to signals.csv")
        return df
    else:
        print("Dataframe is None. Skipping strategy application.")
        return df

def evaluate_trading_bot(df):
    if df is not None:
        # Filter the DataFrame for signals
        signals = df[df['Signal'] == 1]
        
        # Calculate total and average profit/loss
        total_profit = signals['Take Profit'].sum() - signals['Stop Loss'].sum()
        average_profit = total_profit / len(signals) if len(signals) > 0 else 0
        
        # Calculate win rate
        win_rate = signals['Correct Signal'].mean()
        
        # Calculate maximum drawdown
        rolling_max = df['Close'].cummax()
        drawdown = (df['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Print evaluation metrics
        print(f"Total Profit: {total_profit:.2f}")
        print(f"Average Profit: {average_profit:.2f}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
    else:
        print("No data to evaluate.")

def plot_trend(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    plt.plot(df['Date'], df['SMA_10'], label='10-day SMA', color='orange')
    plt.plot(df['Date'], df['SMA_20'], label='20-day SMA', color='green')
    plt.plot(df['Date'], df['EMA_10'], label='10-day EMA', color='red')
    plt.plot(df['Date'], df['EMA_20'], label='20-day EMA', color='purple')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{symbol} Price Trend')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def display_signals(df, live_price=None):
    if df is not None:
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue', linewidth=1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Close Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Add horizontal lines
        min_price = df['Close'].min()
        max_price = df['Close'].max()
        price_increment = 5000
        start_price = int(min_price - (min_price % price_increment)) if min_price % price_increment != 0 else int(min_price)
        
        for price in range(start_price, int(max_price) + price_increment, price_increment):
            ax1.axhline(y=price, color='gray', linestyle='--', linewidth=0.5)
        
        correct_signals = df[df['Correct Signal'] == 1]
        false_signals = df[df['Correct Signal'] == 0]
        current_signals = df[df['Current Signal'] == 1]
        
        ax1.scatter(correct_signals['Date'], correct_signals['Close'], marker='^', color='green', label='Correct Buy Signal', alpha=1)
        ax1.scatter(false_signals['Date'], false_signals['Close'], marker='v', color='red', label='False Buy Signal', alpha=1)
        ax1.scatter(current_signals['Date'], current_signals['Close'], marker='o', color='orange', label='Current Signal', alpha=1)
        
        plt.plot(df['Date'], df['SMA_20'], label='20-day SMA', color='green')
        plt.plot(df['Date'], df['EMA_10'], label='10-day EMA', color='red')
        
        for idx, row in correct_signals.iterrows():
            ax1.annotate(f"Buy\nTP: {row['Take Profit']:.2f}\nSL: {row['Stop Loss']:.2f}",
                         (row['Date'], row['Close']),
                         textcoords="offset points",
                         xytext=(0, 10),  # adjust annotation position
                         ha='center',
                         fontsize=8,
                         color='green',
                         bbox=dict(facecolor='white', alpha=0.6))
        
        for idx, row in false_signals.iterrows():
            ax1.annotate(f"Buy\nTP: {row['Take Profit']:.2f}\nSL: {row['Stop Loss']:.2f}",
                         (row['Date'], row['Close']),
                         textcoords="offset points",
                         xytext=(0, 10),  # adjust annotation position
                         ha='center',
                         fontsize=8,
                         color='red',
                         bbox=dict(facecolor='white', alpha=0.6))
        
        for idx, row in current_signals.iterrows():
            ax1.annotate(f"Current\nTP: {row['Take Profit']:.2f}\nSL: {row['Stop Loss']:.2f}",
                         (row['Date'], row['Close']),
                         textcoords="offset points",
                         xytext=(0, 10),  # adjust annotation position
                         ha='center',
                         fontsize=8,
                         color='orange',
                         bbox=dict(facecolor='white', alpha=0.6))
        
        if live_price is not None:
            ax1.scatter([datetime.now()], [live_price], marker='o', color='blue', label='Live Price', alpha=1)

        fig.suptitle('Close Price with Buy Signals')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('signals_chart.png')
        plt.show()
    else:
        print("No data to display signals.")

def get_live_price():
    response = requests.get(BASE_URL + 'public/get-ticker', params={'instrument_name': symbol})
    ticker_data = response.json()

    if 'result' in ticker_data and 'data' in ticker_data['result']:
        data = ticker_data['result']['data'][0]
        return float(data['a'])
    else:
        print("Failed to fetch ticker data.")
        return None

def trading_bot():
    df = get_market_data(days=30)
    if df is not None:
        df = apply_trading_strategy(df)
    else:
        print("Failed to fetch market data. Skipping this iteration.")

    while True:
        try:
            live_price = get_live_price()
            if live_price is not None:
                print(f"Live Price: {live_price}")

                new_row = pd.DataFrame({'Date': [datetime.now()], 'Close': [live_price], 'Volume': [df['Volume'].mean()], 'Predictions': [None]})
                
                # Exclude empty or all-NA entries
                new_row = new_row.dropna(axis=1, how='all')

                if not new_row.empty:
                    df = pd.concat([df, new_row], ignore_index=True)

                data = pd.read_csv("preprocessed_Bitdata.csv")
                X_test = data.drop(columns=["Close", "Date"]).tail(100)
                live_prediction = ridge_regression.predict(X_test)[0]
                print(f"Live Prediction: {live_prediction}")

                df.loc[df.index[-1], 'Predictions'] = live_prediction

                current_conditions = {
                    "Prediction > Price": live_prediction > live_price,
                    "Price > SMA_10": live_price > df.iloc[-1]['SMA_10'],
                    "Price < Upper Band": live_price < df.iloc[-1]['Upper Band'],
                    "ADX > 20": df.iloc[-1]['ADX'] > 20,
                    "RSI < 70": df.iloc[-1]['RSI'] < 70,
                    "Volume > Threshold": df.iloc[-1]['Volume'] > volume_threshold
                }
                
                if all(current_conditions.values()):
                    print("Conditions met for a new signal.")
                    live_tp = live_price + df.iloc[-1]['ATR'] * take_profit_multiplier
                    live_sl = live_price - df.iloc[-1]['ATR'] * stop_loss_multiplier
                    live_date = datetime.now()
                else:
                    print("No new signal generated.")

                display_signals(df, live_price=live_price)

                # Print the summary of the last generated signal
                last_signal = df[df['Signal'] == 1].iloc[-1]
                status = 'Current' if last_signal['Current Signal'] == 1 else ('Correct' if last_signal['Correct Signal'] == 1 else 'False')
                print(f"Last Generated Signal:\nDate: {last_signal['Date']}\nClose: {last_signal['Close']}\nTP: {last_signal['Take Profit']}\nSL: {last_signal['Stop Loss']}\nStatus: {status}")
            
            time.sleep(60)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(60)

if __name__ == "__main__":
    trading_bot()
