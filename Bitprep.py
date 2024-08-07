import yfinance as yf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to calculate Simple Moving Average (SMA)
def calculate_sma(data, window):
    return data.rolling(window=window).mean()

# Function to calculate Exponential Moving Average (EMA)
def calculate_ema(data, span):
    return data.ewm(span=span, adjust=False).mean()

# Function to calculate Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to calculate Moving Average Convergence Divergence (MACD)
def calculate_macd(data, short_window=12, long_window=26):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd_line = short_ema - long_ema
    signal_line = calculate_ema(macd_line, 9)  # Signal line is typically a 9-period EMA of the MACD line
    return macd_line, signal_line

# Download data
ticker = 'BTC-USD'
df = yf.download(ticker)

# Filter data
df = df.loc['2014-09-17':].copy()

# Drop missing values before calculating technical indicators
df.dropna(inplace=True)

# Calculate technical indicators and add them to the DataFrame
df['SMA_10'] = calculate_sma(df['Adj Close'], 10)
df['EMA_10'] = calculate_ema(df['Adj Close'], 10)
df['RSI'] = calculate_rsi(df['Adj Close'])
df['MACD'], df['MACD_Signal'] = calculate_macd(df['Adj Close'])

# Adding lagged features
for lag in range(1, 6):  # Add more lagged features
    df[f'lag_{lag}'] = df['Adj Close'].shift(lag)

# Adding rolling statistics
for window in [5, 10, 20]:  # Add more rolling windows
    df[f'rolling_mean_{window}'] = df['Adj Close'].rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df['Adj Close'].rolling(window=window).std()

# Drop rows with NaN values generated by technical indicators calculation
df.dropna(inplace=True)

# Select relevant features
selected_features = ['Adj Close', 'Volume', 'SMA_10', 'EMA_10', 'RSI', 'MACD',
                     'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
                     'rolling_mean_5', 'rolling_mean_10', 'rolling_mean_20',
                     'rolling_std_5', 'rolling_std_10', 'rolling_std_20', 'Close', 'High', 'Low']
df = df[selected_features]

# Save the date column separately
dates = df.index

# Separate features and target for scaling
features = df.drop(columns=['Close'])  # Exclude 'Close' as it is the target
target = df['Close']

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)

# Merge the scaled features with the dates and target
scaled_df['Close'] = target.values
scaled_df['Date'] = dates

# Save the preprocessed data
scaled_df.to_csv("preprocessed_Bitdata.csv", index=False)
# Correlation analysis
correlation_matrix = scaled_df.corr()
