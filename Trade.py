import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import numpy as np

# Load the preprocessed data
data = pd.read_csv("preprocessed_Bitdata.csv")

# Use the last 300 days for testing
X_test = data.drop(columns=["Close", "Date"]).tail(300)
y_test = data["Close"].tail(300)
dates = data["Date"].tail(300)

# Load the trained Ridge Regression model
ridge_regression = load('ridge_regression_model.joblib')

# Make predictions
predictions = ridge_regression.predict(X_test)

# Prepare results DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Actual Price": y_test,
    "Predictions": predictions,
    "Volume": data["Volume"].tail(300)  # Assuming 'Volume' is a column in the dataset
})

# Reset index to ensure it's contiguous
df.reset_index(drop=True, inplace=True)

# Transaction fees
maker_fee = 0.0008  # 0.08%
taker_fee = 0.0007  # 0.7%

# Generate buy (1) signals based on model predictions
df['Signal'] = 0
df.loc[df['Predictions'] > df['Actual Price'], 'Signal'] = 1  # Buy signal

# Apply volume filter
volume_threshold = df['Volume'].quantile(0.31)  # Using 31st percentile as threshold
df.loc[df['Volume'] < volume_threshold, 'Signal'] = 0  # No trade if volume is below threshold

# Implement simple static take-profit
take_profit_threshold = 0.08  # 9% take-profit
take_profit_triggered_count = 0

for i in range(1, len(df)):
    if df.loc[i-1, 'Signal'] == 1:  # Buy signal
        entry_price = df.loc[i-1, 'Actual Price']
        take_profit_level = entry_price * (1 + take_profit_threshold)
        
        for j in range(i, len(df)):
            if df.loc[j, 'Actual Price'] > take_profit_level:
                df.loc[j, 'Signal'] = 0  # Exit position if price exceeds take-profit level
                take_profit_triggered_count += 1
                break
            if df.loc[j, 'Signal'] == 1:  # Update entry price on new buy signal
                entry_price = df.loc[j, 'Actual Price']
                take_profit_level = entry_price * (1 + take_profit_threshold)

# Implement strategy returns with a dynamic volume multiplier for buy signals
df['Position'] = df['Signal'].shift(1)
df['Volume Adjusted Return'] = df['Position'] * df['Actual Price'].pct_change()

# Apply a dynamic volume multiplier for buy signals based on volume percentiles
high_volume_threshold = df['Volume'].quantile(0.9)
df['Volume Multiplier'] = df['Volume'].apply(lambda x: 0.32 if x > high_volume_threshold else 10 if x < volume_threshold else 1)

df['Volume Adjusted Return'] *= df['Volume Multiplier']

# Calculate transaction costs
df['Transaction Cost'] = df['Position'].shift(1) * (maker_fee + taker_fee)
df['Net Return'] = df['Volume Adjusted Return'] - df['Transaction Cost']

# Calculate cumulative returns
df['Cumulative Market Returns'] = (1 + df['Actual Price'].pct_change()).cumprod()
df['Cumulative Strategy Returns'] = (1 + df['Net Return']).cumprod()

# Calculate performance metrics
total_return = df['Cumulative Strategy Returns'].iloc[-1] - 1
annualized_return = (1 + total_return) ** (365 / 300) - 1
annualized_volatility = df['Net Return'].std() * np.sqrt(365)
sharpe_ratio = (annualized_return - 0.02) / annualized_volatility if annualized_volatility != 0 else 0
max_drawdown = (df['Cumulative Strategy Returns'].cummax() - df['Cumulative Strategy Returns']).max()

print(f"Total Return: {total_return * 100:.2f}%")
print(f"Annualized Return: {annualized_return * 100:.2f}%")
print(f"Annualized Volatility: {annualized_volatility * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
print(f"Take-profit triggered count: {take_profit_triggered_count}")

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Cumulative Market Returns'], label='Market Returns')
plt.plot(df['Date'], df['Cumulative Strategy Returns'], label='Strategy Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Market Returns vs. Strategy Returns')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Identify successful and unsuccessful signals
df['Signal Result'] = df['Position'] * df['Actual Price'].pct_change()
successful_signals = df[df['Signal Result'] > 0]
unsuccessful_signals = df[df['Signal Result'] <= 0]

# Save results to CSV
df.to_csv("ridge_regression_strategy_results.csv", index=False)

# Print successful and unsuccessful signals
print("Successful Signals:")
print(successful_signals[['Date', 'Actual Price', 'Predictions', 'Signal', 'Signal Result']])

print("Unsuccessful Signals:")
print(unsuccessful_signals[['Date', 'Actual Price', 'Predictions', 'Signal', 'Signal Result']])
