import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load the preprocessed data
data = pd.read_csv("preprocessed_Bitdata.csv")

# Load the saved individual model
model = load('random_forest_model.joblib')  # Replace with the appropriate model file name

# Prepare the data for the last 100 days
last_100_days = data.tail(100)
X_last_100 = last_100_days.drop(columns=["Close", "Date"])
y_actual = last_100_days["Close"].values

# Preprocess the data
scaler = StandardScaler()
X_last_100_scaled = scaler.fit_transform(X_last_100)

# Make predictions using the loaded model
y_pred = model.predict(X_last_100_scaled)

# Create a DataFrame with the results
results = pd.DataFrame({
    'Date': last_100_days['Date'],
    'Actual Price': y_actual,
    'Predicted Price': y_pred
})

# Display the results in a table
print(results)

# Plot the actual and predicted prices
plt.figure(figsize=(14, 7))
plt.plot(results['Date'], results['Actual Price'], label='Actual Price', color='blue', marker='o')
plt.plot(results['Date'], results['Predicted Price'], label='Predicted Price', color='red', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices for the Last 100 Days')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
