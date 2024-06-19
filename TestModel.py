import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import subprocess

result = subprocess.run(['python', 'Bitprep.py'], capture_output=True, text=True)
# Load the preprocessed data
data = pd.read_csv("preprocessed_Bitdata.csv")

# Use the last 100 days for testing
X_test = data.drop(columns=["Close", "Date"]).tail(100)
y_test = data["Close"].tail(100)
dates = data["Date"].tail(100)

# Load the trained models
models = {
    "Random Forest": load('random_forest_model.joblib'),
    "Gradient Boosting": load('gradient_boosting_model.joblib'),
    "Support Vector Machine": load('support_vector_machine_model.joblib'),
    "Ridge Regression": load('ridge_regression_model.joblib'),
    "CatBoost": load('catboost_model.joblib')
}

# Make predictions and evaluate the models
results = pd.DataFrame({"Date": dates, "Actual Price": y_test})

for name, model in models.items():
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[f"{name} Predictions"] = predictions
    print(f"{name} MSE: {mse}")
    print(f"{name} R^2 score: {r2}")

# Save results to CSV
results.to_csv("model_predictions.csv", index=False)

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(results["Date"], results["Actual Price"], label="Actual Price", marker='o')

for name in models.keys():
    plt.plot(results["Date"], results[f"{name} Predictions"], label=f"{name} Predictions", linestyle='--', marker='x')

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Actual vs Predicted Prices (Last 100 Days)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
