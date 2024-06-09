import subprocess
import pandas as pd
from joblib import load, dump
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# Run the preprocessing file
#subprocess.run(["python", "Bitprep.py"])

# Load the preprocessed test data
data = pd.read_csv("prediction_results_last_month.csv")

# Drop rows with missing values
data.dropna(inplace=True)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])


# Keep only the required columns
#data = data[[ 'Volume', 'Price', 'Average_Predicted_Price']]

# Normalize numerical features
scaler = MinMaxScaler()
data[[ 'Volume', 'Average_Predicted_Price']] = scaler.fit_transform(data[[ 'Volume', 'Average_Predicted_Price']])

# Split data into features and target
X = data[['Open','High','Low','Close','Adj Close','Volume','RF_Predictions','SVM_Predictions','NN_Predictions','GB_Predictions']]
y = data['Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
RFmodel = RandomForestRegressor()
RFmodel.fit(X_train, y_train)

SVMmodel = SVR()
SVMmodel.fit(X_train, y_train)

nnmodel = MLPRegressor(max_iter=1000)  # Increase max_iter
nnmodel.fit(X_train, y_train)

gbmodel = GradientBoostingRegressor()
gbmodel.fit(X_train, y_train)

# Save the trained models
dump(RFmodel, "random_forest_model_level2.joblib")
dump(SVMmodel, "svm_model_level2.joblib")
dump(nnmodel, "neural_network_model_level2.joblib")
dump(gbmodel, "gradient_boosting_model_level2.joblib")

# Evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2

RF_mse, RF_r2 = evaluate_model(RFmodel, X_test, y_test)
SVM_mse, SVM_r2 = evaluate_model(SVMmodel, X_test, y_test)
nn_mse, nn_r2 = evaluate_model(nnmodel, X_test, y_test)
gb_mse, gb_r2 = evaluate_model(gbmodel, X_test, y_test)

# Print evaluation results
print("Random Forest Model:")
print("Mean Squared Error:", RF_mse)
print("R^2 Score:", RF_r2)
print("\n")

print("SVM Model:")
print("Mean Squared Error:", SVM_mse)
print("R^2 Score:", SVM_r2)
print("\n")

print("Neural Network Model:")
print("Mean Squared Error:", nn_mse)
print("R^2 Score:", nn_r2)
print("\n")

print("Gradient Boosting Model:")
print("Mean Squared Error:", gb_mse)
print("R^2 Score:", gb_r2)
print("\n")
