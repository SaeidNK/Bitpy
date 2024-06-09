import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump
from catboost import CatBoostRegressor

# Load the historical price data
data = pd.read_csv("preprocessed_Bitdata.csv")

# Split data into features (X) and target variable (y)
X = data.drop(columns=["Close", "Date"])  # Features excluding the target variable and Date
y = data["Close"]  # Target variable

# Split data into training and testing sets using time-based splitting
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define preprocessing steps
numeric_features = X.columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# Define models to be tested
models = {
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR(),
    #"Neural Network": MLPRegressor(max_iter=4000),
    "Ridge Regression": Ridge(),
    "CatBoost": CatBoostRegressor(verbose=0)
}

# Pipeline for models
pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('model', model)])

# Hyperparameters to tune for each model
param_grids = {
    "Random Forest": {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    },
    "Gradient Boosting": {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.1, 0.01],
        'model__max_depth': [3, 5, 10]
    },
    "Support Vector Machine": {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf']
    },
    "Neural Network": {
        'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001, 0.01],
        'model__learning_rate_init': [0.001, 0.005, 0.01]
    },
    "Ridge Regression": {
        'model__alpha': [0.1, 1, 10, 100]
    },
    "CatBoost": {
        'model__iterations': [100, 200, 500],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__depth': [3, 5, 7]
    }
}

# Time Series Split for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# Train and evaluate models
for name, pipeline in pipelines.items():
    print(f"Training {name}...")
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {name}: {grid_search.best_params_}")
    y_pred = grid_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    variance_y = y_test.var()
    nmse = mse / variance_y  # Normalized MSE
    print(f"{name} MSE:", mse)
    print(f"{name} NMSE:", nmse)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} R^2 score:", r2)

    # Save the trained models
    dump(grid_search.best_estimator_, f'{name.lower().replace(" ", "_")}_model.joblib')
