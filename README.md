# BitPy: Bitcoin Price Prediction using Machine Learning

Welcome to BitPy, a machine learning project aimed at predicting Bitcoin prices using various machine learning models. This project involves data preprocessing, model training, evaluation, and prediction visualization.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

BitPy is designed to predict Bitcoin prices using historical data and various technical indicators. It includes multiple machine learning models, such as Random Forest, Gradient Boosting, Support Vector Machine, Ridge Regression, and CatBoost. The project also visualizes the predictions and buy signals.

## Features

- Data preprocessing and technical indicator calculation (SMA, EMA, RSI, MACD)
- Multiple machine learning models for price prediction
- Hyperparameter tuning and cross-validation
- Prediction and buy signal visualization

## Technologies Used

- Python
- yfinance
- scikit-learn
- pandas
- matplotlib
- seaborn
- GridSearchCV

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SaeidNK/BitPy.git
    cd BitPy
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the data preparation script:
    ```bash
    python Bitprep.py
    ```

2. Train the models:
    ```bash
    python Train.py
    ```

3. Run the demo:
    ```bash
    python Demo.py
    ```

