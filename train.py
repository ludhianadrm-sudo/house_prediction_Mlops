"""
train.py
Tasks 2, 3, 4, & 5: Preprocessing, Model Training, MLflow Tracking, and Joblib Saving.
Ensures model accuracy (R² Score >= 80%).
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    print("=== STEP 1: Loading Dataset ===")
    data_path = 'data/house_prices.csv'
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Run generate_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records including example dataset rows.")
    
    print("\n=== STEP 2: Data Preprocessing ===")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    
    # Separate independent variables (X) and dependent variable (y)
    X = df[['area_sqft', 'bedrooms', 'bathrooms', 'house_age']]
    y = df['price']
    
    # Split into 80% training data and 20% testing data (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    print("\n=== STEP 3: Model Development & Evaluation ===")
    # Train Linear Regression model
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    accuracy_percentage = r2 * 100
    
    print(f"\n1. MAE (Mean Absolute Error): ₹{mae:,.2f}")
    print(f"2. RMSE (Root Mean Squared Error): ₹{rmse:,.2f}")
    print(f"3. R² Score: {r2:.4f} (Model Accuracy: {accuracy_percentage:.2f}%)")
    
    if accuracy_percentage >= 80.0:
        print("-> SUCCESS: Model accuracy is >= 80%!")
    else:
        print("-> WARNING: Model accuracy is below 80%.")

    print("\n=== STEP 4: MLflow Experiment Tracking ===")
    mlflow.set_experiment("House Price Prediction")
    
    with mlflow.start_run():
        mlflow.log_param("model_name", "LinearRegression")
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("fit_intercept", True)
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("accuracy_percentage", accuracy_percentage)
        
        mlflow.sklearn.log_model(model, "house_price_model")
        print("MLflow experiment logged successfully.")

    print("\n=== STEP 5: Model Saving with Joblib ===")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/house_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
