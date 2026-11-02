"""
api.py
Task 6: Model Deployment Using FastAPI
Provides REST API endpoints for checking API status and predicting house prices in INR.
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="House Price Prediction API",
    description="MLOps FastAPI service for predicting house prices in Indian Rupees (INR)",
    version="1.0.0"
)

# Path to trained model
MODEL_PATH = "models/house_model.joblib"

# Load trained model at startup
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: Model not found at {MODEL_PATH}. Please run train.py first.")

# Define Pydantic request model with validation
class HouseInput(BaseModel):
    area_sqft: float = Field(..., gt=0, description="House area in square feet (must be > 0)")
    bedrooms: int = Field(..., gt=0, description="Number of bedrooms (must be > 0)")
    bathrooms: int = Field(..., gt=0, description="Number of bathrooms (must be > 0)")
    house_age: int = Field(..., ge=0, description="Age of the house in years (must be >= 0)")

    class Config:
        json_schema_extra = {
            "example": {
                "area_sqft": 1500,
                "bedrooms": 3,
                "bathrooms": 2,
                "house_age": 5
            }
        }

@app.get("/", summary="Check API Status")
def read_root():
    """
    GET endpoint to check whether the API is running.
    """
    return {
        "status": "running",
        "message": "Welcome to the House Price Prediction MLOps API!",
        "docs_url": "/docs"
    }

@app.post("/predict", summary="Predict House Price")
def predict_price(input_data: HouseInput):
    """
    POST endpoint to predict house prices in Indian Rupees (INR) based on area, bedrooms, bathrooms, and age.
    """
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        else:
            raise HTTPException(
                status_code=503, 
                detail="Trained model not found on server. Please run train.py first."
            )
    
    try:
        # Prepare feature array in correct column order:
        # ['area_sqft', 'bedrooms', 'bathrooms', 'house_age']
        features = np.array([[
            input_data.area_sqft,
            input_data.bedrooms,
            input_data.bathrooms,
            input_data.house_age
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Round to 2 decimal places for precision
        predicted_price = round(float(prediction), 2)
        
        return {
            "predicted_price": predicted_price,
            "currency": "INR",
            "input_received": {
                "area_sqft": input_data.area_sqft,
                "bedrooms": input_data.bedrooms,
                "bathrooms": input_data.bathrooms,
                "house_age": input_data.house_age
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
