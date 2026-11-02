# House Price Prediction Using MLOps

A complete, beginner-friendly Machine Learning and MLOps practical assignment for predicting house prices in Indian Rupees (₹).

---

## Project Structure

```text
house-price-mlops/
│
├── data/
│   └── house_prices.csv
│
├── models/
│   └── house_model.joblib
│
├── generate_data.py
├── train.py
├── api.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Step-by-Step Execution Guide

### 1. Install Python Dependencies
Open your terminal in the project root directory and run:
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset (Task 1)
Run the data generation script to create `data/house_prices.csv` with synthetic house records:
```bash
python generate_data.py
```
This will display the first 5 records, dataset info, and missing value counts.

### 3. Train the Model & Track with MLflow (Tasks 2, 3, 4, 5)
Run the training script to preprocess data, train the Linear Regression model, evaluate metrics (MAE, RMSE, R²), log parameters/metrics to MLflow, and save the model via Joblib:
```bash
python train.py
```

### 4. View MLflow Experiment Tracking (Task 4)
To view recorded experiments in the MLflow UI locally, run:
```bash
mlflow ui
```
Open your browser and navigate to: `http://127.0.0.1:5000`

### 5. Start the FastAPI Server (Task 6)
Start the FastAPI server using Uvicorn:
```bash
uvicorn api:app --reload --port 8000
```

### 6. Test the API Using Swagger UI
1. Open your browser and go to:
   `http://127.0.0.1:8000/docs`
2. Expand the `POST /predict` endpoint.
3. Click **"Try it out"**.
4. Enter sample house details in the request body:
   ```json
   {
       "area_sqft": 1500,
       "bedrooms": 3,
       "bathrooms": 2,
       "house_age": 5
   }
   ```
5. Click **"Execute"** and view the predicted house price in Indian Rupees (`INR`).

### 7. Initialize Git and Commit Code (Task 5)
```bash
git init
git add .
git commit -m "Initial commit: House price prediction MLOps project"
```
