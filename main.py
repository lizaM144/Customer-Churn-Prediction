from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

class CustomerInput(BaseModel):
    Age: int
    Gender: str
    Tenure: int
    MonthlyCharges: float
    InternetService: str
    TechSupport: str

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    # map the cateogorical inputs to numbers
    gender_val = 1 if customer.Gender == "Female" else 0
    is_fiber = 1 if customer.InternetService == "Fiber Optic" else 0
    is_no_internet = 1 if customer.InternetService == "No Internet" else 0
    tech_yes = 1 if customer.TechSupport == "Yes" else 0

    #scale numerical features
    input_nums = pd.DataFrame([[customer.Age, customer.Tenure, customer.MonthlyCharges]], 
                              columns=['Age', 'Tenure', 'MonthlyCharges'])
    input_nums_scaled = scaler.transform(input_nums)

    scaled_age = input_nums_scaled[0][0]
    scaled_tenure = input_nums_scaled[0][1]
    scaled_monthly = input_nums_scaled[0][2]

    # creating feature vector as order is important
    # ['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'InternetService_Fiber Optic', 'InternetService_No Internet', 'TechSupport_Yes']
    # features = pd.DataFrame([{
    #     'Age': scaled_age,
    #     'Gender': gender_val,
    #     'Tenure': scaled_tenure,
    #     'MonthlyCharges': scaled_monthly,
    #     'InternetService_Fiber Optic': is_fiber,
    #     'InternetService_No Internet': is_no_internet,
    #     'TechSupport_Yes': tech_yes
    # }])

    features = pd.DataFrame([[
        scaled_age, 
        gender_val, 
        scaled_tenure, 
        scaled_monthly, 
        is_fiber, 
        is_no_internet, 
        tech_yes
    ]], columns=['Age', 'Gender', 'Tenure', 'MonthlyCharges', 
                 'InternetService_Fiber Optic', 'InternetService_No Internet', 'TechSupport_Yes'])

    # predicting churn
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "probability_percent": round(probability * 100, 2),
        "risk_drivers": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"   
    }
