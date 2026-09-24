from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd
import numpy as np


app = FastAPI()


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Load model
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))


# Load car dataset
car = pd.read_csv("cleaned car.csv")


# Home page
@app.get("/")
def home():
    return {
        "message": "Car Price Prediction API is running"
    }


# Get car options
@app.get("/options")
def get_options():

    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    years = sorted(car["year"].unique(), reverse=True)
    fuel_types = car["fuel_type"].unique().tolist()

    return {
        "companies": companies,
        "car_models": car_models,
        "years": years,
        "fuel_types": fuel_types
    }


# Predict car price
@app.post("/predict")
def predict(
    company: str = Form(...),
    car_model: str = Form(...),
    year: int = Form(...),
    fuel_type: str = Form(...),
    driven: int = Form(...)
):

    # Create dataframe
    data = pd.DataFrame(
        [[
            car_model,
            company,
            year,
            driven,
            fuel_type
        ]],
        columns=[
            "name",
            "company",
            "year",
            "kms_driven",
            "fuel_type"
        ]
    )

    # Predict price
    prediction = model.predict(data)

    price = round(float(prediction[0]), 2)

    return {
        "predicted_price": price
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
