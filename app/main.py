from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Delivery ETA Predictor",
    description="Food delivery time prediction using Machine Learning",
    version="1.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "models" / "delivery_eta_model.pkl")


class DeliveryInput(BaseModel):
    Delivery_person_Age: int = Field(ge=19, le=60)
    Delivery_person_Ratings: float = Field(ge=1, le=5)
    Distance_km: float = Field(gt=0, le=50)
    Order_Hour: int = Field(ge=0, le=23)
    Order_Weekday: int = Field(ge=0, le=6)

    Weatherconditions: Literal[
        "Sunny", "Cloudy", "Fog", "Stormy", "Windy", "Sandstorms"
    ]
    Road_traffic_density: Literal["Low", "Medium", "High", "Jam"]
    Vehicle_condition: int = Field(ge=0, le=3)
    Type_of_order: Literal["Snack", "Meal", "Drinks", "Buffet"]
    Type_of_vehicle: Literal[
        "motorcycle", "scooter", "electric_scooter", "bicycle"
    ]
    multiple_deliveries: int = Field(ge=0, le=3)
    Festival: Literal["Yes", "No"]
    City: Literal["Urban", "Metropolitian", "Semi-Urban"]


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "app" / "templates" / "index.html")


@app.post("/predict-eta")
def predict_eta(data: DeliveryInput):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)[0]

    return {
        "predicted_delivery_time_minutes": round(float(prediction), 2)
    }