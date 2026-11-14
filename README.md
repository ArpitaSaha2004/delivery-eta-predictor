# Delivery ETA Predictor 

A machine learning web application that predicts food delivery time based on delivery distance, weather, traffic, vehicle condition, and other order-related factors.

## Features
- Predicts estimated delivery time in minutes.
- Uses Random Forest Regression.
- Calculates delivery distance using the Haversine formula.
- Includes data preprocessing and feature engineering.
- Provides predictions through a FastAPI backend.
- Interactive frontend built with HTML, CSS, and JavaScript.

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · FastAPI · Pydantic · HTML · CSS · JavaScript · Joblib

## Model Performance

| Model | MAE |
|---|---:|
| Linear Regression | 4.91 min |
| Random Forest | **3.21 min** |
| Gradient Boosting | 3.63 min |

Random Forest was selected as the final model.

## Dataset
[Kaggle – Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset)

The dataset is not included in this repository.

## Run Locally

```bash
git clone https://github.com/ArpitaSaha2004/delivery-eta-predictor.git
cd delivery-eta-predictor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` to use the application.

## Author
**Arpita Saha**

[GitHub](https://github.com/ArpitaSaha2004)
