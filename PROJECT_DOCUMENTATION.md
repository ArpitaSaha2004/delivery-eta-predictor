# Delivery ETA Predictor
## Machine Learning-Based Food Delivery Time Prediction

**Project Type:** Machine Learning and Web Application  
**Technologies:** Python, Pandas, NumPy, Scikit-learn, FastAPI, Pydantic, HTML, CSS, JavaScript, Joblib  
**Model:** Random Forest Regressor

---

# 1. Project Overview

The Delivery ETA Predictor is a machine learning-based web application designed to estimate the time required to complete a food delivery.

The application takes information such as delivery distance, delivery person's age and rating, weather conditions, road traffic, vehicle condition, order type, and other delivery-related factors.

Using these inputs, a trained machine learning model predicts the estimated delivery time in minutes.

The project combines data preprocessing, exploratory data analysis, feature engineering, machine learning, model evaluation, and API development into one application.

# 2. Problem Statement

Food delivery time can be affected by several factors, including:

- Distance between the restaurant and customer.
- Road traffic conditions.
- Weather conditions.
- Delivery person's experience and rating.
- Vehicle condition.
- Number of deliveries assigned to the delivery person.
- Order time and day of the week.

The objective of this project is to develop a machine learning model that learns patterns from historical food delivery data and predicts estimated delivery time based on these factors.

# 3. Project Objectives

1. Understand and analyse a real-world food delivery dataset.
2. Clean missing and inconsistent data.
3. Perform exploratory data analysis.
4. Create meaningful features from existing dataset columns.
5. Train and compare different regression algorithms.
6. Evaluate model performance using Mean Absolute Error.
7. Save the trained model for future predictions.
8. Develop a REST API using FastAPI.
9. Build a user-friendly interface using HTML, CSS, and JavaScript.
10. Implement input validation to prevent invalid user data.

# 4. Dataset Description

The project uses the **Food Delivery Dataset by Gaurav Malik**, obtained from Kaggle.

The training dataset contains:

- **45,593 rows**
- **20 original columns**

The dataset contains information about delivery personnel, restaurants, customers, weather, traffic, vehicles, orders, and delivery duration.

## Important dataset columns

| Column | Description |
|---|---|
| Delivery_person_Age | Age of delivery person |
| Delivery_person_Ratings | Rating of delivery person |
| Restaurant_latitude | Restaurant latitude |
| Restaurant_longitude | Restaurant longitude |
| Delivery_location_latitude | Customer location latitude |
| Delivery_location_longitude | Customer location longitude |
| Order_Date | Date of order |
| Time_Orderd | Order placement time |
| Time_Order_picked | Order pickup time |
| Weatherconditions | Weather during delivery |
| Road_traffic_density | Traffic condition |
| Vehicle_condition | Vehicle condition |
| Type_of_order | Food order category |
| Type_of_vehicle | Delivery vehicle |
| multiple_deliveries | Number of assigned deliveries |
| Festival | Festival indicator |
| City | City category |
| Time_taken(min) | Delivery time target |

The dataset also contains an ID and delivery person ID, which were not used as predictive features.

# 5. Data Cleaning

The original dataset contained inconsistent formatting and missing values represented as the string `"NaN"`.

The following cleaning operations were performed.

### 5.1 Removing extra spaces

Column names and text values were stripped of unnecessary leading and trailing spaces.

```python
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].str.strip()
```

### 5.2 Handling missing values

The string representations of missing values were converted into actual NumPy missing values.

```python
import numpy as np

df = df.replace(["NaN", "nan", ""], np.nan)
```

Missing values were handled later through Scikit-learn pipelines:

- Numerical columns: Median imputation.
- Categorical columns: Most frequent value imputation.

This approach ensures that imputation values are learned from the training data rather than the test data.

### 5.3 Cleaning the target column

The target column originally contained values such as `(min) 24`.

The numerical delivery time was extracted and converted into a numeric data type.

```python
df["Time_taken(min)"] = (
    df["Time_taken(min)"].str.extract(r"(\d+)")[0]
)

df["Time_taken(min)"] = pd.to_numeric(
    df["Time_taken(min)"],
    errors="coerce"
)
```

### 5.4 Converting numerical columns

```python
numeric_columns = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "multiple_deliveries"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
```

### 5.5 Cleaning weather information

The weather column contained an unnecessary text prefix.

```python
df["Weatherconditions"] = (
    df["Weatherconditions"]
    .str.replace("conditions ", "", regex=False)
)
```

# 6. Exploratory Data Analysis and Findings

Exploratory Data Analysis (EDA) was performed to understand the dataset, identify missing values, and examine the distribution of important variables.

## 6.1 Dataset dimensions

After loading the original training dataset:

- Rows: 45,593
- Columns: 20

## 6.2 Missing values

Some important columns contained missing values.

| Column | Missing values |
|---|---:|
| Delivery_person_Age | 1,854 |
| Delivery_person_Ratings | 1,908 |
| Time_Orderd | 1,731 |
| Road_traffic_density | 601 |
| multiple_deliveries | 993 |
| Festival | 228 |
| City | 1,200 |
| Weatherconditions | 616 |

Missing values were not removed because doing so would unnecessarily reduce the amount of training data.

Instead, imputation was performed inside the preprocessing pipeline.

## 6.3 Weather distribution

The dataset contains six weather categories:

| Weather | Records |
|---|---:|
| Fog | 7,654 |
| Stormy | 7,586 |
| Cloudy | 7,536 |
| Sandstorms | 7,495 |
| Windy | 7,422 |
| Sunny | 7,284 |

There were also 616 missing weather records.

The distribution shows that the dataset includes a reasonably varied range of weather conditions.

## 6.4 Road traffic distribution

| Traffic condition | Records |
|---|---:|
| Low | 15,477 |
| Jam | 14,143 |
| Medium | 10,947 |
| High | 4,425 |

There were 601 missing records.

## 6.5 Target variable analysis

The target variable is `Time_taken(min)`.

Its descriptive statistics were:

| Statistic | Value |
|---|---:|
| Count | 45,593 |
| Mean | 26.29 minutes |
| Standard deviation | 9.38 minutes |
| Minimum | 10 minutes |
| 25th percentile | 19 minutes |
| Median | 26 minutes |
| 75th percentile | 32 minutes |
| Maximum | 54 minutes |

The average delivery time in the dataset is approximately 26.29 minutes.

# 7. Feature Engineering

Feature engineering was performed to create useful information from existing dataset columns.

## 7.1 Distance calculation using Haversine Formula

The dataset provides restaurant and customer coordinates.

Instead of using four separate latitude and longitude values directly, the geographical distance was calculated.

The Haversine formula estimates the great-circle distance between two geographical coordinates.

\[
a = \sin^2\left(\frac{\Delta \phi}{2}\right)
+\cos(\phi_1)\cos(\phi_2)
\sin^2\left(\frac{\Delta \lambda}{2}\right)
\]

\[
d = 2R\arcsin(\sqrt{a})
\]

Where:

- \(R\) = Earth's radius (approximately 6371 km).
- \(\phi\) = Latitude in radians.
- \(\lambda\) = Longitude in radians.
- \(d\) = Distance in kilometres.

Python implementation:

```python
def calculate_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return 6371 * c


df["Distance_km"] = calculate_distance(
    df["Restaurant_latitude"],
    df["Restaurant_longitude"],
    df["Delivery_location_latitude"],
    df["Delivery_location_longitude"]
)
```

This produces a new feature called `Distance_km`.

## 7.2 Extracting order hour

The order time was converted into a datetime format to extract the hour.

```python
df["Order_Hour"] = pd.to_datetime(
    df["Time_Orderd"],
    format="%H:%M:%S",
    errors="coerce"
).dt.hour
```

The order hour helps the model identify time-related delivery patterns.

## 7.3 Extracting weekday

The order date was converted into datetime format.

```python
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

df["Order_Weekday"] = df["Order_Date"].dt.day_of_week
```

The weekday values are:

- Monday = 0
- Tuesday = 1
- Wednesday = 2
- Thursday = 3
- Friday = 4
- Saturday = 5
- Sunday = 6

## 7.4 Pickup time investigation

Order time and pickup time were also examined to understand the time between order placement and pickup.

```python
df["Order_Time"] = pd.to_datetime(
    df["Time_Orderd"],
    format="%H:%M:%S",
    errors="coerce"
)

df["Pickup_Time"] = pd.to_datetime(
    df["Time_Order_picked"],
    format="%H:%M:%S",
    errors="coerce"
)

df["Order_to_Pickup"] = (
    df["Pickup_Time"] - df["Order_Time"]
).dt.total_seconds() / 60
```

Some negative differences occurred because orders placed before midnight could be picked up after midnight.

These differences were adjusted by adding 1440 minutes when the difference was negative.

The resulting order-to-pickup time had an average of approximately 9.99 minutes.

However, its correlation with the target variable was approximately -0.0085, indicating almost no linear correlation.

The dataset did not conclusively establish whether the target represented total order-to-delivery time or only another delivery stage. Therefore, this feature was not used to manufacture a second-stage target or included in the final prediction inputs.

# 8. Feature Selection

The following 13 features were selected for model training.

```python
features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Distance_km",
    "Order_Hour",
    "Order_Weekday",
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City"
]

X = df[features]

y = df["Time_taken(min)"]
```

The resulting dataset dimensions were:

- X: 45,593 rows × 13 features.
- y: 45,593 target values.

# 9. Train-Test Split

The dataset was divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The split produced:

| Dataset | Records |
|---|---:|
| Training set | 36,474 |
| Testing set | 9,119 |

The training data was used to train the machine learning models, while the testing data was used to evaluate their performance.

# 10. Data Preprocessing

The numerical and categorical features required different preprocessing techniques.

Scikit-learn's `Pipeline` and `ColumnTransformer` were used to organize these operations.

## 10.1 Numerical features

Numerical columns included:

- Delivery person age
- Delivery person rating
- Distance
- Order hour
- Weekday
- Vehicle condition
- Multiple deliveries

The numerical pipeline used:

1. `SimpleImputer(strategy="median")`
2. `StandardScaler()`

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

## 10.2 Categorical features

Categorical columns included:

- Weather
- Road traffic
- Order type
- Vehicle type
- Festival
- City

The categorical pipeline used:

1. Most frequent value imputation.
2. One-Hot Encoding.

```python
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

One-Hot Encoding converts categorical values into numerical representations that machine learning algorithms can process.

## 10.3 Combining preprocessing steps

```python
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
```

This ensures that the correct preprocessing operations are applied to each group of features.

# 11. Model Training and Comparison

Three regression algorithms were trained and compared.

## 11.1 Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values.

It attempts to establish a linear relationship between input variables and the target variable.

## 11.2 Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

For regression, the predictions from the trees are averaged to produce the final output.

It can learn complex relationships between different input variables.

## 11.3 Gradient Boosting Regressor

Gradient Boosting builds decision trees sequentially. Each new tree attempts to improve the predictions made by the previous trees.

## 11.4 Model evaluation

Mean Absolute Error (MAE) was used to compare the models.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

Where:

- \(y_i\) = Actual delivery time.
- \(\hat{y}_i\) = Predicted delivery time.
- \(n\) = Number of test records.

A lower MAE indicates that predictions are closer to actual values on average.

## 11.5 Model comparison results

| Model | MAE |
|---|---:|
| Dummy Regressor (mean baseline) | 7.58 minutes |
| Linear Regression | 4.91 minutes |
| Gradient Boosting | 3.63 minutes |
| Random Forest | 3.21 minutes |

### Key finding

Random Forest Regressor achieved the lowest MAE of 3.21 minutes among the tested models.

Compared with the baseline MAE of 7.58 minutes, this represents approximately a 57.7% reduction in mean absolute error.

Therefore, Random Forest was selected as the final model.

These results are based on the project's random 80:20 train-test split and should not be interpreted as a guarantee of performance on future real-world deliveries.

# 12. Final Model Training and Saving

The selected Random Forest model was combined with the preprocessing pipeline.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

best_model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    )
])

best_model.fit(X_train, y_train)
```

The trained model was saved using Joblib.

```python
import os

os.makedirs("../models", exist_ok=True)

joblib.dump(
    best_model,
    "../models/delivery_eta_model.pkl"
)
```

The saved model is located at:

`models/delivery_eta_model.pkl`

Saving the complete pipeline means that the preprocessing operations and trained model can be reused together during prediction.

# 13. Backend Development Using FastAPI

FastAPI was used to develop the backend REST API.

The backend performs three important tasks:

1. Loads the saved machine learning model.
2. Receives delivery information from the frontend.
3. Returns the predicted delivery time.

## 13.1 Loading the trained model

```python
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "delivery_eta_model.pkl"
)
```

## 13.2 Input validation using Pydantic

Pydantic validates the incoming request data before it reaches the prediction function.

Examples of constraints include:

| Input | Validation |
|---|---|
| Age | 19–60 |
| Rating | 1–5 |
| Distance | Greater than 0 and up to 50 km |
| Order hour | 0–23 |
| Weekday | 0–6 |
| Vehicle condition | 0–3 |
| Multiple deliveries | 0–3 |
| Weather | Allowed dataset categories |
| Road traffic | Allowed dataset categories |
| Order type | Allowed dataset categories |
| Vehicle type | Allowed dataset categories |
| Festival | Yes or No |
| City | Allowed dataset categories |

The numerical limits are application-level validation choices.

## 13.3 Prediction endpoint

The application exposes a POST endpoint:

`/predict-eta`

The endpoint receives the user's inputs, converts them into a Pandas DataFrame, and sends them to the trained model.

```python
@app.post("/predict-eta")
def predict_eta(data: DeliveryInput):

    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_delivery_time_minutes":
            round(float(prediction), 2)
    }
```

The prediction is returned as a JSON response.

# 14. Frontend Development

The frontend was developed using HTML, CSS, and JavaScript.

## 14.1 HTML

HTML provides the structure of the application.

It includes input fields for:

- Delivery person age and rating.
- Delivery distance.
- Order hour and weekday.
- Weather and road traffic.
- Vehicle condition and vehicle type.
- Order type.
- Multiple deliveries.
- Festival status.
- City.

## 14.2 CSS

CSS was used to create a clean and responsive interface.

The main design features include:

- Two-column form layout.
- Consistent input sizes.
- Custom dropdown arrows.
- Responsive layout for smaller screens.
- Blue prediction button.
- Centered prediction popup.

## 14.3 JavaScript

JavaScript handles form submission and communication with FastAPI.

The form values are collected using `FormData`, converted into the appropriate numerical types, and sent to the API using the Fetch API.

```javascript
const response = await fetch("/predict-eta", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
});
```

The response is displayed inside a centered popup.

The popup supports:

- Loading message.
- Successful prediction message.
- Error message.
- Closing through the Close button.
- Closing by clicking outside the popup.
- Closing using the Escape key.

# 15. Input Validation in the Frontend

HTML validation attributes prevent users from entering invalid values through the interface.

Examples:

```html
<input
    type="number"
    name="Delivery_person_Age"
    min="19"
    max="60"
    step="1"
    placeholder="Enter age (19–60)"
    required
>
```

```html
<input
    type="number"
    name="Delivery_person_Ratings"
    min="1"
    max="5"
    step="0.1"
    placeholder="Enter rating (1–5)"
    required
>
```

Dropdowns begin with a disabled placeholder option.

This ensures that users must select an option instead of unintentionally submitting the first available category.

Validation is performed in both the frontend and backend.

# 16. Testing the Application

The backend was tested through FastAPI's interactive Swagger documentation.

Swagger is available at:

`http://127.0.0.1:8000/docs`

A sample API request was tested using the following values:

```json
{
    "Delivery_person_Age": 30,
    "Delivery_person_Ratings": 4.7,
    "Distance_km": 5.0,
    "Order_Hour": 19,
    "Order_Weekday": 5,
    "Weatherconditions": "Sunny",
    "Road_traffic_density": "Medium",
    "Vehicle_condition": 1,
    "Type_of_order": "Snack",
    "Type_of_vehicle": "motorcycle",
    "multiple_deliveries": 0,
    "Festival": "No",
    "City": "Urban"
}
```

The API successfully returned:

```json
{
    "predicted_delivery_time_minutes": 15.55
}
```

The frontend was also connected successfully to the prediction endpoint.

The prediction returned by the browser can differ from this sample when different inputs are provided.

# 17. Project Folder Structure

The project is organized as follows:

```text
delivery-eta-predictor/
│
├── app/
│   ├── main.py
│   └── templates/
│       └── index.html
│
├── data/
│   └── train.csv
│
├── models/
│   └── delivery_eta_model.pkl
│
├── notebooks/
│   └── step1.ipynb
│
├── requirements.txt
└── setup.sh
```

# 18. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization, where used |
| Scikit-learn | Preprocessing, model training and evaluation |
| Random Forest | Final regression model |
| Joblib | Saving and loading the model |
| FastAPI | Backend API |
| Pydantic | Input validation |
| HTML | Webpage structure |
| CSS | Interface styling |
| JavaScript | API communication and popup handling |
| Jupyter Notebook | Data analysis and model development |

# 19. Challenges and Learnings

During development, several practical challenges were addressed.

### Data cleaning

The dataset contained missing values represented as strings, inconsistent formatting, and unnecessary text in numerical columns.

This helped demonstrate the importance of data quality before model training.

### Feature engineering

Restaurant and customer coordinates were transformed into a distance feature using the Haversine formula.

Date and time information was also converted into useful numerical features.

### Missing-value handling

Different strategies were required for numerical and categorical columns.

Scikit-learn pipelines provided a structured way to handle both.

### Model selection

Multiple regression algorithms were compared rather than selecting a model without evaluation.

Random Forest performed best among the tested models.

### Model deployment

The trained model was saved and integrated into a FastAPI backend.

This demonstrated how a machine learning model can be used outside a Jupyter Notebook.

### Frontend and backend integration

JavaScript Fetch API was used to send user inputs to FastAPI and display the prediction.

This connected machine learning with practical web application development.

# 20. Limitations and Future Improvements

The current application is a working prototype and has some limitations.

1. The dataset's exact definition of the delivery-time target is not fully established.
2. The model is trained on historical dataset records and may not represent current real-world delivery patterns.
3. The application uses manually entered delivery information rather than live tracking data.
4. The current evaluation uses a random train-test split.
5. The model has not yet been evaluated on an independent external dataset.

Possible future improvements include:

- Add live weather API integration.
- Integrate maps and route distance estimation.
- Use real-time traffic information.
- Evaluate the model using cross-validation and a separate external dataset.
- Experiment with XGBoost and other advanced regression models.
- Develop a more detailed prediction dashboard.
- Deploy the application online.
- Add automated tests and API documentation.

# 21. Conclusion

The Delivery ETA Predictor demonstrates the complete workflow of developing a machine learning-based web application.

Starting from a raw food delivery dataset, the project involved data cleaning, exploratory analysis, feature engineering, preprocessing, model training, model comparison, and evaluation.

Random Forest Regressor achieved the best performance among the tested models, with a Mean Absolute Error of 3.21 minutes on the test set.

The trained model was saved using Joblib and integrated into a FastAPI backend. An HTML, CSS, and JavaScript frontend was developed to allow users to submit delivery information and receive estimated delivery times through a popup interface.

Overall, this project provided practical experience in machine learning, data analysis, Python programming, API development, input validation, and frontend-backend integration.

---

**Project Status:** Working local prototype with a trained regression model and connected web interface.

**Important:** The numerical findings and test results in this report are from the project's recorded experiments. The full source code is maintained separately in the project files and notebook.