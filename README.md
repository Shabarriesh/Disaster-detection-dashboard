# Disaster Detection Dashboard

## Problem Statement
The Disaster Detection Dashboard is an experimental, prototype early warning system designed to predict the risk of natural disasters based on environmental conditions. It allows users to manually input current weather metrics and receive risk assessments via a web interface, with the option for automated SMS alerts.

## Supported Disasters
- **Earthquake**
- **Flood**
- **Hurricane**

## Project Workflow
1. User creates an account or logs in via a custom bcrypt-secured system.
2. User selects a disaster category from the dashboard.
3. User manually inputs environmental metrics (e.g., Temperature, Humidity, Wind Speed, Rainfall).
4. The dashboard processes these inputs through pre-trained Machine Learning models (or rule-based fallbacks).
5. If a high risk is detected, a warning is displayed.
6. An automated Twilio SMS alert is sent to registered phone numbers.

## Machine Learning Models
This project utilizes baseline models trained on historical dataset samples. The trained models are saved as `.joblib` files and loaded directly into the dashboard for real-time inference.

* **Flood Model:** Random Forest Classifier
* **Hurricane Model:** Random Forest Classifier (with StandardScaler pipeline)
* **Earthquake Model:** Logistic Regression

## Data Preprocessing
The models use standard preprocessing steps. For Hurricane prediction, `StandardScaler` is fit on the training data and preserved via an `sklearn` Pipeline. For Flood prediction, `LabelEncoder` is used for soil moisture categories. 

## Model Evaluation
* **Hurricane Model:** Achieved an indicative ~90.9% test accuracy.
* **Earthquake Model:** Achieved 100% test accuracy. *(Note: This is invalid and highly inflated due to known target leakage in the original training methodology, where the target was mathematically derived from an input feature).*
* **Flood Model:** Evaluated at 0% test accuracy due to the dataset containing only a single test sample that was incorrectly predicted.

## Streamlit Dashboard
The frontend is built using Streamlit with custom CSS. It features:
* Disaster selection menus.
* Slider/numeric inputs matching the model features.
* Model confidence/probability displays (e.g., "Model confidence: 78.5%").

## Twilio SMS Alerts
The dashboard integrates with the Twilio API. If a disaster's predicted likelihood is classified as high risk, the system dispatches SMS warnings to user-provided phone numbers.

## Authentication
The application includes a custom-built JSON storage authentication system. Passwords are securely hashed and checked using the `bcrypt` library.

## Limitations
This is an educational/portfolio prototype and should **not** be used for actual disaster prediction. 

* **Dataset Sizes:** The datasets used for model training are extremely small and not statistically robust for real-world modeling. 
  * `flood.csv`: 4 rows
  * `hurricane.csv`: 51 rows
  * `earthquake.csv`: 53 rows
* **Known ML Limitations:** The Earthquake and Flood models currently suffer from target leakage, where the input features deterministically define the target label.
* **No Real-time API:** The application does not automatically fetch live weather data; all inputs must be manually provided by the user.

## Project Structure
```
├── app.py / disaster.py        # Streamlit dashboard application
├── save_models.py              # Script to train and export ML models
├── models/                     # Directory containing trained .joblib models
├── *.csv                       # Datasets
└── users.json                  # Encrypted user storage
```

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Configure Twilio credentials in `.streamlit/secrets.toml` or as environment variables (`TWILIO_SID`, `TWILIO_TOKEN`, `TWILIO_PHONE`).
3. Run the Streamlit application: `streamlit run disaster.py`
