# Disaster Detection Dashboard

This project is an educational prototype that demonstrates an end-to-end workflow for applying trained machine-learning models to environmental inputs through a Streamlit interface. It showcases how to train, serialize, and serve predictive models within a web application to assess environmental risks.

## Dashboard Preview

<!-- A real dashboard screenshot can be added here later -->

## Overview

The dashboard explores the integration of machine learning classification models to evaluate the risk of natural disasters based on environmental conditions. It supports three disaster categories: **Earthquake**, **Flood**, and **Hurricane**. 

Through the application, users can manually input environmental metrics or upload dataset previews to assess the current risk level. The dashboard dynamically loads pre-trained `scikit-learn` models to provide risk predictions and confidence probabilities. If high risk is detected, the system can optionally trigger SMS alerts via a Twilio API integration.

## Key Features

- **Disaster Selection**: Switch between Earthquake, Flood, and Hurricane risk assessment modes.
- **Manual Prediction Inputs**: Dynamic forms for inputting relevant environmental features (e.g., temperature, wind speed, rainfall, river level).
- **CSV Upload and Preview**: Functionality to upload and preview historical environmental datasets via Pandas.
- **Trained Model Loading**: Serialized model ingestion at runtime using Joblib.
- **Prediction Probability/Confidence**: Display of the model's confidence percentage alongside the binary risk prediction.
- **User Registration/Login**: Basic authentication system handling user signup and session management.
- **Bcrypt Password Hashing**: Passwords are securely hashed and validated at runtime.
- **Twilio SMS Alerts**: Configurable automated SMS warnings triggered when a high risk is detected.
- **Model Fallback Logic**: Hardcoded rule-based thresholds are used as a fallback if the trained models fail to load or are unavailable.

## Engineering Highlights

- **Inference Integration**: Integrated serialized scikit-learn models into a Streamlit inference workflow.
- **Separation of Concerns**: Separated model training and evaluation logic from the application layer.
- **Model Persistence**: Used Joblib to persist and load trained model artifacts.
- **Runtime Preprocessing**: Implemented preprocessing required for model inference.
- **Prediction Transparency**: Added model probability/confidence outputs to prediction results.
- **Secure Authentication**: Implemented bcrypt-based password hashing for user authentication.
- **External Notifications**: Integrated Twilio for conditional SMS notifications.
- **Robustness**: Added fallback prediction logic when trained models are unavailable.
- **Clean Structure**: Structured datasets and model artifacts into dedicated directories.
- **Security-Conscious**: Executed a strict repository cleanup, keeping runtime credentials completely outside the tracked source code.

## Machine Learning Pipeline

The project implements a straightforward pipeline to handle data from raw CSVs to interactive inference:

Data → preprocessing → model training → model serialization → model loading → user input → prediction → risk assessment

- **Pandas**: Used for loading the historical datasets (`.csv`), splitting features/targets, and rendering DataFrame previews in the dashboard.
- **NumPy**: Employed for vectorized conditional logic during target variable generation.
- **scikit-learn**: Provides the classification algorithms (`RandomForestClassifier`, `LogisticRegression`), preprocessing utilities (`StandardScaler`, `LabelEncoder`), and evaluation metrics.
- **Joblib**: Handles the serialization (`save_models.py`) and deserialization (`disaster.py`) of the trained model artifacts to enable fast inference without retraining.

## Models

| Disaster | Model | Preprocessing / Notes |
|----------|-------|-----------------------|
| Earthquake | Logistic Regression | No preprocessing. The target is synthetically derived from the `Magnitude` feature. |
| Flood | Random Forest | `LabelEncoder` applied to Soil Moisture. The target is synthetically derived from `Rainfall` and `River Level`. |
| Hurricane | Random Forest | Features are standardized using `StandardScaler` inside a `Pipeline`. Target uses the native dataset label. |

## Model Evaluation

| Model | Reported Accuracy | Evaluation Caveat |
|---|---:|---|
| Earthquake | 100% | Target is mathematically derived from Magnitude; target leakage makes this result unreliable |
| Hurricane | ~91% | Small dataset; not representative of production performance |
| Flood | 0% | Extremely small test split; only one test sample |

These results are included for transparency and should not be interpreted as production-level model performance.

## Limitations

As an educational prototype, this system has several limitations:
- Datasets used for training are extremely small and not representative of real-world meteorological datasets.
- Certain target variables are artificially derived directly from input features, leading to target leakage in training.
- The evaluation metrics should not be construed as indicative of a real production deployment.

## Application Architecture

```text
Disaster-detection-dashboard/
├── data/
│   ├── earthquake.csv
│   ├── flood.csv
│   └── hurricane.csv
├── models/
│   ├── earthquake_model.joblib
│   ├── flood_model.joblib
│   ├── flood_le.joblib
│   └── hurricane_model.joblib
├── disaster.py
├── save_models.py
├── evaluate_metrics.py
├── requirements.txt
├── runtime.txt
├── .gitignore
├── users.example.json
└── README.md
```

## Technology Stack

**Programming**
- Python

**Data & Machine Learning**
- Pandas
- NumPy
- Scikit-learn
- Joblib

**Application**
- Streamlit

**Authentication / Security**
- bcrypt

**External Services**
- Twilio

**Development**
- Git / GitHub

## Running the Project

1. Clone the repository:
```bash
git clone https://github.com/Shabarriesh/Disaster-detection-dashboard.git
cd Disaster-detection-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the requirements:
```bash
pip install -r requirements.txt
```

4. Configure Twilio credentials safely (Optional):
To enable SMS alerts, set the following environment variables or add them to your Streamlit secrets file (`.streamlit/secrets.toml`):
- `TWILIO_SID`
- `TWILIO_TOKEN`
- `TWILIO_PHONE`

5. Run the Streamlit application:
```bash
streamlit run disaster.py
```

## Security Notes

- **Credentials Excluded**: Runtime credentials and API keys are intentionally excluded from source control.
- **Environment Variables**: Twilio credentials should be supplied through secure environment variables or local Streamlit secrets.
- **Private Data Excluded**: User and runtime JSON files containing private session information and password hashes (`users.json`, `remember_me.json`, etc.) are actively excluded via `.gitignore`.
- **Safe Templates**: The tracked `users.example.json` is strictly provided as a safe structural template.
- The repository must never contain real API credentials or actual user data.
