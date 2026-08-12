import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

def print_metrics(name, y_test, y_pred, y_train_len, y_test_len, leaked=False):
    print(f"\n{'='*40}")
    print(f"MODEL: {name}")
    print(f"{'='*40}")
    print(f"Training samples: {y_train_len}")
    print(f"Test samples: {y_test_len}")
    
    if len(y_test) == 0:
        print("Accuracy: Not reliably measurable due to dataset size")
        return
        
    acc = accuracy_score(y_test, y_pred)
    try:
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
    except:
        prec, rec, f1 = "N/A", "N/A", "N/A"
        
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.2%} {'(INVALID/LEAKED)' if leaked else ''}")
    print(f"Precision: {prec if isinstance(prec, str) else f'{prec:.2f}'}")
    print(f"Recall: {rec if isinstance(rec, str) else f'{rec:.2f}'}")
    print(f"F1-Score: {f1 if isinstance(f1, str) else f'{f1:.2f}'}")
    print(f"Confusion Matrix:\n{cm}")
    if leaked:
        print("Note: Metrics are invalid due to target leakage (features mathematically determine the target).")
    elif len(y_test) < 15:
        print("Note: Metrics are not statistically reliable due to extremely small dataset size.")

# ----------------------------------------
# FLOOD MODEL (Preserving existing logic)
# ----------------------------------------
try:
    df_flood = pd.read_csv('data/flood.csv')
    le = LabelEncoder()
    # PRESERVING LEAKAGE: original code fits le before split, and uses rainfall/river level in features
    df_flood['Soil Moisture (%)'] = le.fit_transform(df_flood['Soil Moisture (%)'])
    df_flood['Flood'] = (df_flood['Rainfall (mm)'] > 60) | (df_flood['River Level (m)'] > 60 )
    
    X_f = df_flood[['Rainfall (mm)', 'River Level (m)', 'Soil Moisture (%)']]
    y_f = df_flood['Flood']
    
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.2, random_state=42)
    
    rf_flood = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_flood.fit(X_train_f, y_train_f)
    y_pred_f = rf_flood.predict(X_test_f)
    
    print_metrics("Flood (Random Forest)", y_test_f, y_pred_f, len(y_train_f), len(y_test_f), leaked=True)
    
    joblib.dump(rf_flood, 'models/flood_model.joblib')
    joblib.dump(le, 'models/flood_le.joblib')
except Exception as e:
    print("Error training Flood model:", e)

# ----------------------------------------
# HURRICANE MODEL
# ----------------------------------------
try:
    df_hurr = pd.read_csv('data/hurricane.csv')
    X_h = df_hurr[['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']]
    y_h = df_hurr['Hurricane']
    
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
    
    # Using pipeline to save scaler and model together securely
    hurr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    hurr_pipeline.fit(X_train_h, y_train_h)
    y_pred_h = hurr_pipeline.predict(X_test_h)
    
    print_metrics("Hurricane (Random Forest)", y_test_h, y_pred_h, len(y_train_h), len(y_test_h), leaked=False)
    
    joblib.dump(hurr_pipeline, 'models/hurricane_model.joblib')
except Exception as e:
    print("Error training Hurricane model:", e)

# ----------------------------------------
# EARTHQUAKE MODEL (Preserving existing logic)
# ----------------------------------------
try:
    df_eq = pd.read_csv('data/earthquake.csv')
    # PRESERVING LEAKAGE: target is based on magnitude, which is an input feature
    df_eq['Earthquake'] = np.where(df_eq['Magnitude'] > 6.0, 1, 0)
    
    X_e = df_eq[['Temperature (°C)', 'Humidity (%)', 'Magnitude']]
    y_e = df_eq['Earthquake']
    
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_e, y_e, test_size=0.2, random_state=42)
    
    lr_eq = LogisticRegression(random_state=42, max_iter=1000)
    lr_eq.fit(X_train_e, y_train_e)
    y_pred_e = lr_eq.predict(X_test_e)
    
    print_metrics("Earthquake (Logistic Regression)", y_test_e, y_pred_e, len(y_train_e), len(y_test_e), leaked=True)
    
    joblib.dump(lr_eq, 'models/earthquake_model.joblib')
except Exception as e:
    print("Error training Earthquake model:", e)
