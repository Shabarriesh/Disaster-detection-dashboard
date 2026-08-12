import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Flood
try:
    df = pd.read_csv('data/flood.csv')
    le = LabelEncoder()
    df['Soil Moisture (%)'] = le.fit_transform(df['Soil Moisture (%)'])
    df['Flood'] = (df['Rainfall (mm)'] > 60) | (df['River Level (m)'] > 60 )
    X = df[['Rainfall (mm)', 'River Level (m)'] + [col for col in df.columns if 'Soil Moisture (%)' in col]]
    y = df['Flood']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Flood Accuracy:', accuracy_score(y_test, y_pred))
except Exception as e:
    print('Flood error:', e)

# Hurricane
try:
    df2 = pd.read_csv('data/hurricane.csv')
    X = df2[['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']]
    y = df2['Hurricane']
    X_train, X_test, y_train, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred2 = model.predict(X_test_scaled)
    print('Hurricane Accuracy:', accuracy_score(y_test2, y_pred2))
except Exception as e:
    print('Hurricane error:', e)

# Earthquake
try:
    df3 = pd.read_csv('data/earthquake.csv')
    df3['Earthquake'] = np.where(df3['Magnitude'] > 6.0, 1, 0)
    X = df3[['Temperature (°C)', 'Humidity (%)', 'Magnitude']]
    y = df3['Earthquake']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Earthquake Accuracy:', accuracy_score(y_test, y_pred))
except Exception as e:
    print('Earthquake error:', e)
