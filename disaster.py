

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import bcrypt
import json
import os
from twilio.rest import Client

# ------------------- Twilio Config -------------------
try:
    TWILIO_SID = st.secrets["TWILIO_SID"]
    TWILIO_TOKEN = st.secrets["TWILIO_TOKEN"]
    TWILIO_PHONE = st.secrets["TWILIO_PHONE"]
except:
    TWILIO_SID = os.getenv("TWILIO_SID", "")
    TWILIO_TOKEN = os.getenv("TWILIO_TOKEN", "")
    TWILIO_PHONE = os.getenv("TWILIO_PHONE", "")

USERS_FILE = "users.json"
REMEMBER_FILE = "remember_user.json"

# ------------------- Send SMS -------------------
def send_warning_sms(disaster_type):
    phone_numbers = st.session_state.get("phone")
    if not phone_numbers:
        return False
    body = f"⚠️ ALERT: High risk of a {disaster_type}. Please take precautions."
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    try:
        numbers = [num.strip() for num in phone_numbers.split(",") if num.strip()]
        for number in numbers:
            client.messages.create(
                body=body,
                from_=TWILIO_PHONE,
                to=number
            )
        return True
    except Exception as e:
        st.error(f"Failed to send SMS: {e}")
        return False

# ------------------- Auth System -------------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user(email, password):
    users = load_users()
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[email] = hashed_pw
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def check_login(email, password):
    users = load_users()
    if email in users and bcrypt.checkpw(password.encode(), users[email].encode()):
        return True
    return False

def login_page():
    st.markdown("""
        <div style='text-align: center;'>
            <h2 style='color: white;'>🔐 Login to Disaster Dashboard</h2>
        </div>
    """, unsafe_allow_html=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if check_login(email, password):
            st.session_state.logged_in = True
            st.session_state.email = email
            with open(REMEMBER_FILE, "w") as f:
                json.dump({"email": email}, f)
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
    if st.button("Register"):
        st.session_state.logged_in = "register"

def register_page():
    st.markdown("""
        <div style='text-align: center;'>
            <h2 style='color: white;'>📝 Register Account</h2>
        </div>
    """, unsafe_allow_html=True)
    email = st.text_input("New Email")
    password = st.text_input("New Password", type="password")
    if st.button("Create Account"):
        save_user(email, password)
        st.success("Account created! Please login.")
        st.session_state.logged_in = False

# ------------------- Initialize App -------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if st.session_state.logged_in == "register":
    register_page()
    st.stop()
elif not st.session_state.logged_in:
    login_page()
    st.stop()

# ------------------- Main Dashboard -------------------
st.markdown("""
    <style>
    .main-container {
        background-image: url('https://images.unsplash.com/photo-1556740738-b6a63e27c4df');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding: 20px;
        border-radius: 12px;
        color: white;
        backdrop-filter: brightness(0.6);
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='main-container'>
        <h1>🌍 Disaster Prediction Dashboard</h1>
        <p>Upload environmental data and manually input features to assess disaster risks like Earthquakes, Floods, or Hurricanes. SMS alerts will be sent if risks are detected.</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {}
    try:
        models['Earthquake'] = joblib.load('models/earthquake_model.joblib')
        models['Flood'] = joblib.load('models/flood_model.joblib')
        models['Flood_LE'] = joblib.load('models/flood_le.joblib')
        models['Hurricane'] = joblib.load('models/hurricane_model.joblib')
    except Exception as e:
        st.warning("ML Models not found in models/ directory. Falling back to rule-based logic.")
    return models

models = load_models()

disaster_type = st.selectbox("Select Disaster Type", ["Earthquake", "Flood", "Hurricane"])
st.session_state.phone = st.text_input("📱 Phone Number(s) for SMS (comma-separated)", value=st.session_state.get("phone", ""))

# ------------------- Dataset Upload and Preview -------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(data.head())

# ------------------- Manual Prediction -------------------
st.subheader("📊 Manual Input for Prediction")

if disaster_type == "Earthquake":
    temperature = st.number_input("Temperature (°C)", value=25.0)
    humidity = st.number_input("Humidity (%)", value=60.0)
    wind_speed = st.number_input("Wind Speed (km/h)", value=15.0)
    magnitude = st.number_input("Magnitude", min_value=0.0, max_value=10.0, value=5.0)
    
    model = models.get('Earthquake')
    if model:
        pred = model.predict([[temperature, humidity, magnitude]])[0]
        risk = bool(pred == 1)
        proba = model.predict_proba([[temperature, humidity, magnitude]])[0][1] * 100
    else:
        risk = magnitude >= 6.0
        proba = None
        
elif disaster_type == "Flood":
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=50.0)
    river_level = st.number_input("River Level (m)", min_value=0.0, value=3.0)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, value=40.0)
    
    model = models.get('Flood')
    le = models.get('Flood_LE')
    if model and le:
        try:
            sm_encoded = le.transform([int(soil_moisture)])[0]
        except:
            sm_encoded = 0
        pred = model.predict([[rainfall, river_level, sm_encoded]])[0]
        risk = bool(pred == 1)
        proba = model.predict_proba([[rainfall, river_level, sm_encoded]])[0][1] * 100
    else:
        risk = (rainfall >= 100.0) and (river_level >= 5.0) and (soil_moisture >= 70.0)
        proba = None
        
else:
    temperature = st.number_input("Temperature (°C)", value=28.0)
    humidity = st.number_input("Humidity (%)", value=70.0)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, value=80.0)
    
    model = models.get('Hurricane')
    if model:
        pred = model.predict([[temperature, humidity, wind_speed]])[0]
        risk = bool(pred == 1)
        proba = model.predict_proba([[temperature, humidity, wind_speed]])[0][1] * 100
    else:
        risk = (wind_speed >= 120.0) and (humidity >= 75.0) and (temperature >= 27.0)
        proba = None

if st.button("Predict Risk"):
    if risk:
        msg = f"⚠️ High likelihood of a {disaster_type} detected."
        if proba is not None:
            msg += f" (Model confidence: {proba:.1f}%)"
        st.error(msg)
        sent = send_warning_sms(disaster_type)
        if sent:
            st.success("📢 SMS alert sent successfully!")
    else:
        msg = f"✅ No immediate risk of {disaster_type} detected."
        if proba is not None:
            msg += f" (Model confidence: {proba:.1f}%)"
        st.success(msg)
