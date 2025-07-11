# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model components from models folder
model = joblib.load("models/logistic_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

st.set_page_config(page_title="Customer Satisfaction Predictor", layout="centered")
st.title("📊 Customer Satisfaction Prediction App")

st.markdown("""
This app predicts the **Customer Satisfaction Rating** (on a scale of 1 to 5) based on support ticket details.
""")

# Input form
with st.form("prediction_form"):
    st.subheader("📝 Enter Ticket Details")

    age = st.slider("Customer Age", 18, 100, 30)
    gender = st.selectbox("Gender", ['Female', 'Male', 'Other'])
    product = st.selectbox("Product Purchased", ['LG Smart TV', 'GoPro Hero', 'Dell XPS', 'Microsoft Office', 'Autodesk AutoCAD'])
    ticket_type = st.selectbox("Ticket Type", ['Technical issue', 'Billing inquiry', 'Cancellation', 'Product inquiry', 'Refund request'])
    status = st.selectbox("Ticket Status", ['Closed', 'Open', 'Pending Customer Response'])
    resolution = st.selectbox("Resolution", ['Issue resolved', 'Refund initiated', 'Replaced product', 'Not Provided'])
    priority = st.selectbox("Ticket Priority", ['Low', 'Medium', 'High', 'Critical'])
    channel = st.selectbox("Ticket Channel", ['Email', 'Chat', 'Social media'])
    response_delay = st.number_input("Response Delay (hrs)", 0.0, 200.0, 24.0)
    resolution_time = st.number_input("Resolution Time (hrs)", 0.0, 200.0, 48.0)
    purchase_month = st.selectbox("Purchase Month", list(range(1, 13)))

    # Age group binning
    age_group = pd.cut([age], bins=[0, 20, 30, 40, 50, 60, 70, 100],
                       labels=['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'])[0]

    submit = st.form_submit_button("🔍 Predict Satisfaction")

# Encoding map
label_maps = {
    'Customer Gender': {'Female': 0, 'Male': 1, 'Other': 2},
    'Product Purchased': {
        'Autodesk AutoCAD': 0, 'Dell XPS': 1, 'GoPro Hero': 2,
        'LG Smart TV': 3, 'Microsoft Office': 4
    },
    'Ticket Type': {
        'Billing inquiry': 0, 'Cancellation': 1, 'Product inquiry': 2,
        'Refund request': 3, 'Technical issue': 4
    },
    'Ticket Status': {'Closed': 0, 'Open': 1, 'Pending Customer Response': 2},
    'Resolution': {
        'Issue resolved': 0, 'Not Provided': 1,
        'Refund initiated': 2, 'Replaced product': 3
    },
    'Ticket Priority': {'Critical': 0, 'High': 1, 'Low': 2, 'Medium': 3},
    'Ticket Channel': {'Chat': 0, 'Email': 1, 'Social media': 2},
    'Age Group': {'<20': 0, '20-29': 1, '30-39': 2, '40-49': 3, '50-59': 4, '60-69': 5, '70+': 6}
}

if submit:
    # Prepare input dictionary
    input_dict = {
        'Customer Age': age,
        'Customer Gender': label_maps['Customer Gender'][gender],
        'Product Purchased': label_maps['Product Purchased'][product],
        'Ticket Type': label_maps['Ticket Type'][ticket_type],
        'Ticket Status': label_maps['Ticket Status'][status],
        'Resolution': label_maps['Resolution'][resolution],
        'Ticket Priority': label_maps['Ticket Priority'][priority],
        'Ticket Channel': label_maps['Ticket Channel'][channel],
        'Response Delay (hrs)': response_delay,
        'Resolution Time (hrs)': resolution_time,
        'Purchase Month': purchase_month,
        'Age Group': label_maps['Age Group'][age_group]
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df[features])

    prediction = model.predict(input_scaled)[0]
    st.success(f"✅ Predicted Customer Satisfaction Rating: **{prediction} / 5**")
