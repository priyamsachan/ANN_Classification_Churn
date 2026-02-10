import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# --- Page Configuration ---
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

model = load_model()

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --- Streamlit UI ---
st.title('📉 Customer Churn Prediction')
st.markdown("Enter the customer's details below to predict the likelihood of them leaving the service.")

# Create two columns for inputs to make it compact
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('📅 Age', 18, 92, 30)
    balance = st.number_input('💰 Balance', min_value=0.0, value=0.0, step=100.0)

with col2:
    credit_score = st.number_input('🎯 Credit Score', min_value=300, max_value=850, value=650)
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, step=500.0)
    tenure = st.slider('⏳ Tenure (Years)', 0, 10, 5)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)

# Full width inputs for binary choices
col3, col4 = st.columns(2)
with col3:
    has_cr_card = st.radio('💳 Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
with col4:
    is_active_member = st.radio('✨ Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)

# --- Data Processing ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale and Predict
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])

# --- Display Results ---
st.divider()
st.subheader("Prediction Results")

# Visual feedback based on probability
col_res1, col_res2 = st.columns([1, 2])

with col_res1:
    st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")

with col_res2:
    if prediction_proba > 0.5:
        st.error('**High Risk:** The customer is likely to churn.')
        st.progress(prediction_proba)
    else:
        st.success('**Low Risk:** The customer is likely to stay.')
        st.progress(prediction_proba)
        st.balloons() # Interactive celebration for loyal customers

# --- Footer ---
st.markdown("---")
st.markdown("<h4 style='text-align: center;'>Created by Priyam Sachan</h4>", unsafe_allow_html=True)

