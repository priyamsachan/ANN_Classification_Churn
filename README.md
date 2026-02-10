# 📊 Customer Churn Prediction using ANN

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

An interactive deep learning web application that predicts the likelihood of a customer leaving a bank. This project utilizes an **Artificial Neural Network (ANN)** and is deployed via **Streamlit**.

## 🚀 Live Demo
[Insert your Streamlit Cloud Link Here]

## 🧐 Overview
Customer churn is a critical metric for businesses. This project provides a data-driven approach to identify "at-risk" customers by analyzing demographic and financial data such as credit score, geography, gender, age, tenure, balance, and more.

## 🛠️ Tech Stack
* **Modeling:** TensorFlow / Keras (Artificial Neural Networks)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Serialization:** Pickle (for saving Scalers and Encoders)
* **Frontend:** Streamlit

## 🏗️ Project Structure
```text
├── model.h5                # Trained Keras model
├── scaler.pkl              # StandardScaler object
├── label_encoder_gender.pkl # Label encoder for Gender
├── onehot_encoder_geo.pkl   # OneHot encoder for Geography
├── app.py                  # Streamlit web application
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
