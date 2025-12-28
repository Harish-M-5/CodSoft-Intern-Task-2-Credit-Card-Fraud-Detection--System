import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Credit Card Fraud Detection")

try:
    model = pickle.load(open("fraud_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    st.error("❌ Model not found. Please run model_train.py first.")
    st.stop()

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to check fraud")

feature_count = model.coef_.shape[1]

inputs = []
for i in range(feature_count):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    data = np.array(inputs).reshape(1, -1)
    data = scaler.transform(data)
    result = model.predict(data)

    if result[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")
