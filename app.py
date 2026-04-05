import streamlit as st
import joblib
import numpy as np

model = joblib.load('models/Baseline_RandomForest_final.joblib')

st.title("Accident Severity Prediction")

# Example input
feature1 = st.number_input("Enter feature")

if st.button("Predict"):
    prediction = model.predict([[feature1]])
    st.write("Prediction:", prediction)
