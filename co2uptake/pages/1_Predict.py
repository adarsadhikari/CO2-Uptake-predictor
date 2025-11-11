import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import shap
import matplotlib.pyplot as plt

st.title("Predict CO2 Uptake")
st.write("Enter input values to make a prediction:")


column1, column2 = st.columns(2)
with column1:
    T = st.number_input("Temperature (K)", value=298.0,min_value=1e-6)
    HOA = st.number_input("Heat of adsorption (kcal/mol)", value=1.39,min_value=1e-6)
    UCV = st.number_input("Unit cell volume (Å³)", value=401.46,min_value=1e-6)
    ASA = st.number_input("Accessible surface area (Å²)", value=9.7,min_value=1e-6)
with column2:
    P = st.number_input("Pressure (bar)", value=0.1,min_value=1e-6)
    MP = st.number_input("Metal percentage", value=0.47,min_value=1e-6)
    D = st.number_input("Density (g/cm³)", value=0.1,min_value=1e-6)
    AVAF = st.number_input("Accessible void area fraction", value=0.14,min_value=1e-6)


# feature creation from user input
feature_names=['T','P','HOA','M%','UCV','D','ASA','AVAF']
X_sample = pd.DataFrame(
    [[T, P, HOA, MP, UCV, D, ASA, AVAF]],
    columns=feature_names)


#function to make prediction
def predict_model(X):   
    model = XGBRegressor()
    model.load_model("assets\model.json")
    output=model.predict(X)
    return (output)


#function to explain the prediction
def explain_model(X):
    model = XGBRegressor()
    model.load_model("assets\model.json")
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    st.subheader("Waterfall plot for feature contribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(sv[0],show=False)
    st.pyplot(fig)
    

if "prediction" not in st.session_state:
    st.session_state.prediction = None


if st.button("Predict"):
    st.session_state.prediction = predict_model(X_sample)

if st.session_state.prediction is not None:
    st.success(f"The predicted uptake is: {st.session_state.prediction[0]:.3f}")

    if st.button("Show SHAP Plot"):
        explain_model(X_sample)


        

    