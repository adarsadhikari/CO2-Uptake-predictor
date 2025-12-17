import os
import streamlit as st
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

st.title("Predict CO2 Uptake")

#defining directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(current_dir,"..", "assets")
model_path = os.path.join(assets_dir, "model.json")
samplefile_path = os.path.join(assets_dir, "sample.csv")

model = XGBRegressor()
model.load_model(model_path)

#function to make prediction
def predict_model(X):   
    output=model.predict(X)
    return (output)

#function to explain the prediction
def explain_model(X):
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    st.subheader("Waterfall plot for feature contribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(sv[0],show=False)
    st.pyplot(fig)

# Initialize session state
if "randval" not in st.session_state:
    st.session_state.randval = None

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = None

#select options      
colA, colB = st.columns(2)

with colA:
    random_btn = st.button("Generate features from database")
with colB:
    manual_btn = st.button("Enter features manually")

if "mode" not in st.session_state:
    st.session_state.mode = None
if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = None

if random_btn:
    st.session_state.mode = "random"
elif manual_btn:
    st.session_state.mode = "manual"

mode = st.session_state.mode

# Reset prediction + random value when mode changes
if mode != st.session_state.prev_mode:
    st.session_state.prediction = None
    st.session_state.randval = None
    st.session_state.prev_mode = mode

# --- RANDOM MODE ---
if mode == "random":

    df = pd.read_csv(samplefile_path)

    if random_btn:
        st.session_state.randval = df.sample()
        st.session_state.prediction = None

    if st.session_state.randval is not None:
        randval = st.session_state.randval
        st.write(randval.drop(['Adsorption'], axis=1))

        column1, column2 = st.columns(2)
        with column1:
            T = st.number_input("Temperature (K)", value=randval['T'].iloc[0], disabled=True)
            HOA = st.number_input("Heat of adsorption (kcal/mol)", value=randval['HOA'].iloc[0], disabled=True)
            UCV = st.number_input("Unit cell volume (Å³)", value=randval['UCV'].iloc[0], disabled=True)
            ASA = st.number_input("Accessible surface area (Å²)", value=randval['ASA'].iloc[0], disabled=True)

        with column2:
            P = st.number_input("Pressure (bar)", value=randval['P'].iloc[0], disabled=True)
            MP = st.number_input("Metal percentage", value=randval['M%'].iloc[0], disabled=True)
            D = st.number_input("Density (g/cm³)", value=randval['D'].iloc[0], disabled=True)
            AVAF = st.number_input("Accessible void area fraction", value=randval['AVAF'].iloc[0], disabled=True)

        X_sample = randval.drop(['Filename', 'Adsorption'], axis=1)

        if st.button("Predict"):
            st.session_state.prediction = predict_model(X_sample)

        if st.session_state.prediction is not None:
            st.success(f"The actual uptake is: {randval['Adsorption'].iloc[0]:.3f} mmol/g")
            st.success(f"The predicted uptake is: {st.session_state.prediction[0]:.3f} mmol/g")

            if st.button("Explain prediction"):
                explain_model(X_sample)

# --- MANUAL MODE ---
elif mode == "manual":

    st.write("Enter input values to make a prediction:")

    column1, column2 = st.columns(2)
    with column1:
        T = st.number_input("Temperature (K)", value=298.0)
        HOA = st.number_input("Heat of adsorption (kcal/mol)", value=1.39)
        UCV = st.number_input("Unit cell volume (Å³)", value=401.46)
        ASA = st.number_input("Accessible surface area (Å²)", value=9.7)

    with column2:
        P = st.number_input("Pressure (bar)", value=0.1)
        MP = st.number_input("Metal percentage", value=0.47)
        D = st.number_input("Density (g/cm³)", value=0.1)
        AVAF = st.number_input("Accessible void area fraction", value=0.14)

    X_sample = pd.DataFrame([[T, P, HOA, MP, UCV, D, ASA, AVAF]],
                            columns=['T', 'P', 'HOA', 'M%', 'UCV', 'D', 'ASA', 'AVAF'])

    inputs = [T, P, HOA, MP, UCV, D, ASA, AVAF]

    if st.button("Predict"):
        if any(val <= 0 for val in inputs):
            st.warning("All input values must be greater than 0.")
            st.session_state.prediction = None
            st.stop()

        st.session_state.prediction = predict_model(X_sample)

    if st.session_state.prediction is not None:
        st.success(f"The predicted uptake is: {st.session_state.prediction[0]:.3f} mmol/g")

        if st.button("Explain prediction"):
            explain_model(X_sample)
