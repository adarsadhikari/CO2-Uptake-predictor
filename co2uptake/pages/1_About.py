import os
import streamlit as st

st.set_page_config(page_title="Home", layout="centered")

st.title("Predict CO2 Uptake")

st.markdown("""
Welcome to the **CO₂ Uptake Predictor** — an interactive machine learning tool designed to estimate 
the **CO₂ adsorption capacity** of **Metal–Organic Frameworks (MOFs)** based on their key structural 
and thermodynamic properties.""")
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir,"..", "assets", "Waterfll.jpg")
# st.image("assets/Waterfll.jpg", width="stretch", caption="An example of a waterfall plot")
st.image(image_path, width='stretch', caption="An example of a waterfall plot")

st.markdown("""
---

### 🔍 **What does this app do?**
This application allows you to input eight important material features such as:

- **Temperature (K)**  
- **Pressure (bar)**  
- **Heat of adsorption (kcal/mol)**  
- **Metal percentage (%)**  
- **Unit cell volume (Å³)**  
- **Density (g/cm³)**  
- **Accessible surface area (Å²)**  
- **Accessible void area fraction**

Based on these inputs, it predicts the **CO₂ uptake capacity** 
of the MOF.

---

### 📈 **Beyond Prediction — Understanding the Model**

Simply predicting a value isn’t enough to trust a machine learning model.  
That’s why this app goes one step further: it **explains the model’s decision**.

After generating a prediction, you can click **“Explain Prediction”** to view a **SHAP Waterfall Plot**.

---

### 🌊 **What is a Waterfall Plot?**

A **SHAP (SHapley Additive exPlanations)** waterfall plot breaks down the prediction into **individual feature contributions**.

- Features that **increase** the predicted CO₂ uptake are shown in **red** (positive impact).  
- Features that **decrease** it are shown in **blue** (negative impact).  
- The combined effect of these contributions results in the final prediction.

This makes the model’s reasoning **transparent** — you can see *why* it predicted a certain uptake value for your input conditions.

---

### ⚙️ **How to Use**
1. Go to the **Home** page.  
2. Choose how you want to provide features:
    - **Generate From Database** → Automatically load a feature set from the database.
    - **Enter Features Manually** → Type in the feature values yourself. 
3. Click **Predict** to see the predicted CO₂ uptake.  
4. Click **Explain Prediction** to visualize how each feature influenced the prediction.

---

### 💡 **Goal of this Project**

This tool demonstrates how **machine learning** can assist researchers in understanding which factors most strongly affect CO₂ uptake performance in MOFs.

It’s not just a predictor, but also an **explainable AI** system for scientific insight.

---
""")


