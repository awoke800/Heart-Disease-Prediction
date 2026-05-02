
## deployment of the model using streamlit
import streamlit as st
import joblib
import pandas as pd


from IPython.display import Image, display

import streamlit as st

st.image("https://raw.githubusercontent.com/FarzadNekouee/Heart_Disease_Prediction/master/image.jpg")



# 1. Model Loadings
# Ensure the model file is in the same directory as this script.
try:
    # Loading the pre-trained classification model
    model = joblib.load('model.pkl')
except Exception as e:
    # Displays an error in the UI if the model fails to load
    st.error(f"Error loading model: {e}")

# Application UI Header
st.title("Heart Disease Prediction App")
st.write("Enter the patient's clinical data to predict heart disease risk.")

# 2. User Input Fields (Feature Collection)
# Creating a mix of numerical inputs and dropdown menus for the 13 required features
age = st.number_input("Age", min_value=1, max_value=100, value=50)
sex = st.selectbox("Sex (1=Male, 0=Female)", options=[1, 0])
cp = st.selectbox("Chest pain type (0-3)", options=[0, 1, 2, 3])
trestbps = st.number_input("Blood Pressure (BP)", value=120)
chol = st.number_input("Cholesterol", value=200)
fbs = st.selectbox("FBS over 120 (1=True, 0=False)", options=[0, 1])
restecg = st.selectbox("EKG results (0-2)", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate (Max HR)", value=150)
exang = st.selectbox("Exercise angina (1=Yes, 0=No)", options=[0, 1])
oldpeak = st.number_input("ST depression", value=0.0)
slope = st.selectbox("Slope of ST (0-2)", options=[0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", options=[0, 1, 2, 3])
thal = st.selectbox("Thallium (1-3)", options=[1, 2, 3])

# 3. Prediction Logic
if st.button("Predict"):
    # Convert inputs into a DataFrame
    # IMPORTANT: Column names must match the feature names used during training exactly.
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
                            columns=['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'])
    
    try:
        # Performing the prediction (0 or 1)
        prediction = model.predict(input_data)
        
        # Displaying the result with Streamlit alert components
        if prediction[0] == 1:
            st.error("Result: presenece'")
        else:
            st.success("absence'")
    except Exception as e:
        # Captures feature name mismatches or data type errors
        st.error(f"Prediction Error: {e}")