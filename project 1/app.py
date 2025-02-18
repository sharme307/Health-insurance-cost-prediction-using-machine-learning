import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Sidebar navigation
nav = st.sidebar.radio("Navigation", ["About", "Predict"])

# Load dataset via file uploader
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
    return None

# About section
if nav == "About":
    st.title("Health Insurance Premium Predictor")
    st.text("This application predicts health insurance premiums based on user input.")
    st.image(r'C:\Users\sharme k\project\Health Insurance.jpeg', caption='Understanding Health Insurance', use_container_width=True)

# Prediction section
if nav == "Predict":
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    df = load_data(uploaded_file)
    
    if df is not None:
        # Data preprocessing
        df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
        df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
        df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

        # Features and target variable
        x = df.drop(columns='charges', axis=1)
        y = df['charges']

        # Train the model globally
        rfr = RandomForestRegressor()
        rfr.fit(x, y)

        # User input fields for prediction
        age = st.number_input("Age: ", step=1, min_value=0)
        sex = st.radio("Sex", ("Male", "Female"))
        s = 0 if sex == "Male" else 1
        bmi = st.number_input("BMI (Body Mass Index): ", min_value=10.0, max_value=50.0)
        children = st.number_input("Number of children: ", step=1, min_value=0)
        smoke = st.radio("Do you smoke?", ("Yes", "No"))
        sm = 0 if smoke == "Yes" else 1
        region = st.selectbox('Region', ('SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'))
        reg = {'SouthEast': 0, 'SouthWest': 1, 'NorthEast': 2, 'NorthWest': 3}[region]

        # Prediction button
        if st.button("Predict"):
            predicted_premium = rfr.predict([[age, s, bmi, children, sm, reg]])
            st.subheader("Predicted Premium")
            st.text(f"₹{predicted_premium[0]:,.2f}")
