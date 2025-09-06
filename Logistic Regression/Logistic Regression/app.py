import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("logistic_model.pkl")

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details below to predict survival.")

# Sidebar for user inputs
st.sidebar.header("Passenger Features")

def user_input_features():
    Pclass = st.sidebar.selectbox("Passenger Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
    SibSp = st.sidebar.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    Parch = st.sidebar.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
    Fare = st.sidebar.number_input("Ticket Fare", min_value=0.0, value=32.2)

    data = {"Pclass": Pclass, "SibSp": SibSp, "Parch": Parch, "Fare": Fare}
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Passenger Survived (Probability: {prediction_proba:.2f})")
    else:
        st.error(f"❌ Passenger Did Not Survive (Probability: {prediction_proba:.2f})")
