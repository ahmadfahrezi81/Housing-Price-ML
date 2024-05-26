import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from predict_model import predict
import numpy as np

# Load the trained model
model = joblib.load("svr_model.sav")


# Function to preprocess input data
def preprocess_input(
    area,
    bedrooms,
    bathrooms,
    stories,
    mainroad,
    guestroom,
    basement,
    hotwaterheating,
    airconditioning,
    parking,
    prefarea,
    furnishingstatus,
):
    # Create a dictionary with input values
    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad_yes": 1 if mainroad == "yes" else 0,
        "guestroom_yes": 1 if guestroom == "yes" else 0,
        "basement_yes": 1 if basement == "yes" else 0,
        "hotwaterheating_yes": 1 if hotwaterheating == "yes" else 0,
        "airconditioning_yes": 1 if airconditioning == "yes" else 0,
        "parking": parking,
        "prefarea_yes": 1 if prefarea == "yes" else 0,
    }

    # Handle furnishing status
    if furnishingstatus == "furnished":
        input_data["furnishingstatus_furnished"] = 1
        input_data["furnishingstatus_semi-furnished"] = 0
        input_data["furnishingstatus_unfurnished"] = 0
    elif furnishingstatus == "semi-furnished":
        input_data["furnishingstatus_furnished"] = 0
        input_data["furnishingstatus_semi-furnished"] = 1
        input_data["furnishingstatus_unfurnished"] = 0
    else:
        input_data["furnishingstatus_furnished"] = 0
        input_data["furnishingstatus_semi-furnished"] = 0
        input_data["furnishingstatus_unfurnished"] = 1

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    return input_df


# Main Streamlit UI
st.title("Home Price Prediction")
st.write("---")

# Get input from the user
area = st.number_input("Area of the house (in sq ft)", min_value=0, step=1)
bedrooms = st.number_input("No. of bedrooms", min_value=0, step=1)
bathrooms = st.number_input("No. of bathrooms", min_value=0, step=1)
stories = st.number_input("No. of stories", min_value=0, step=1)
mainroad = st.radio("Main road access", ("yes", "no"))
guestroom = st.radio("Guest room availability", ("yes", "no"))
basement = st.radio("Basement availability", ("yes", "no"))
hotwaterheating = st.radio("Hot water heating", ("yes", "no"))
airconditioning = st.radio("Air conditioning", ("yes", "no"))
parking = st.number_input("No. of parking spaces", min_value=0, step=1)
prefarea = st.radio("Preferred area", ("yes", "no"))
furnishingstatus = st.radio(
    "Furnishing status", ("furnished", "semi-furnished", "unfurnished")
)

# Predict button
if st.button("Predict Now"):
    cost = predict(
        np.array(
            [
                [
                    area,
                    bedrooms,
                    bathrooms,
                    stories,
                    mainroad,
                    guestroom,
                    basement,
                    hotwaterheating,
                    airconditioning,
                    parking,
                    prefarea,
                    furnishingstatus,
                ]
            ]
        )
    )
    st.text(cost[0])
    # # Preprocess input data
    # input_df = preprocess_input(
    #     price,
    #     area,
    #     bedrooms,
    #     bathrooms,
    #     stories,
    #     mainroad,
    #     guestroom,
    #     basement,
    #     hotwaterheating,
    #     airconditioning,
    #     parking,
    #     prefarea,
    #     furnishingstatus,
    # )

    # # Make prediction
    # prediction = model.predict(input_df)

    # Display prediction
    # st.write("---")
    # st.write(f"Predicted price of the house: {prediction[0]:.2f}")
