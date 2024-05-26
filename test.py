# model.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    # Load the dataset
    data = pd.read_csv("House Pricing.csv")
    return data


def preprocess_data(data):
    # Convert categorical variables to numerical using one-hot encoding
    data = pd.get_dummies(
        data,
        columns=[
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
            "furnishingstatus",
        ],
        drop_first=True,
    )
    return data


def train_model(data):
    # Split data into features and target variable
    X = data.drop("price", axis=1)
    y = data["price"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    st.write("Root Mean Squared Error:", rmse)
    st.write("R-squared Score:", r2)

    return model


def main():
    st.title("House Price Prediction")

    # Load data
    data = load_data()

    # Preprocess data
    data = preprocess_data(data)

    # Train model
    model = train_model(data)

    # User input
    st.sidebar.header("Enter House Details")
    area = st.sidebar.number_input("Area (sqft)")
    bedrooms = st.sidebar.number_input("Bedrooms")
    bathrooms = st.sidebar.number_input("Bathrooms")
    stories = st.sidebar.number_input("Stories")
    mainroad = st.sidebar.selectbox("Main Road Access", ["Yes", "No"])
    guestroom = st.sidebar.selectbox("Guest Room", ["Yes", "No"])
    basement = st.sidebar.selectbox("Basement", ["Yes", "No"])
    hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ["Yes", "No"])
    airconditioning = st.sidebar.selectbox("Air Conditioning", ["Yes", "No"])
    parking = st.sidebar.number_input("Parking Spaces")
    prefarea = st.sidebar.selectbox("Preferred Area", ["Yes", "No"])
    furnishingstatus = st.sidebar.selectbox(
        "Furnishing Status", ["Semi-Furnished", "Unfurnished", "Furnished"]
    )

    # Convert user input to model-compatible format
    mainroad = 1 if mainroad == "Yes" else 0
    guestroom = 1 if guestroom == "Yes" else 0
    basement = 1 if basement == "Yes" else 0
    hotwaterheating = 1 if hotwaterheating == "Yes" else 0
    airconditioning = 1 if airconditioning == "Yes" else 0
    prefarea = 1 if prefarea == "Yes" else 0
    furnishingstatus = {"Semi-Furnished": 1, "Unfurnished": 2, "Furnished": 3}[
        furnishingstatus
    ]

    # Make prediction
    input_data = pd.DataFrame(
        {
            "area": [area],
            "bedrooms": [bedrooms],
            "bathrooms": [bathrooms],
            "stories": [stories],
            "mainroad_yes": [mainroad],
            "guestroom_yes": [guestroom],
            "basement_yes": [basement],
            "hotwaterheating_yes": [hotwaterheating],
            "airconditioning_yes": [airconditioning],
            "parking": [parking],
            "prefarea_yes": [prefarea],
            "furnishingstatus": [furnishingstatus],
        }
    )

    prediction = model.predict(input_data)

    # Display prediction
    st.header("Predicted House Price")
    st.write("₹", prediction[0])


if __name__ == "__main__":
    main()
