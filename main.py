import streamlit as st
import joblib
import pandas as pd

# Set the page configuration
st.set_page_config(
    page_title="House Price Predictor", page_icon="🏠", layout="centered"
)

# Load the trained model and scaler
loaded_objects = joblib.load("house_price_model.sav")
loaded_model = loaded_objects["model"]
scaler = loaded_objects["scaler"]


def main():
    st.title("House Price Prediction App")

    with st.sidebar:
        st.header("Configuration")

        # Input fields for user to enter house features
        bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=6, value=3)
        bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=5, value=1)
        floors = st.slider("Number of Floors", min_value=1, max_value=5, value=1)

        # Living area first
        sqft_living = st.slider(
            "Living Area (sqft)", min_value=0, max_value=6000, value=2000
        )

        # Lot area slider that depends on living area
        sqft_lot = st.slider(
            "Lot Area (sqft)",
            min_value=sqft_living,
            max_value=200000,
            value=max(10000, sqft_living),
        )

        sqft_above = st.slider(
            "Above Ground Area (sqft)", min_value=700, max_value=6000, value=1340
        )

        # Waterfront feature with radio buttons
        waterfront = st.radio("Waterfront", options=["No", "Yes"])
        waterfront = 1 if waterfront == "Yes" else 0

        view = st.slider("View Rating", min_value=0, max_value=4, value=0)
        condition = st.slider("Condition", min_value=1, max_value=5, value=3)
        yr_built = st.slider("Year Built", min_value=1900, max_value=2014, value=1955)

    # Button to make a prediction
    if st.button("Predict House Price"):
        # Calculate house age
        house_age = 2014 - yr_built

        # Inverse the number of bedrooms
        inverted_bedrooms = 7 - bedrooms

        # Prepare the feature array for prediction
        features = [
            [
                inverted_bedrooms,
                bathrooms,
                sqft_living,
                sqft_lot,
                floors,
                waterfront,
                view,
                condition,
                sqft_above,
                house_age,  # Use house age instead of yr_built
            ]
        ]

        # Convert features to DataFrame
        feature_names = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "waterfront",
            "view",
            "condition",
            "sqft_above",
            "house_age",
        ]
        features_df = pd.DataFrame(features, columns=feature_names)

        # Scale the features
        features_scaled = scaler.transform(features_df)

        # Make the prediction
        predicted_price = loaded_model.predict(features_scaled)[0]

        # Display the predicted house price with improved UI
        st.markdown(
            """
            <div style="background-color: #80848c; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: #ffffff;">Predicted House Price</h2>
                <p style="font-size: 24px; color: #0d6320;"><strong>${:,.2f}</strong></p>
            </div>
            """.format(
                predicted_price
            ),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<h6>Made with &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by Ahmad, Nisya, Hendrick</h6>',
            unsafe_allow_html=True,
        )
