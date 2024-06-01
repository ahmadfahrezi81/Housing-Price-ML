import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Load the trained model
loaded_model = joblib.load("house_price_model.sav")

# Set up the Streamlit app
st.title("House Price Prediction App")

# Input fields for user to enter house features with sliders where applicable
bedrooms = st.slider('Number of Bedrooms', min_value=0, max_value=10, value=3)
bathrooms = st.slider('Number of Bathrooms', min_value=0, max_value=10, value=1)
sqft_living = st.slider('Living Area (sqft)', min_value=0, max_value=10000, value=1340)
sqft_lot = st.slider('Lot Area (sqft)', min_value=0, max_value=100000, value=7912)
floors = st.slider('Number of Floors', min_value=0.0, max_value=5.0, value=1.5)

# Waterfront feature with radio buttons
waterfront = st.radio('Waterfront', options=['No', 'Yes'])
waterfront = 1 if waterfront == 'Yes' else 0

view = st.slider('View Rating', min_value=0, max_value=4, value=0)
condition = st.slider('Condition', min_value=1, max_value=5, value=3)
sqft_above = st.slider('Above Ground Area (sqft)', min_value=0, max_value=10000, value=1340)
sqft_basement = st.slider('Basement Area (sqft)', min_value=0, max_value=5000, value=0)
yr_built = st.slider('Year Built', min_value=1900, max_value=2022, value=1955)
yr_renovated = st.slider('Year Renovated', min_value=0, max_value=2022, value=2005)

# Button to make a prediction
if st.button("Predict House Price"):
    # Prepare the feature array for prediction
    features = [[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition,
                 sqft_above, sqft_basement, yr_built, yr_renovated]]
    
    # Convert features to DataFrame
    feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition',
                     'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
    features_df = pd.DataFrame(features, columns=feature_names)
    
    # Make the prediction
    predicted_price = loaded_model.predict(features_df)[0]

    # Display the predicted house price with improved UI
    st.markdown(
        """
        <div style="background-color: #80848c; padding: 20px; border-radius: 10px; text-align: center;">
            <h2 style="color: #ffffff;">Predicted House Price</h2>
            <p style="font-size: 24px; color: #0d6320;"><strong>${:,.2f}</strong></p>
        </div>
        """.format(predicted_price),
        unsafe_allow_html=True
    )

# Optionally add a footer or additional information
# st.markdown(
#     """
#     <hr>
#     <p style="text-align: center;">This prediction is based on the latest market data and trends.</p>
#     """
# )
