import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
from model import predict

st.title("Home Price Prediction")

st.write("---")

# Price of the house
price = st.number_input("Price of the house", min_value=0, step=1)

# Area of the house
area = st.number_input("Area of the house (in sq ft)", min_value=0, step=1)

# No. of bedrooms in the house
bedrooms = st.number_input("No. of bedrooms", min_value=0, step=1)

# No. of bathrooms in the house
bathrooms = st.number_input("No. of bathrooms", min_value=0, step=1)

# No. of stories
stories = st.number_input("No. of stories", min_value=0, step=1)

# Main road access
mainroad = st.radio("Main road access", ("yes", "no"))

# Guest room availability
guestroom = st.radio("Guest room availability", ("yes", "no"))

# Basement availability
basement = st.radio("Basement availability", ("yes", "no"))

# Hot water heating
hotwaterheating = st.radio("Hot water heating", ("yes", "no"))

# Air conditioning
airconditioning = st.radio("Air conditioning", ("yes", "no"))

# No. of parking spaces
parking = st.number_input("No. of parking spaces", min_value=0, step=1)

# Preferred area
prefarea = st.radio("Preferred area", ("yes", "no"))

# Furnishing status
furnishingstatus = st.radio(
    "Furnishing status", ("furnished", "semi-furnished", "unfurnished")
)

button = st.button("Predict Now")

# Display the values at the top
st.write("---")
st.write(f"**Price of the house:** {price}")
st.write(f"**Area of the house:** {area} sq ft")
st.write(f"**No. of bedrooms:** {bedrooms}")
st.write(f"**No. of bathrooms:** {bathrooms}")
st.write(f"**No. of stories:** {stories}")
st.write(f"**Main road access:** {mainroad}")
st.write(f"**Guest room availability:** {guestroom}")
st.write(f"**Basement availability:** {basement}")
st.write(f"**Hot water heating:** {hotwaterheating}")
st.write(f"**Air conditioning:** {airconditioning}")
st.write(f"**No. of parking spaces:** {parking}")
st.write(f"**Preferred area:** {prefarea}")
st.write(f"**Furnishing status:** {furnishingstatus}")

if button:
    cost = predict(np.array([[area, bedrooms, balconies, age]]))
    st.text(cost[0])
