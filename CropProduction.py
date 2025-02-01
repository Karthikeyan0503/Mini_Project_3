import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64

# Load Models and Preprocessor
model = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Function to add CSS styles
def add_css_styles(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Add CSS Styles
add_css_styles('style.css')

# Streamlit App
st.markdown(
    """
    <h1 style="color:black; font-weight:bold;">🌾 Farmer Crop Yield Predictor</h1>
    """,
    unsafe_allow_html=True,
)

def add_background(image_path):
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode('utf-8')
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Add background image
add_background("Crop.jpeg")

# Create a card-like structure
st.markdown(
    """
    <div class="card">
    """,
    unsafe_allow_html=True,
)

# Placeholder for displaying the predicted value
predicted_value_placeholder = st.empty()

with st.form("prediction_form"):
    # Create two columns inside the card
    col1, col2 = st.columns(2)

    # Column 1 Inputs
    with col1:
        Domain = st.text_input("Domain", value="Crops and livestock products")
        Area = st.text_input("Area")  # Standard input for Area
        Item = st.text_input("Item")  # Standard input for Item
        Year = st.number_input("Year", min_value=2000, max_value=2100, step=1)
        Area_harvested = st.number_input("Area harvested", min_value=0.0)
        Laying = st.number_input("Laying", min_value=0.0)

    # Column 2 Inputs
    with col2:
        Milk_Animals = st.number_input("Milk Animals", min_value=0.0)
        Producing_AnimalsorSlaughtered = st.number_input("Producing Animals or Slaughtered", min_value=0.0)
        Stocks = st.number_input("Stocks", min_value=0.0)
        Yield = st.number_input("Yield", min_value=0.0, format="%.2f")
        YieldorCarcass_Weight = st.number_input("Yield or Carcass Weight", min_value=0.0, format="%.2f")

    # Submit button inside the form
    submit_button = st.form_submit_button(label="Predict")

if submit_button:
    feature = np.array([[Domain, Area, Item, Year, Area_harvested, Laying, Milk_Animals,
                         Producing_AnimalsorSlaughtered, Stocks, Yield, YieldorCarcass_Weight]])
    transform_feature = preprocessor.transform(feature)
    predicted_value = model.predict(transform_feature).reshape(1, -1)[0][0]
    
    # Display predicted value above the form with green bold color
    predicted_value_placeholder.markdown(f"""
        <div style="background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; border-radius: 5px; text-align: center;">
            <h2 style="color: green; font-weight: bold;">Predicted Value: {predicted_value}</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown(
    """
    </div>
    """,
    unsafe_allow_html=True,
)


