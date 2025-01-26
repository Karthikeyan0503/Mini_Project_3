import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


file_path = 'filtered_data.csv' 
data = pd.read_csv(file_path)


st.title("Crop Production Prediction App")
st.write("### Dataset Preview")
st.dataframe(data.head())

st.sidebar.header("User Input Features")
selected_area = st.sidebar.selectbox("Select Area", data['Area'].unique())
selected_item = st.sidebar.selectbox("Select Crop (Item)", data['Item'].unique())
selected_year = st.sidebar.slider("Select Year", int(data['Year'].min()), int(data['Year'].max()), step=1)


filtered_data = data[
    (data['Area'] == selected_area) & (data['Item'] == selected_item)
]

if filtered_data.empty:
    st.write("No data available for the selected filters. Try another combination.")
else:
   
    X = filtered_data[['Year', 'Value']]
    y = filtered_data['Value']

 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    user_input = np.array([[selected_year, filtered_data['Value'].mean()]]).reshape(1, -2)
    prediction = model.predict(user_input)


    st.write("### Prediction Results")
    st.write(f"**Predicted Production**: {prediction[0]:.2f} tons")

    y_pred = model.predict(X_test)
    st.write("### Model Performance Metrics")
    st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

