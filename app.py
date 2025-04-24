import streamlit as st
import pandas as pd
import pickle

# Load the trained pipeline
model = pickle.load(open('car_price_prediction_model_streamlit.pkl', 'rb'))

# Load the dataset to extract company-model mappings
car = pd.read_csv("cleaned car.csv")
car = car[['name', 'company', 'year', 'Price', 'kms_driven', 'fuel_type']]
car = car.drop_duplicates().sort_values(by='company')

# Create company -> list of models dictionary
company_model_dict = car.groupby('company')['name'].unique().apply(list).to_dict()

# Streamlit UI
st.title("🚗 Car Price Prediction App")
st.write("Enter the car details below to get an estimated resale price.")

# Company selection
company = st.selectbox("Select Car Company", sorted(company_model_dict.keys()))

# Model selection based on selected company
model_options = company_model_dict[company]
name = st.selectbox("Select Car Model", sorted(model_options))

# Other inputs
year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2018)
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000)
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])

# Predict button
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'name': [name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    })

    # Predict
    predicted_price = model.predict(input_df)
    st.success(f"💰 Estimated Price: ₹ {int(predicted_price[0]):,}")
