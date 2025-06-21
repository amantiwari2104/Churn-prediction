import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

from tensorflow.keras.models import load_model

model = load_model('model.keras', compile=False)


with open('laber_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('preprocessor.pkl','rb') as file:
    preprocessor = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

onehot_encoder_geo = preprocessor.named_transformers_['OneHotEncoder']

##streamlit app
st.title('Customer churn prediction')

#user input 
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}) 


geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Reset index to align properly
input_data = input_data.reset_index(drop=True)

# Place the one-hot encoded columns at the front
input_data = pd.concat([geo_encoded_df, input_data], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')