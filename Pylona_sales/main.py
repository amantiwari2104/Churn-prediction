import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Load Data and Model ---
@st.cache_data
def load_data():
    df = pd.read_csv("engineered_dataset.csv", parse_dates=['Month'])
    df.set_index('Month', inplace=True)
    return df

@st.cache_resource
def load_trained_model():
    model = load_model("lstm_sales_model.h5")
    return model

# --- Load ---
df = load_data()
model = load_trained_model()

# --- Streamlit UI ---
st.title("Cosmetic Sales Forecast")
st.write("Predict next month's sales for different product-size combinations")

# Select product and size
product_options = sorted(set([col.split('_')[0] for col in df.columns]))
product = st.selectbox("Select a product", product_options)

# Get all size options for selected product
sizes = [col.split('_')[1] for col in df.columns if product in col]
size = st.selectbox("Select a size (ml)", sizes)

# Extract correct column
selected_column = f"{product}_{size}"
st.write(f"You selected: {selected_column}")

# --- Prepare input sequence ---
timesteps = 3
if len(df) < timesteps:
    st.warning("Not enough data to predict.")
    st.stop()

# Normalize the full dataset
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Get last 3 time steps for prediction
X_input = scaled_data[-timesteps:, :].reshape(1, timesteps, df.shape[1])

# Predict
predicted = model.predict(X_input)

# Inverse scale prediction
predicted_sales = scaler.inverse_transform(predicted)[0]

# Get predicted value for selected product-size
col_index = df.columns.get_loc(selected_column)
forecast_value = predicted_sales[col_index]

st.success(f"📈 Forecasted Sales for next month: **{int(forecast_value)} units** of {product} {size}ml")

# --- Show recent trend ---
st.subheader("Recent Sales Trend")
fig, ax = plt.subplots(figsize=(10, 4))
df[selected_column].plot(ax=ax, label="Actual")
ax.axhline(forecast_value, color='red', linestyle='--', label='Predicted')
ax.set_title(f"{product} {size}ml Sales")
ax.legend()
st.pyplot(fig)

st.caption("Model: LSTM, trained on 12 product-size combinations")