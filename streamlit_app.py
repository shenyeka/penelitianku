import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from streamlit.components.v1 import html

st.set_page_config(layout="centered")

# Style Settings
st.markdown("""
<style>
    body {
        background-color: #ffefef;
        color: maroon;
    }
    .header {
        color: maroon;
        font-size: 28px;
    }
    .subheader {
        color: maroon;
    }
    button {
        background-color: #f4c2c2;
        color: maroon;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    button:hover {
        background-color: #ebacac;
    }
</style>
""", unsafe_allow_html=True)

# State to store data
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'arima_model' not in st.session_state:
    st.session_state.arima_model = None
if 'residuals' not in st.session_state:
    st.session_state.residuals = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'mape' not in st.session_state:
    st.session_state.mape = None

# App Title
st.title("Prediksi Permintaan Darah")
st.markdown("Aplikasi ini digunakan untuk memprediksi permintaan darah menggunakan model ARIMA yang dioptimasi ANFIS + ABC.")

# Step 1: Welcome Screen
if st.session_state.step == 1:
    st.header("Selamat Datang")
    if st.button("Lanjut"):
        st.session_state.step = 2

# Step 2: Upload Dataset
elif st.session_state.step == 2:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Contoh data:")
        st.write(st.session_state.data.head())
    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 1
    if col2.button("Lanjut") and st.session_state.data is not None:
        st.session_state.step = 3

# Step 3: Preprocessing Data
elif st.session_state.step == 3:
    st.header("Preprocessing Data")
    df = st.session_state.data.copy()
    df['Bulan'] = pd.to_datetime(df['Bulan'])
    df = df.set_index('Bulan')
    st.session_state.data = df
    st.line_chart(df['Jumlah permintaan'])
    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 2
    if col2.button("Lanjut"):
        st.session_state.step = 4

# Step 4: Plot Data
elif st.session_state.step == 4:
    st.header("Plot Data")
    st.line_chart(st.session_state.data['Jumlah permintaan'])
    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 3
    if col2.button("Lanjut"):
        st.session_state.step = 5

# Step 5: ARIMA Modeling
elif st.session_state.step == 5:
    st.header("Pemodelan ARIMA")
    df = st.session_state.data
    model = ARIMA(df['Jumlah permintaan'], order=(1, 1, 1))
    fitted_model = model.fit()
    st.session_state.arima_model = fitted_model

    pred = fitted_model.predict(start=1, end=len(df)-1, typ='levels')
    actual = df['Jumlah permintaan'].iloc[1:]
    mape = mean_absolute_percentage_error(actual, pred) * 100
    st.session_state.mape = mape
    st.line_chart(pred)
    st.write(f"Nilai MAPE: {mape:.2f}")

    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 4
    if col2.button("Lanjut"):
        st.session_state.step = 6

# Step 6: ARIMA Residuals
elif st.session_state.step == 6:
    st.header("Residual ARIMA")
    residuals = st.session_state.arima_model.resid
    st.session_state.residuals = residuals
    st.line_chart(residuals)
    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 5
    if col2.button("Lanjut"):
        st.session_state.step = 7

# Step 7: ANFIS + ABC Modeling
elif st.session_state.step == 7:
    st.header("Pemodelan ANFIS + ABC")
    st.write("(Placeholder) Implementasi ANFIS + ABC di sini.")

    # Running Animation Placeholder
    st.markdown("""
    <div style='text-align: center;'>
        <img src="path_to_your_blood_splash_animation.gif" style="width:150px; height:auto;">
        <p style="font-size: 20px; color: maroon;">Running...</p>
    </div>
    """, unsafe_allow_html=True)

    # Add bee animation or placeholder
    st.markdown("""
    <div style='text-align: center;'>
        <img src="path_to_your_bee_animation.gif" style="width:150px; height:auto;">
        <p style="font-size: 20px; color: maroon;">Optimizing...</p>
    </div>
    """, unsafe_allow_html=True)
