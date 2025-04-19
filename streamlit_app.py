import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from pywt import wavedec, waverec

st.set_page_config(layout="centered")

# State untuk menyimpan data
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

st.title("Prediksi Permintaan Darah dengan ARIMA-ANFIS + ABC")

# Step 1: Selamat Datang
if st.session_state.step == 1:
    st.header("Selamat Datang")
    st.write("Aplikasi ini digunakan untuk memprediksi permintaan darah menggunakan model ARIMA yang dioptimasi ANFIS + ABC.")
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

# Step 5: Pemodelan ARIMA
elif st.session_state.step == 5:
    st.header("Pemodelan ARIMA")
    df = st.session_state.data
    model = ARIMA(df['Jumlah permintaan'], order=(1,1,1))
    fitted_model = model.fit()
    st.session_state.arima_model = fitted_model

    pred = fitted_model.predict(start=1, end=len(df)-1, typ='levels')
    actual = df['Jumlah permintaan'].iloc[1:]
    mape = mean_absolute_percentage_error(actual, pred)*100
    st.session_state.mape = mape
    st.line_chart(pred)
    st.write(f"Nilai MAPE: {mape:.2f}")

    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 4
    if col2.button("Lanjut"):
        st.session_state.step = 6

# Step 6: Residual ARIMA
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

# Step 7: Pemodelan ANFIS + ABC
elif st.session_state.step == 7:
    st.header("Pemodelan ANFIS + ABC")
    st.write("(Placeholder) Implementasi ANFIS + ABC di sini.")
    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 6
