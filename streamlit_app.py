import streamlit as st
import pandas as pd
import math
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# === Halaman Navigasi ===
menu = st.sidebar.selectbox("Menu", [
    "Selamat Datang",
    "Upload Dataset",
    "Preprocessing Data",
    "Plot Data",
    "Pemodelan ARIMA",
    "Residual ARIMA",
    "ANFIS + ABC"
])

# === Selamat Datang ===
if menu == "Selamat Datang":
    st.title("Prediksi Permintaan Darah")
    st.write("Selamat datang di aplikasi prediksi permintaan darah menggunakan model ARIMA-ANFIS dengan optimasi Artificial Bee Colony.")

# === Upload Dataset ===
elif menu == "Upload Dataset":
    st.title("Upload Dataset")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Preview Data:")
        st.dataframe(data.head())
        st.session_state['data'] = data

# === Preprocessing Data ===
elif menu == "Preprocessing Data":
    st.title("Preprocessing Data")
    if 'data' in st.session_state:
        data = st.session_state['data']
        data['Bulan'] = pd.to_datetime(data['Bulan'])
        data = data.set_index('Bulan')
        st.write("Cek data null:")
        st.write(data.isnull().sum())
        st.session_state['data_processing'] = data
    else:
        st.warning("Silakan upload data terlebih dahulu.")

# === Plot Data ===
elif menu == "Plot Data":
    st.title("Plot Data Jumlah Permintaan Darah")
    if 'data_processing' in st.session_state:
        data_processing = st.session_state['data_processing']
        plt.figure(figsize=(10, 5))
        plt.plot(data_processing, label='Jumlah permintaan darah')
        plt.title('Data Jumlah Permintaan Darah 2011-2024')
        plt.xlabel('Tahun')
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu.")

# === Pemodelan ARIMA ===
elif menu == "Pemodelan ARIMA":
    st.title("Pemodelan ARIMA")
    if 'data_processing' in st.session_state:
        data = st.session_state['data_processing']
        train = data.iloc[:-39]
        test = data.iloc[-39:]

        from statsmodels.tsa.arima.model import ARIMA
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

        model_arima = ARIMA(train, order=(1, 1, 0)).fit()
        pred = model_arima.forecast(steps=len(test))
        test['Jumlah permintaan_pred'] = pred.values

        st.write("ARIMA Summary")
        st.text(model_arima.summary())

        mse = mean_squared_error(test['Jumlah permintaan'], test['Jumlah permintaan_pred'])
        mae = mean_absolute_error(test['Jumlah permintaan'], test['Jumlah permintaan_pred'])
        mape = mean_absolute_percentage_error(test['Jumlah permintaan'], test['Jumlah permintaan_pred']) * 100
        rmse = np.sqrt(mse)

        st.write(f"MSE: {mse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")

        st.session_state['model_arima'] = model_arima
        st.session_state['test'] = test
    else:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu.")

# === Residual ARIMA ===
elif menu == "Residual ARIMA":
    st.title("Residual ARIMA")
    if 'model_arima' in st.session_state:
        resid = st.session_state['model_arima'].resid
        st.line_chart(resid)
        st.session_state['residual'] = resid
    else:
        st.warning("Model ARIMA belum tersedia.")

# === ANFIS + ABC ===
elif menu == "ANFIS + ABC":
    st.title("ANFIS + Optimasi ABC")
    st.write("Bagian ini menampilkan hasil pemodelan ANFIS dengan optimasi menggunakan Artificial Bee Colony.")
    st.info("Implementasi lengkap ANFIS + ABC cukup panjang. Untuk kodenya dapat Anda lihat di file utama Python Anda.")
    st.code("""
# Contoh penggunaan:
best_params, best_loss = abc_optimizer(lag32, lag33, rules, target)
prediksi = predict_next_step(lag32_val, lag33_val)
    """)
