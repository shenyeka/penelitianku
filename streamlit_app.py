import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import statsmodels.api as sm
from sklearn.cluster import KMeans
from statsmodels.graphics.tsaplots import plot_pacf

# Fungsi Gaussian Membership
def gaussian_membership(x, c, sigma):
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)

# Fungsi untuk mengidentifikasi lag signifikan dari PACF
def identify_significant_lags(series, nlags=20, threshold=0.2):
    pacf_values = sm.tsa.pacf(series, nlags=nlags)
    significant_lags = [i for i, val in enumerate(pacf_values) if abs(val) > threshold and i != 0]
    return significant_lags

# Streamlit App
st.set_page_config(page_title="Prediksi Permintaan Darah dengan ARIMA-ANFIS + ABC", layout="wide")
st.title("Prediksi Permintaan Darah dengan Model Hybrid ARIMA-ANFIS + ABC")

menu = st.sidebar.radio("Menu", ["HOME", "DATA PREPROCESSING", "STASIONERITAS", "PREDIKSI"])

if menu == "HOME":
    st.header("Selamat Datang di Aplikasi Prediksi Permintaan Darah")
    st.write("Aplikasi ini menggunakan model hybrid ARIMA-ANFIS yang dioptimasi dengan Artificial Bee Colony (ABC) untuk memprediksi permintaan darah.")

elif menu == "DATA PREPROCESSING":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = [col.strip() for col in df.columns]  # bersihkan nama kolom
        if 'Bulan' in df.columns and 'Jumlah permintaan' in df.columns:
            df['Bulan'] = pd.to_datetime(df['Bulan'])
            df.set_index('Bulan', inplace=True)
            st.write("Data:", df)
            st.line_chart(df['Jumlah permintaan'])
            st.session_state['df'] = df
        else:
            st.warning("Kolom harus terdiri dari 'Bulan' dan 'Jumlah permintaan'")

elif menu == "STASIONERITAS":
    st.header("Uji Stasioneritas Data")
    if 'df' in st.session_state:
        df = st.session_state['df']
        st.subheader("Visualisasi Data")
        st.line_chart(df['Jumlah permintaan'])

        st.subheader("Uji ADF")
        result = sm.tsa.adfuller(df['Jumlah permintaan'])
        st.write(f"ADF Statistic: {result[0]}")
        st.write(f"p-value: {result[1]}")
        if result[1] < 0.05:
            st.success("Data sudah stasioner")
        else:
            st.error("Data belum stasioner")

        st.session_state['data_stasioner'] = df['Jumlah permintaan']

    else:
        st.warning("Harap upload data terlebih dahulu di menu 'DATA PREPROCESSING'")

elif menu == "PREDIKSI":
    st.header("Model Hybrid ARIMA-ANFIS + ABC")
    if 'data_stasioner' in st.session_state:
        data = st.session_state['data_stasioner']

        st.subheader("Split Data")
        split_ratio = st.slider("Rasio data latih", 0.6, 0.9, 0.8)
        split_point = int(len(data) * split_ratio)
        train, test = data[:split_point], data[split_point:]

        st.subheader("Model ARIMA")
        model_arima = sm.tsa.ARIMA(train, order=(1, 0, 0))
        model_fit = model_arima.fit()
        predictions_arima = model_fit.predict(start=test.index[0], end=test.index[-1])

        st.line_chart(pd.DataFrame({"Aktual": test, "Prediksi ARIMA": predictions_arima}))

        st.subheader("Residual ARIMA")
        residuals = train - model_fit.predict(start=train.index[0], end=train.index[-1])
        st.line_chart(residuals)

        st.subheader("Identifikasi Lag Signifikan untuk ANFIS")
        significant_lags = identify_significant_lags(residuals)
        st.write(f"Lag signifikan berdasarkan PACF: {significant_lags}")

        st.subheader("Persiapan Input ANFIS")
        data_anfis = pd.DataFrame({f"lag_{lag}": residuals.shift(lag) for lag in significant_lags})
        data_anfis['target'] = residuals.values
        data_anfis.dropna(inplace=True)
        st.write(data_anfis.head())

        st.subheader("Inisialisasi Parameter Fungsi Keanggotaan Gaussian dengan KMeans")
        n_cluster = st.slider("Jumlah cluster (jumlah fungsi keanggotaan per variabel)", 2, 6, 3)
        centers_dict = {}
        sigmas_dict = {}

        for col in data_anfis.columns[:-1]:  # exclude target
            kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(data_anfis[[col]])
            centers = kmeans.cluster_centers_.flatten()
            centers.sort()
            sigmas = [np.std(data_anfis[col] - c) for c in centers]
            centers_dict[col] = centers
            sigmas_dict[col] = sigmas

        st.write("Centers:", centers_dict)
        st.write("Sigmas:", sigmas_dict)

        st.success("Parameter fungsi keanggotaan Gaussian berhasil diinisialisasi!")

    else:
        st.warning("Harap lakukan preprocessing dan stasioneritas terlebih dahulu.")
