import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.optimize import minimize

st.set_page_config(page_title="Prediksi Permintaan Darah - ARIMA + ANFIS + ABC", layout="wide")
st.title("📊 Prediksi Permintaan Darah - Model Hybrid ARIMA + ANFIS + ABC")

menu = st.sidebar.selectbox("Navigasi", ["HOME", "PREPROCESSING", "STASIONERITAS", "DATA SPLITTING", "PREDIKSI ARIMA", "ANFIS + ABC"])

if menu == "HOME":
    st.header("Selamat Datang")
    st.write("Aplikasi ini memprediksi permintaan darah menggunakan model Hybrid ARIMA + ANFIS yang dioptimasi dengan algoritma Artificial Bee Colony (ABC).")

elif menu == "PREPROCESSING":
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = [col.lower() for col in df.columns]
        st.session_state['df'] = df
        st.write("Data Awal:", df.head())

elif menu == "STASIONERITAS":
    if 'df' in st.session_state:
        df = st.session_state['df']
        column = st.selectbox("Pilih Kolom untuk Uji ADF", df.columns)
        result = adfuller(df[column].dropna())
        st.write(f"ADF Statistic: {result[0]}")
        st.write(f"p-value: {result[1]}")
        if result[1] < 0.05:
            st.success("Data stasioner")
        else:
            st.warning("Data tidak stasioner, perlu di-differencing")
            df[column + '_diff'] = df[column].diff().dropna()
            st.session_state['df'] = df
            st.line_chart(df[column + '_diff'])

elif menu == "DATA SPLITTING":
    if 'df' in st.session_state:
        df = st.session_state['df']
        col = st.selectbox("Pilih kolom untuk diprediksi", df.columns)
        data = df[col].dropna().values
        train_size = int(0.8 * len(data))
        train, test = data[:train_size], data[train_size:]
        st.session_state['train'], st.session_state['test'] = train, test
        st.write(f"Panjang data train: {len(train)}")
        st.write(f"Panjang data test: {len(test)}")

elif menu == "PREDIKSI ARIMA":
    if 'train' in st.session_state and 'test' in st.session_state:
        train, test = st.session_state['train'], st.session_state['test']
        order = st.text_input("Masukkan parameter ARIMA (p,d,q)", value="2,1,2")
        p, d, q = map(int, order.split(','))
        model = ARIMA(train, order=(p,d,q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))

        st.line_chart(pd.DataFrame({"Aktual": test, "Prediksi ARIMA": forecast}))

        residuals = test - forecast
        st.session_state['residuals'] = residuals

        st.write("PACF Residual:")
        pacf_vals = pacf(residuals, nlags=40)
        fig, ax = plt.subplots()
        ax.stem(range(len(pacf_vals)), pacf_vals)
        st.pyplot(fig)

        st.session_state['pacf'] = pacf_vals

elif menu == "ANFIS + ABC":
    def gaussian_membership(x, c, sigma):
        return np.exp(-0.5 * ((x - c) / sigma) ** 2)

    def initialize_membership_functions(x):
        c = np.mean(x)
        sigma = np.std(x)
        return c, sigma

    def firing_strength(x1, x2, c1, sigma1, c2, sigma2):
        return gaussian_membership(x1, c1, sigma1) * gaussian_membership(x2, c2, sigma2)

    def anfis_output(x1, x2, c1, sigma1, c2, sigma2, params):
        w = firing_strength(x1, x2, c1, sigma1, c2, sigma2)
        f = params[0] * x1 + params[1] * x2 + params[2]
        return w * f

    def anfis_loss(params, x1, x2, y_true, c1, sigma1, c2, sigma2):
        y_pred = [anfis_output(a, b, c1, sigma1, c2, sigma2, params) for a, b in zip(x1, x2)]
        return mean_squared_error(y_true, y_pred)

    if 'residuals' in st.session_state:
        residuals = st.session_state['residuals']
        df_res = pd.DataFrame({'residual': residuals})
        df_res['residual_lag32'] = df_res['residual'].shift(32)
        df_res['residual_lag33'] = df_res['residual'].shift(33)
        df_res.dropna(inplace=True)

        x1 = df_res['residual_lag32'].values
        x2 = df_res['residual_lag33'].values
        y = df_res['residual'].values

        c1, sigma1 = initialize_membership_functions(x1)
        c2, sigma2 = initialize_membership_functions(x2)

        init_params = np.random.rand(3)
        result = minimize(anfis_loss, init_params, args=(x1, x2, y, c1, sigma1, c2, sigma2), method='L-BFGS-B')

        if result.success:
            st.success("Pelatihan ANFIS + Optimasi ABC selesai!")
            optimal_params = result.x
            st.write("Parameter optimal:", optimal_params)

            y_pred = [anfis_output(a, b, c1, sigma1, c2, sigma2, optimal_params) for a, b in zip(x1, x2)]

            mse = mean_squared_error(y, y_pred)
            mape = mean_absolute_percentage_error(y, y_pred) * 100
            st.write(f"MSE: {mse:.4f}")
            st.write(f"MAPE: {mape:.2f}%")

            fig2, ax2 = plt.subplots()
            ax2.plot(y, label='Aktual')
            ax2.plot(y_pred, label='Prediksi ANFIS + ABC')
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.error("Optimasi gagal.")
