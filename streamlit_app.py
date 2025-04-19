import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from numba import jit
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    layout="wide"
)

# Styling CSS
st.markdown("""
    <style>
        body {
            background-color: #FADADD;
            font-family: 'Arial', sans-serif;
            font-size: 18px;
        }

        .header-container {
            background: #800000;
            color: white;
            padding: 40px;
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            transform: scale(1.05);
            transition: transform 0.3s ease;
        }

        .header-container:hover {
            transform: scale(1.1);
        }

        .content {
            text-align: justify;
            font-size: 20px;
            line-height: 1.8;
            margin: 20px 10%;
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["HOME", "DATA PREPROCESSING", "STASIONERITAS DATA", "DATA SPLITTING", "PREDIKSI"])

# ======================== HOME ========================
if menu == "HOME":
    st.markdown("<div class='header-container'>PREDIKSI PERMINTAAN DARAH<br>MENGGUNAKAN MENGGUNAKAN ARIMA-ANFIS DENGAN OPTIMASI ABC</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content">
    Antarmuka ini menggunakan metode hybrid <b>ARIMA-ANFIS</b> dengan optimasi <b>Artificial Bee Colony</b> (ABC)
    untuk memprediksi permintaan darah pada Unit Transfusi Darah (UTD).<br><br>
    Silakan mulai dengan mengunggah data pada menu <b>DATA PREPROCESSING</b>.
    </div>
    """, unsafe_allow_html=True)

# ==================== DATA PREPROCESSING ====================
elif menu == "DATA PREPROCESSING":
    st.markdown("<div class='header-container'>DATA PREPROCESSING</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview Data:")
        st.write(data.head())

        # Pilih kolom waktu
        time_col = st.selectbox("Pilih Kolom Waktu sebagai Index", options=data.columns)
        if time_col:
            data[time_col] = pd.to_datetime(data[time_col])
            data.set_index(time_col, inplace=True)
            st.write("Data Setelah Menetapkan Index Waktu:")
            st.write(data.head())

            # Tangani missing values
            missing = data.isnull().sum()
            if missing.any():
                st.warning("Data memiliki missing values. Menghapus baris dengan nilai kosong.")
                data.dropna(inplace=True)

            # Tampilkan plot
            st.write("Plot Data Setelah Preprocessing:")
            fig, ax = plt.subplots()
            sns.lineplot(data=data, ax=ax)
            ax.set_title("Data Time Series")
            st.pyplot(fig)

            # Simpan data ke session_state
            st.session_state["data"] = data

            st.success("Preprocessing selesai. Silakan lanjut ke menu 'STASIONERITAS DATA'.")

# ================== STASIONERITAS DATA =====================
elif menu == "STASIONERITAS DATA":
    st.markdown("<div class='header-container'>STASIONERITAS DATA</div>", unsafe_allow_html=True)

    if "data" in st.session_state:
        data = st.session_state["data"]
        st.write("Menggunakan data hasil preprocessing.")

        col = st.selectbox("Pilih kolom untuk diuji stasioneritas:", data.columns)

        if col:
            # Uji ADF awal
            st.subheader("Uji ADF - Sebelum Differencing")
            adf_result = adfuller(data[col])
            st.write(f"ADF Statistic: {adf_result[0]:.4f}")
            st.write(f"P-Value: {adf_result[1]:.4f}")
            for key, val in adf_result[4].items():
                st.write(f"Critical Value ({key}): {val:.4f}")

            if adf_result[1] < 0.05:
                st.success("✅ Data sudah stasioner.")
            else:
                st.warning("⚠️ Data tidak stasioner. Melakukan differencing...")

                # Differencing
                data_diff = data[col].diff().dropna()

                # Simpan data differencing ke session
                st.session_state["data_diff"] = data_diff

                # Plot hasil differencing
                st.subheader("Plot Setelah Differencing:")
                fig, ax = plt.subplots()
                sns.lineplot(x=data_diff.index, y=data_diff.values, ax=ax)
                ax.set_title("Data Setelah Differencing")
                st.pyplot(fig)

                # Uji ADF ulang setelah differencing
                st.subheader("Uji ADF - Setelah Differencing")
                adf_diff_result = adfuller(data_diff)
                st.write(f"ADF Statistic: {adf_diff_result[0]:.4f}")
                st.write(f"P-Value: {adf_diff_result[1]:.4f}")
                for key, val in adf_diff_result[4].items():
                    st.write(f"Critical Value ({key}): {val:.4f}")

                if adf_diff_result[1] < 0.05:
                    st.success("✅ Data sudah stasioner setelah differencing.")
                else:
                    st.error("❌ Data masih belum stasioner setelah differencing.")

            # Plot ACF dan PACF
            st.subheader("Plot ACF dan PACF:")
            fig_acf, ax_acf = plt.subplots()
            plot_acf(data[col].dropna(), lags=40, ax=ax_acf)
            st.pyplot(fig_acf)

            fig_pacf, ax_pacf = plt.subplots()
            plot_pacf(data[col].dropna(), lags=40, ax=ax_pacf)
            st.pyplot(fig_pacf)

    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu di menu 'DATA PREPROCESSING'.")

# =================== DATA SPLITTING ===================
elif menu == "DATA SPLITTING":
    st.markdown("<div class='header-container'>DATA SPLITTING</div>", unsafe_allow_html=True)

    uploaded_split_file = st.file_uploader("Unggah Data yang Akan Di-Split (CSV)", type=["csv"])

    if uploaded_split_file is not None:
        df = pd.read_csv(uploaded_split_file)

        st.write("Preview Data:")
        st.write(df.head())

        time_column = st.selectbox("Pilih Kolom Waktu (jika ada)", ["Tidak Ada"] + list(df.columns))

        if time_column != "Tidak Ada":
            df[time_column] = pd.to_datetime(df[time_column])
            df.set_index(time_column, inplace=True)

        if len(df.columns) == 1:
            col_name = df.columns[0]

            train_size = int(len(df) * 0.8)
            train_data = df.iloc[:train_size]
            test_data = df.iloc[train_size:]

            st.session_state["train_data"] = train_data
            st.session_state["test_data"] = test_data

            st.success("✅ Data berhasil di-split dengan rasio 80% training dan 20% testing.")

            st.subheader("Data Training:")
            st.write(train_data.tail())
            st.line_chart(train_data)

            st.subheader("Data Testing:")
            st.write(test_data.head())
            st.line_chart(test_data)

        else:
            st.warning("⚠️ Data harus hanya memiliki 1 kolom target untuk proses split time series.")

    else:
        st.info("Silakan unggah data yang ingin Anda split.")

# =================== PREDIKSI ======================
elif menu == "PREDIKSI":

    train = st.session_state.get('train_data')
    test = st.session_state.get('test_data')

    if train is not None and test is not None:
        st.subheader("1. Parameter ARIMA")
        p = st.number_input("p", min_value=0, value=1)
        d = st.number_input("d", min_value=0, value=1)
        q = st.number_input("q", min_value=0, value=1)

        if st.button("Latih Model ARIMA"):
            series = train.iloc[:, 0]
            model = ARIMA(series, order=(p, d, q)).fit()
            st.session_state['model_arima'] = model
            st.success("Model ARIMA berhasil dilatih")

            preds = model.forecast(steps=len(test))
            test = test.copy()
            test['prediksi'] = preds.values
            st.write("MAPE:", mean_absolute_percentage_error(test.iloc[:, 0], test['prediksi']) * 100)
            st.line_chart({"Aktual": test.iloc[:, 0], "Prediksi": test['prediksi']})

            st.session_state['residual_arima'] = model.resid

    if 'residual_arima' in st.session_state:
        residual = st.session_state['residual_arima']
        st.line_chart(residual)

        df = pd.DataFrame({'residual': residual})
        df.dropna(inplace=True)
        scaler = MinMaxScaler()
        df['residual'] = scaler.fit_transform(df[['residual']])

        pacf_values = pacf(df['residual'], nlags=33)
        ci = 1.96 / np.sqrt(len(df))
        significant_lags = [i for i, val in enumerate(pacf_values) if abs(val) > ci and i != 0]
        st.write("Lag signifikan:", significant_lags)

        for lag in significant_lags:
            df[f'lag{lag}'] = df['residual'].shift(lag)
        df.dropna(inplace=True)

        st.session_state['data_anfis'] = df
        st.write(df.head())

        plt.figure(figsize=(10, 6))
        plot_pacf(df['residual'], lags=33)
        st.pyplot(plt)

        def initialize_membership_functions(data, num_clusters=4):
            kmeans = KMeans(n_clusters=num_clusters).fit(data.reshape(-1, 1))
            centers = np.sort(kmeans.cluster_centers_.flatten())
            sigma = (centers[1] - centers[0]) / 2
            return centers, sigma

        def gaussian_membership(x, c, sigma):
            return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

        def firing_strength(x1, x2, c1, s1, c2, s2):
            x1_low = gaussian_membership(x1, c1[0], s1)
            x1_high = gaussian_membership(x1, c1[1], s1)
            x2_low = gaussian_membership(x2, c2[0], s2)
            x2_high = gaussian_membership(x2, c2[1], s2)

            return np.array([
                x1_low * x2_low,
                x1_low * x2_high,
                x1_high * x2_low,
                x1_high * x2_high
            ]).T

        @jit(nopython=True)
        def anfis_predict(params, x1, x2, rules):
            n_samples = x1.shape[0]
            n_rules = rules.shape[1]
            p = params[:n_rules]
            q = params[n_rules:2*n_rules]
            r = params[2*n_rules:3*n_rules]
            out = np.zeros(n_samples)
            for i in range(n_samples):
                num, denom = 0.0, 0.0
                for j in range(n_rules):
                    rule = rules[i, j]
                    num += rule * (p[j] * x1[i] + q[j] * x2[i] + r[j])
                    denom += rule
                out[i] = num / (denom + 1e-8)
            return out

        @jit(nopython=True)
        def loss_fn(params, x1, x2, rules, y):
            preds = anfis_predict(params, x1, x2, rules)
            return np.mean((y - preds)**2)

        def abc_optimizer(x1, x2, rules, y, n_food=200, n_onlooker=200, iters=1500, limit=150):
            n_params = 3 * rules.shape[1]
            foods = np.random.uniform(-1, 1, (n_food, n_params))
            fit = np.array([loss_fn(f, x1, x2, rules, y) for f in foods])
            trial = np.zeros(n_food)
            best = foods[np.argmin(fit)]
            best_fit = np.min(fit)
            for t in range(iters):
                for i in range(n_food):
                    k = np.random.randint(n_food)
                    while k == i:
                        k = np.random.randint(n_food)
                    phi = np.random.uniform(-1, 1, n_params)
                    candidate = foods[i] + phi * (foods[i] - foods[k])
                    candidate = np.clip(candidate, -1, 1)
                    f_candidate = loss_fn(candidate, x1, x2, rules, y)
                    if f_candidate < fit[i]:
                        foods[i] = candidate
                        fit[i] = f_candidate
                        trial[i] = 0
                        if f_candidate < best_fit:
                            best = candidate
                            best_fit = f_candidate
                    else:
                        trial[i] += 1
                    if trial[i] > limit:
                        foods[i] = np.random.uniform(-1, 1, n_params)
                        fit[i] = loss_fn(foods[i], x1, x2, rules, y)
                        trial[i] = 0
            return best, best_fit

        if st.button("Latih Model ANFIS"):
            df = st.session_state['data_anfis']
            x1 = df[f'lag{significant_lags[-2]}'].values
            x2 = df[f'lag{significant_lags[-1]}'].values
            y = df['residual'].values

            c1, s1 = initialize_membership_functions(x1)
            c2, s2 = initialize_membership_functions(x2)
            rules = firing_strength(x1, x2, c1, s1, c2, s2)
            best_params, loss = abc_optimizer(x1, x2, rules, y)

            st.write("Loss ANFIS:", loss)
            preds = anfis_predict(best_params, x1, x2, rules)
            st.line_chart({"Actual": y, "Predicted": preds})
