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
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import pacf
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import matplotlib.pyplot as plt

    st.title("PREDIKSI PERMINTAAN DARAH MENGGUNAKAN ARIMA")

    train = st.session_state.get('train_data')
    test = st.session_state.get('test_data')

    if train is not None and test is not None:
        st.subheader("1. Tentukan Parameter ARIMA (p,d,q)")
        p = st.number_input("Masukkan nilai p:", min_value=0, value=1)
        d = st.number_input("Masukkan nilai d:", min_value=0, value=1)
        q = st.number_input("Masukkan nilai q:", min_value=0, value=1)

        if st.button("Latih Model ARIMA"):
            model_arima = ARIMA(train, order=(p, d, q))
            model_arima = model_arima.fit()
            st.success("Model ARIMA berhasil dilatih.")
            st.write(model_arima.summary())

            start_test = len(train)
            pred = model_arima.forecast(steps=len(test))
            test['prediksi'] = pred.values

            st.subheader("4. Evaluasi Model dengan MAPE")
            mape = mean_absolute_percentage_error(test.iloc[:, 0], test['prediksi']) * 100
            st.write(f"MAPE ARIMA: {mape:.2f}%")

            st.line_chart({"Data Aktual": test.iloc[:, 0], "Prediksi ARIMA": test['prediksi']})

            # Simpan model & residual ke session_state
            st.session_state['model_arima'] = model_arima
            st.session_state['residual_arima'] = model_arima.resid

        # Jika model sudah ada, tampilkan tombol lanjutan
        if 'model_arima' in st.session_state:
            st.subheader("5. Residual ARIMA")

            if st.button("Lihat Residual ARIMA"):
                residual = st.session_state['residual_arima']
                st.line_chart(residual)

                # Simpan ke DataFrame untuk normalisasi
                data_anfis = pd.DataFrame({'residual': residual})
                st.session_state['data_anfis_raw'] = data_anfis

            if st.button("Lanjutkan ke Normalisasi Residual"):
                if 'data_anfis_raw' in st.session_state:
                    data_anfis = st.session_state['data_anfis_raw']
                    scaler_residual = MinMaxScaler()
                    data_anfis['residual'] = scaler_residual.fit_transform(data_anfis[['residual']])
                    st.session_state['data_anfis'] = data_anfis
                    st.session_state['scaler_residual'] = scaler_residual
                    st.success("Residual berhasil dinormalisasi.")
                    st.write(data_anfis.head())
                    st.info("Silakan lanjut ke menu ANFIS untuk melatih model hybrid.")
                else:
                    st.warning("Residual belum tersedia. Klik 'Lihat Residual ARIMA' terlebih dahulu.")

            # Langkah baru: Menentukan Input ANFIS dengan PACF
            if st.button("Tentukan Input ANFIS dari PACF"):
                if 'data_anfis' in st.session_state:
                    data_anfis = st.session_state['data_anfis']
                    
                    # Hitung PACF dan cari lag signifikan
                    jp = data_anfis['residual']
                    pacf_values = pacf(jp, nlags=33)
                    n = len(jp)  # jumlah data
                    ci = 1.96 / np.sqrt(n)  # Batas interval kepercayaan 95% untuk PACF
                    significant_lags = [i for i, val in enumerate(pacf_values) if abs(val) > ci and i != 0]
                    st.write(f"Lag signifikan (berdasarkan interval kepercayaan): {significant_lags}")

                    # Menambahkan lag signifikan ke data
                    for lag in significant_lags:
                        data_anfis[f'residual_lag{lag}'] = data_anfis['residual'].shift(lag)
                    data_anfis.dropna(inplace=True)

                    st.session_state['data_anfis_with_lags'] = data_anfis
                    st.success("Lag signifikan berhasil ditambahkan.")
                    st.write(data_anfis.head())

                    # Plot PACF
                    st.subheader("Plot Partial Autocorrelation Function (PACF)")
                    plt.figure(figsize=(10, 6))
                    plot_pacf(jp, lags=33, method='ywm', alpha=0.05)
                    plt.title('Partial Autocorrelation Function (PACF) residual ARIMA')
                    st.pyplot(plt)

# Tambahan: Memilih Target dan Input ANFIS
if 'data_anfis_with_lags' in st.session_state:
    st.subheader("6. Tentukan Target dan Input untuk ANFIS")

    data_anfis = st.session_state['data_anfis_with_lags']
    all_columns = list(data_anfis.columns)

    target_col = st.selectbox("Pilih kolom target:", all_columns, index=0)
    input_cols = st.multiselect("Pilih kolom input (lag signifikan):", [col for col in all_columns if col != target_col])

    if st.button("Simpan Dataset ANFIS"):
        if target_col and input_cols:
            X = data_anfis[input_cols].values
            y = data_anfis[target_col].values.reshape(-1, 1)

            st.session_state['X_anfis'] = X
            st.session_state['y_anfis'] = y

            st.success("✅ Dataset ANFIS berhasil disimpan.")
            st.write("Shape Input (X):", X.shape)
            st.write("Shape Target (y):", y.shape)

        else:
            st.warning("⚠️ Mohon pilih target dan minimal satu input untuk menyimpan dataset ANFIS.")

elif selected_menu == "7. Latih Model ANFIS + Optimasi ABC":
    st.subheader("7. Pelatihan ANFIS + Optimasi Artificial Bee Colony (ABC)")

    if 'X_anfis' in st.session_state and 'y_anfis' in st.session_state:
        X = st.session_state['X_anfis']
        y = st.session_state['y_anfis']

        # Konfigurasi parameter ABC
        st.markdown("### Konfigurasi Optimasi ABC")
        pop_size = st.number_input("Jumlah Lebah (populasi)", min_value=5, max_value=100, value=20)
        max_iter = st.number_input("Maksimum Iterasi", min_value=10, max_value=500, value=50)

        if st.button("🚀 Jalankan Pelatihan dan Optimasi"):
            st.write("⏳ Sedang melakukan pelatihan ANFIS dan optimasi ABC...")

            # ---------------------------
            # 1. Setup Fungsi Keanggotaan awal
            from sklearn.cluster import KMeans
            import numpy as np

            def initialize_gaussian_mf(X, n_mfs=2):
                kmeans = KMeans(n_clusters=n_mfs, random_state=0).fit(X)
                centers = kmeans.cluster_centers_
                stds = np.std(X, axis=0) / n_mfs
                return centers, stds

            centers, stds = initialize_gaussian_mf(X)

            # ---------------------------
            # 2. Definisikan Fungsi ANFIS Manual
            def gaussian(x, c, sigma):
                return np.exp(-0.5 * ((x - c) / sigma) ** 2)

            def anfis_predict(X, params):
                n_features = X.shape[1]
                n_rules = int(len(params) / (3 * n_features + 1))

                y_pred = []
                for x in X:
                    firing_strengths = []
                    outputs = []

                    for i in range(n_rules):
                        rule_params = params[i * (3 * n_features + 1):(i + 1) * (3 * n_features + 1)]
                        mf_params = rule_params[:2 * n_features]
                        linear_params = rule_params[2 * n_features:]

                        rule_strength = 1
                        for j in range(n_features):
                            c = mf_params[j]
                            sigma = mf_params[j + n_features]
                            rule_strength *= gaussian(x[j], c, sigma)

                        output = np.dot(linear_params[:-1], x) + linear_params[-1]
                        firing_strengths.append(rule_strength)
                        outputs.append(output)

                    firing_strengths = np.array(firing_strengths)
                    outputs = np.array(outputs)
                    if np.sum(firing_strengths) == 0:
                        y = 0
                    else:
                        y = np.dot(firing_strengths, outputs) / np.sum(firing_strengths)
                    y_pred.append(y)

                return np.array(y_pred).reshape(-1, 1)

            # ---------------------------
            # 3. Fungsi Evaluasi untuk ABC
            def evaluate_fitness(params):
                y_pred = anfis_predict(X, params)
                return np.mean((y - y_pred) ** 2)

            # ---------------------------
            # 4. Artificial Bee Colony (ABC)
            def abc_optimize(fitness_func, dim, lb, ub, pop_size=20, max_iter=100):
                population = np.random.uniform(lb, ub, (pop_size, dim))
                fitness = np.array([fitness_func(ind) for ind in population])
                trial = np.zeros(pop_size)

                for it in range(max_iter):
                    for i in range(pop_size):
                        k = np.random.randint(pop_size)
                        while k == i:
                            k = np.random.randint(pop_size)
                        phi = np.random.uniform(-1, 1, dim)
                        new_solution = population[i] + phi * (population[i] - population[k])
                        new_solution = np.clip(new_solution, lb, ub)
                        new_fitness = fitness_func(new_solution)

                        if new_fitness < fitness[i]:
                            population[i] = new_solution
                            fitness[i] = new_fitness
                            trial[i] = 0
                        else:
                            trial[i] += 1

                    # Scout bee
                    limit = 20
                    for i in range(pop_size):
                        if trial[i] > limit:
                            population[i] = np.random.uniform(lb, ub, dim)
                            fitness[i] = fitness_func(population[i])
                            trial[i] = 0

                    st.text(f"Iterasi {it + 1}/{max_iter} - Best MSE: {np.min(fitness):.4f}")

                best_index = np.argmin(fitness)
                return population[best_index], fitness[best_index]

            # ---------------------------
            # 5. Jalankan ABC
            n_features = X.shape[1]
            n_rules = 2  # bisa dikembangkan jadi opsi input user
            dim = n_rules * (3 * n_features + 1)
            lb = -1 * np.ones(dim)
            ub = 1 * np.ones(dim)

            best_params, best_mse = abc_optimize(
                fitness_func=evaluate_fitness,
                dim=dim,
                lb=lb,
                ub=ub,
                pop_size=pop_size,
                max_iter=max_iter
            )

            st.success(f"✅ Optimasi selesai. MSE (Train): {best_mse:.4f}")

            # ---------------------------
            # 6. Prediksi pada Data
            y_pred = anfis_predict(X, best_params)

            # ---------------------------
            # 7. Evaluasi dan Visualisasi
            from sklearn.metrics import mean_squared_error
            import matplotlib.pyplot as plt

            train_mse = mean_squared_error(y, y_pred)
            st.write(f"📉 MSE pada Data Training: {train_mse:.4f}")

            fig, ax = plt.subplots()
            ax.plot(y, label='Aktual')
            ax.plot(y_pred, label='Prediksi')
            ax.set_title("Hasil Prediksi ANFIS + ABC (Data Training)")
            ax.legend()
            st.pyplot(fig)

    else:
        st.warning("⚠️ Dataset ANFIS belum tersedia. Silakan simpan dulu di langkah 6.")
