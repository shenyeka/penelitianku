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
import numpy as np
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
        
        .stButton>button {
            background-color: #800000;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #A00000;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["HOME", "DATA PREPROCESSING", "STASIONERITAS DATA", "DATA SPLITTING", "PREDIKSI ARIMA", "ANFIS-ABC OPTIMIZATION"])

# ======================== HOME ========================
if menu == "HOME":
    st.markdown("<div class='header-container'>PREDIKSI PERMINTAAN DARAH<br>MENGGUNAKAN ARIMA-ANFIS DENGAN OPTIMASI ABC</div>", unsafe_allow_html=True)
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

# =================== PREDIKSI ARIMA ======================
elif menu == "PREDIKSI ARIMA":
    st.markdown("<div class='header-container'>PREDIKSI DENGAN ARIMA</div>", unsafe_allow_html=True)

    train = st.session_state.get('train_data')
    test = st.session_state.get('test_data')

    if train is not None and test is not None:
        st.subheader("1. Tentukan Parameter ARIMA (p,d,q)")
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("Masukkan nilai p:", min_value=0, value=1)
        with col2:
            d = st.number_input("Masukkan nilai d:", min_value=0, value=1)
        with col3:
            q = st.number_input("Masukkan nilai q:", min_value=0, value=1)

        if st.button("Latih Model ARIMA"):
            model_arima = ARIMA(train, order=(p, d, q))
            model_arima = model_arima.fit()
            st.success("Model ARIMA berhasil dilatih.")
            st.write(model_arima.summary())

            start_test = len(train)
            pred = model_arima.forecast(steps=len(test))
            test['prediksi'] = pred.values

            st.subheader("2. Evaluasi Model dengan MAPE")
            mape = mean_absolute_percentage_error(test.iloc[:, 0], test['prediksi']) * 100
            st.write(f"MAPE ARIMA: {mape:.2f}%")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(test.index, test.iloc[:, 0], label='Data Aktual')
            ax.plot(test.index, test['prediksi'], label='Prediksi ARIMA', linestyle='--')
            ax.set_title("Perbandingan Data Aktual dan Prediksi ARIMA")
            ax.legend()
            st.pyplot(fig)

            # Simpan model & residual ke session_state
            st.session_state['model_arima'] = model_arima
            st.session_state['residual_arima'] = model_arima.resid
            st.session_state['test_with_pred'] = test

        # Jika model sudah ada, tampilkan tombol lanjutan
        if 'model_arima' in st.session_state:
            st.subheader("3. Residual ARIMA")

            if st.button("Lihat Residual ARIMA"):
                residual = st.session_state['residual_arima']
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.plot(residual)
                ax1.set_title("Residual ARIMA")
                
                plot_acf(residual, lags=40, ax=ax2)
                ax2.set_title("ACF Residual")
                st.pyplot(fig)

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
                    st.info("Silakan lanjut ke menu ANFIS-ABC OPTIMIZATION untuk melatih model hybrid.")
                else:
                    st.warning("Residual belum tersedia. Klik 'Lihat Residual ARIMA' terlebih dahulu.")

# =================== ANFIS-ABC OPTIMIZATION ======================
elif menu == "ANFIS-ABC OPTIMIZATION":
    st.markdown("<div class='header-container'>ANFIS-ABC OPTIMIZATION</div>", unsafe_allow_html=True)
    
    if 'data_anfis' in st.session_state:
        data_anfis = st.session_state['data_anfis']
        
        st.subheader("1. Tentukan Input ANFIS dari PACF Residual")
        
        # Hitung PACF dan cari lag signifikan
        jp = data_anfis['residual']
        pacf_values = plot_pacf(jp, lags=33, method='ywm', alpha=0.05)
        st.pyplot()
        
        n = len(jp)  # jumlah data
        ci = 1.96 / np.sqrt(n)  # Batas interval kepercayaan 95% untuk PACF
        significant_lags = [i for i, val in enumerate(pacf_values) if abs(val) > ci and i != 0]
        st.write(f"Lag signifikan (berdasarkan interval kepercayaan): {significant_lags}")
        
        # Pilih lag yang akan digunakan
        selected_lags = st.multiselect("Pilih lag untuk input ANFIS:", 
                                     options=significant_lags,
                                     default=significant_lags[:2] if len(significant_lags) >= 2 else significant_lags)
        
        if len(selected_lags) < 2:
            st.warning("Pilih minimal 2 lag untuk input ANFIS")
        else:
            # Menambahkan lag terpilih ke data
            for lag in selected_lags:
                data_anfis[f'residual_lag{lag}'] = data_anfis['residual'].shift(lag)
            data_anfis.dropna(inplace=True)
            
            st.session_state['data_anfis_with_lags'] = data_anfis
            st.success("Lag signifikan berhasil ditambahkan.")
            st.write(data_anfis.head())
            
            # Pilih target dan input
            st.subheader("2. Konfigurasi ANFIS")
            all_columns = list(data_anfis.columns)
            
            target_col = st.selectbox("Pilih kolom target:", all_columns, index=0)
            input_cols = st.multiselect("Pilih kolom input:", 
                                      [col for col in all_columns if col != target_col],
                                      default=[col for col in all_columns if col != target_col][:2])
            
            if st.button("Persiapkan Data ANFIS"):
                if target_col and input_cols:
                    X = data_anfis[input_cols].values
                    y = data_anfis[target_col].values.reshape(-1, 1)
                    
                    st.session_state['X_anfis'] = X
                    st.session_state['y_anfis'] = y
                    st.session_state['input_cols'] = input_cols
                    st.session_state['target_col'] = target_col
                    
                    st.success("✅ Dataset ANFIS berhasil disiapkan.")
                    st.write("Shape Input (X):", X.shape)
                    st.write("Shape Target (y):", y.shape)
                    
                    # Visualisasi data input
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, col in enumerate(input_cols):
                        ax.plot(data_anfis[col], label=f'Input {i+1}: {col}')
                    ax.plot(data_anfis[target_col], label='Target', linewidth=2)
                    ax.legend()
                    ax.set_title("Visualisasi Input dan Target ANFIS")
                    st.pyplot(fig)
                else:
                    st.warning("Pilih target dan minimal satu input untuk menyimpan dataset ANFIS.")
    
    if 'X_anfis' in st.session_state and 'y_anfis' in st.session_state:
        st.subheader("3. Pelatihan ANFIS dengan Optimasi ABC")
        
        # Parameter ABC
        col1, col2, col3 = st.columns(3)
        with col1:
            num_food_sources = st.number_input("Jumlah Food Sources:", min_value=50, value=200)
        with col2:
            max_iter = st.number_input("Maksimum Iterasi:", min_value=100, value=1000)
        with col3:
            num_clusters = st.number_input("Jumlah Cluster MF:", min_value=2, max_value=5, value=3)
        
        if st.button("Mulai Pelatihan ANFIS-ABC"):
            X = st.session_state['X_anfis']
            y = st.session_state['y_anfis'].flatten()
            input_cols = st.session_state['input_cols']
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Inisialisasi fungsi keanggotaan dengan K-Means
            def initialize_membership_functions(data, num_clusters=3):
                kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(data.reshape(-1, 1))
                centers = np.sort(kmeans.cluster_centers_.flatten())
                sigma = (centers[1] - centers[0]) / 2
                return centers, sigma
            
            # Fungsi keanggotaan Gaussian
            @jit(nopython=True)
            def gaussian_membership(x, c, sigma):
                return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
            
            # Hitung firing strength
            @jit(nopython=True)
            def calculate_firing_strength(X, centers_list, sigma_list):
                n_samples = X.shape[0]
                n_inputs = X.shape[1]
                n_rules = len(centers_list[0])  # Jumlah MF per input
                
                # Hitung semua nilai keanggotaan
                membership_values = np.ones((n_samples, n_inputs, n_rules))
                for i in range(n_inputs):
                    for j in range(n_rules):
                        membership_values[:, i, j] = gaussian_membership(
                            X[:, i], centers_list[i][j], sigma_list[i]
                        )
                
                # Hitung firing strength (product rule)
                rules = np.ones((n_samples, n_rules**n_inputs))
                for s in range(n_samples):
                    # Buat grid untuk semua kombinasi aturan
                    grid = np.meshgrid(*[range(n_rules) for _ in range(n_inputs)])
                    combinations = np.vstack([g.ravel() for g in grid]).T
                    
                    for c, comb in enumerate(combinations):
                        firing = 1.0
                        for i, mf_idx in enumerate(comb):
                            firing *= membership_values[s, i, mf_idx]
                        rules[s, c] = firing
                
                # Normalisasi firing strength
                rules = rules / (rules.sum(axis=1, keepdims=True) + 1e-8)
                return rules
            
            # Prediksi ANFIS
            @jit(nopython=True)
            def anfis_predict(params, X, rules):
                n_samples = X.shape[0]
                n_inputs = X.shape[1]
                n_rules = rules.shape[1]
                
                # Split parameter
                p = params[:n_inputs*n_rules].reshape(n_inputs, n_rules)
                r = params[n_inputs*n_rules:]
                
                outputs = np.zeros(n_samples)
                
                for i in range(n_samples):
                    weighted_sum = 0.0
                    rule_sum = 0.0
                    
                    for j in range(n_rules):
                        # Hitung output linear untuk setiap rule
                        linear_output = r[j]
                        for k in range(n_inputs):
                            linear_output += p[k, j] * X[i, k]
                        
                        # Tambahkan ke output dengan bobot firing strength
                        weighted_sum += rules[i, j] * linear_output
                        rule_sum += rules[i, j]
                    
                    outputs[i] = weighted_sum / (rule_sum + 1e-8)
                
                return outputs
            
            # Fungsi loss (MSE)
            @jit(nopython=True)
            def loss_function(params, X, y, rules):
                preds = anfis_predict(params, X, rules)
                error = np.mean((y - preds)**2)
                return error
            
            # Inisialisasi ANFIS
            centers_list = []
            sigma_list = []
            
            for i in range(X.shape[1]):
                centers, sigma = initialize_membership_functions(X[:, i], num_clusters)
                centers_list.append(centers)
                sigma_list.append(sigma)
            
            rules = calculate_firing_strength(X, centers_list, sigma_list)
            
            # Inisialisasi parameter dengan Linear Regression
            X_reg = np.zeros((X.shape[0], rules.shape[1]*(X.shape[1]+1)))
            for i in range(rules.shape[1]):
                for j in range(X.shape[1]):
                    X_reg[:, i*(X.shape[1]+1)+j] = rules[:, i] * X[:, j]
                X_reg[:, i*(X.shape[1]+1)+X.shape[1]] = rules[:, i]
            
            lin_reg = LinearRegression().fit(X_reg, y)
            params_initial = np.concatenate([
                lin_reg.coef_[:X.shape[1]*rules.shape[1]],
                lin_reg.coef_[X.shape[1]*rules.shape[1]:]
            ])
            
            # ABC Optimizer
            def abc_optimizer(loss_func, initial_params, bounds, 
                            num_food_sources=50, max_iter=1000, limit=100):
                
                n_params = len(initial_params)
                food_sources = np.zeros((num_food_sources, n_params))
                fitness = np.zeros(num_food_sources)
                trial = np.zeros(num_food_sources, dtype=np.int32)
                
                # Inisialisasi food sources
                for i in range(num_food_sources):
                    food_sources[i] = initial_params + np.random.uniform(-0.1, 0.1, n_params)
                    fitness[i] = loss_func(food_sources[i])
                
                best_idx = np.argmin(fitness)
                best_fitness = fitness[best_idx]
                best_params = food_sources[best_idx].copy()
                
                for iteration in range(max_iter):
                    # Employed bees phase
                    for i in range(num_food_sources):
                        # Pilih solusi tetangga secara acak
                        k = np.random.randint(0, num_food_sources)
                        while k == i:
                            k = np.random.randint(0, num_food_sources)
                        
                        # Hasilkan solusi baru
                        phi = np.random.uniform(-1, 1, n_params)
                        new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                        new_solution = np.clip(new_solution, bounds[0], bounds[1])
                        
                        # Evaluasi solusi baru
                        new_fitness = loss_func(new_solution)
                        
                        # Seleksi greedy
                        if new_fitness < fitness[i]:
                            food_sources[i] = new_solution
                            fitness[i] = new_fitness
                            trial[i] = 0
                            
                            if new_fitness < best_fitness:
                                best_fitness = new_fitness
                                best_params = new_solution.copy()
                        else:
                            trial[i] += 1
                    
                    # Onlooker bees phase
                    fitness_inv = 1.0 / (1.0 + fitness)
                    probs = fitness_inv / np.sum(fitness_inv)
                    
                    for _ in range(num_food_sources):
                        # Pilih food source berdasarkan probabilitas
                        i = np.random.choice(num_food_sources, p=probs)
                        
                        # Pilih solusi tetangga
                        k = np.random.randint(0, num_food_sources)
                        while k == i:
                            k = np.random.randint(0, num_food_sources)
                        
                        # Hasilkan solusi baru
                        phi = np.random.uniform(-1, 1, n_params)
                        new_solution = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                        new_solution = np.clip(new_solution, bounds[0], bounds[1])
                        
                        # Evaluasi solusi baru
                        new_fitness = loss_func(new_solution)
                        
                        # Seleksi greedy
                        if new_fitness < fitness[i]:
                            food_sources[i] = new_solution
                            fitness[i] = new_fitness
                            trial[i] = 0
                            
                            if new_fitness < best_fitness:
                                best_fitness = new_fitness
                                best_params = new_solution.copy()
                        else:
                            trial[i] += 1
                    
                    # Scout bees phase
                    for i in range(num_food_sources):
                        if trial[i] > limit:
                            food_sources[i] = np.random.uniform(bounds[0], bounds[1], n_params)
                            fitness[i] = loss_func(food_sources[i])
                            trial[i] = 0
                            
                            if fitness[i] < best_fitness:
                                best_fitness = fitness[i]
                                best_params = food_sources[i].copy()
                    
                    # Update progress
                    progress = (iteration + 1) / max_iter
                    progress_bar.progress(progress)
                    status_text.text(f"Iterasi {iteration+1}/{max_iter} - Best MSE: {best_fitness:.6f}")
                
                return best_params, best_fitness
            
            # Definisikan fungsi loss untuk ABC
            def abc_loss(params):
                return loss_function(params, X, y, rules)
            
            # Jalankan optimasi ABC
            bounds = (-1, 1)  # Batas parameter
            best_params, best_mse = abc_optimizer(
                abc_loss, params_initial, bounds, 
                num_food_sources=num_food_sources, 
                max_iter=max_iter
            )
            
            # Simpan hasil optimasi
            st.session_state['anfis_params'] = best_params
            st.session_state['anfis_mse'] = best_mse
            st.session_state['anfis_rules'] = rules
            
            st.success(f"Optimasi ABC selesai! Best MSE: {best_mse:.6f}")
            
            # Prediksi dengan parameter terbaik
            y_pred = anfis_predict(best_params, X, rules)
            
            # Plot hasil prediksi
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(y, label='Target')
            ax.plot(y_pred, label='Prediksi ANFIS-ABC', linestyle='--')
            ax.legend()
            ax.set_title("Perbandingan Target dan Prediksi ANFIS-ABC")
            st.pyplot(fig)
            
            # Hitung RMSE dan MAPE
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            
            st.subheader("4. Evaluasi Model ANFIS-ABC")
            col1, col2 = st.columns(2)
            col1.metric("RMSE", f"{rmse:.4f}")
            col2.metric("MAPE", f"{mape:.2f}%")
            
            # Gabungkan dengan prediksi ARIMA jika ada
            if 'test_with_pred' in st.session_state:
                test_data = st.session_state['test_with_pred']
                arima_pred = test_data['prediksi'].values
                
                # Normalisasi data testing
                scaler = st.session_state['scaler_residual']
                test_residual = test_data.iloc[:, 0] - arima_pred
                test_residual_norm = scaler.transform(test_residual.values.reshape(-1, 1)).flatten()
                
                # Siapkan input ANFIS untuk data testing
                X_test = np.zeros((len(test_residual_norm)-max(selected_lags), len(selected_lags)))
                for i, lag in enumerate(selected_lags):
                    X_test[:, i] = test_residual_norm[lag:len(test_residual_norm)-(max(selected_lags)-lag)]
                
                # Hitung firing strength untuk data testing
                rules_test = calculate_firing_strength(X_test, centers_list, sigma_list)
                
                # Prediksi residual dengan ANFIS
                residual_pred_norm = anfis_predict(best_params, X_test, rules_test)
                residual_pred = scaler.inverse_transform(residual_pred_norm.reshape(-1, 1)).flatten()
                
                # Gabungkan prediksi ARIMA + ANFIS
                hybrid_pred = arima_pred[max(selected_lags):] + residual_pred
                
                # Evaluasi hybrid model
                actual_values = test_data.iloc[max(selected_lags):, 0]
                hybrid_mape = mean_absolute_percentage_error(actual_values, hybrid_pred) * 100
                
                st.subheader("5. Evaluasi Model Hybrid ARIMA-ANFIS")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(test_data.index, test_data.iloc[:, 0], label='Aktual')
                ax.plot(test_data.index, test_data['prediksi'], label='ARIMA', linestyle='--')
                ax.plot(test_data.index[max(selected_lags):], hybrid_pred, label='Hybrid ARIMA-ANFIS', linestyle='-.')
                ax.legend()
                ax.set_title("Perbandingan Model ARIMA dan Hybrid ARIMA-ANFIS")
                st.pyplot(fig)
                
                col1, col2 = st.columns(2)
                col1.metric("MAPE ARIMA", f"{mape:.2f}%")
                col2.metric("MAPE Hybrid ARIMA-ANFIS", f"{hybrid_mape:.2f}%", 
                            delta=f"{(mape - hybrid_mape):.2f}% improvement" if hybrid_mape < mape else "")
                
                st.session_state['hybrid_pred'] = hybrid_pred
                st.session_state['hybrid_mape'] = hybrid_mape
    else:
        st.warning("Silakan siapkan data ANFIS terlebih dahulu di menu sebelumnya.")
