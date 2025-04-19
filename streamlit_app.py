import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import pacf
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from numba import jit
import warnings
warnings.filterwarnings('ignore')

# Judul Aplikasi
st.title('Sistem Prediksi Permintaan Darah')
st.subheader('Menggunakan ARIMA-ANFIS dengan Optimasi ABC')

# Inisialisasi session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = None
if 'data_stationary' not in st.session_state:
    st.session_state.data_stationary = None
if 'model_arima' not in st.session_state:
    st.session_state.model_arima = None
if 'arima_results' not in st.session_state:
    st.session_state.arima_results = None
if 'data_anfis' not in st.session_state:
    st.session_state.data_anfis = None
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'best_loss' not in st.session_state:
    st.session_state.best_loss = None

# Fungsi untuk navigasi
def next_step():
    st.session_state.current_step += 1

def prev_step():
    if st.session_state.current_step > 1:
        st.session_state.current_step -= 1

# Fungsi untuk menghitung metrik
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(mse)
    return mse, mae, mape, rmse

# Step 1: Upload Data
if st.session_state.current_step == 1:
    st.header("Step 1: Upload Dataset")
    
    uploaded_file = st.file_uploader("Upload file CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success('Data berhasil diupload!')
            
            st.write("Preview data:")
            st.dataframe(data.head())
            
            # Pilih kolom yang akan digunakan
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                date_column = st.selectbox('Pilih kolom tanggal', date_cols if date_cols else [None])
            with col2:
                value_column = st.selectbox('Pilih kolom nilai', numeric_cols)
            
            if date_column and value_column:
                st.session_state.data_preprocessed = data.set_index(date_column)[[value_column]]
                st.session_state.data_preprocessed.columns = ['Jumlah permintaan']
                
                st.write("Data yang akan diproses:")
                st.dataframe(st.session_state.data_preprocessed.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Lanjut ke Step 2", on_click=next_step):
                        pass
                with col2:
                    st.button("Reset", on_click=lambda: st.session_state.clear())

# Step 2: Preprocessing Data
elif st.session_state.current_step == 2 and st.session_state.data_preprocessed is not None:
    st.header("Step 2: Preprocessing Data")
    data = st.session_state.data_preprocessed
    
    st.write("Data sebelum preprocessing:")
    st.dataframe(data.head())
    
    # Cek missing values
    st.subheader("Cek Missing Values")
    st.write(data.isnull().sum())
    
    # Handle missing values
    if data.isnull().sum().any():
        handle_missing = st.selectbox("Metode handling missing values:", 
                                    ["Drop NA", "Fill with Mean", "Fill with Median"])
        
        if st.button("Proses Data"):
            if handle_missing == "Drop NA":
                data = data.dropna()
            elif handle_missing == "Fill with Mean":
                data = data.fillna(data.mean())
            else:
                data = data.fillna(data.median())
            
            st.session_state.data_preprocessed = data
            st.success("Preprocessing selesai!")
            
            st.write("Data setelah preprocessing:")
            st.dataframe(data.head())
            
            # Visualisasi data
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data, label='Jumlah permintaan darah')
            ax.set_title('Data Jumlah Permintaan Darah')
            ax.legend()
            st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Kembali ke Step 1", on_click=prev_step)
    with col2:
        if st.button("Lanjut ke Step 3", on_click=next_step) and not data.isnull().sum().any():
            pass
    with col3:
        st.button("Reset", on_click=lambda: st.session_state.clear())

# Step 3: Uji Stasioneritas
elif st.session_state.current_step == 3 and st.session_state.data_preprocessed is not None:
    st.header("Step 3: Uji Stasioneritas")
    data = st.session_state.data_preprocessed
    
    st.write("Data yang digunakan:")
    st.dataframe(data.head())
    
    if st.button("Lakukan ADF Test"):
        result = adfuller(data['Jumlah permintaan'])
        
        st.write("Hasil ADF Test:")
        st.write(f"ADF Statistic: {result[0]:.4f}")
        st.write(f"p-value: {result[1]:.4f}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"   {key}: {value:.4f}")
        
        if result[1] > 0.05:
            st.warning("Data tidak stasioner (p-value > 0.05)")
            st.session_state.data_stationary = None
        else:
            st.success("Data stasioner (p-value ≤ 0.05)")
            st.session_state.data_stationary = data.copy()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Kembali ke Step 2", on_click=prev_step)
    with col2:
        if st.session_state.get('data_stationary') is not None:
            if st.button("Lanjut ke Step 4", on_click=next_step):
                pass
    with col3:
        st.button("Reset", on_click=lambda: st.session_state.clear())

# Step 4: Differencing (jika diperlukan)
elif st.session_state.current_step == 4:
    st.header("Step 4: Differencing Data")
    
    if st.session_state.data_stationary is None:
        st.warning("Data belum stasioner, lakukan differencing")
        
        order_diff = st.slider("Orde Differencing", 1, 3, 1)
        
        if st.button("Lakukan Differencing"):
            data_diff = st.session_state.data_preprocessed.diff(order_diff).dropna()
            st.session_state.data_stationary = data_diff
            
            # Plot hasil differencing
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(st.session_state.data_preprocessed, label='Data Asli')
            ax1.set_title('Data Asli')
            ax1.legend()
            
            ax2.plot(data_diff, label=f'Data setelah differencing orde {order_diff}')
            ax2.set_title('Data setelah Differencing')
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Uji stasioneritas setelah differencing
            result = adfuller(data_diff['Jumlah permintaan'])
            st.write("ADF Statistic setelah differencing:", result[0])
            st.write("p-value setelah differencing:", result[1])
            
            if result[1] > 0.05:
                st.warning("Data masih tidak stasioner, coba orde differencing yang lebih tinggi")
            else:
                st.success("Data sekarang stasioner!")
    else:
        st.success("Data sudah stasioner")
        st.write(st.session_state.data_stationary.head())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Kembali ke Step 3", on_click=prev_step)
    with col2:
        if st.session_state.get('data_stationary') is not None:
            if st.button("Lanjut ke Step 5", on_click=next_step):
                pass
    with col3:
        st.button("Reset", on_click=lambda: st.session_state.clear())

# Step 5: Plot ACF/PACF
elif st.session_state.current_step == 5 and st.session_state.data_stationary is not None:
    st.header("Step 5: Plot ACF dan PACF")
    data = st.session_state.data_stationary
    
    lags = st.slider("Jumlah Lag", 10, 50, 40)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    plot_acf(data, lags=lags, ax=ax1)
    plot_pacf(data, lags=lags, ax=ax2)
    st.pyplot(fig)
    
    st.write("Berdasarkan plot ACF/PACF, tentukan parameter ARIMA:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.number_input("Parameter p (AR)", 0, 10, 1)
    with col2:
        d = st.number_input("Parameter d (I)", 0, 3, 1)
    with col3:
        q = st.number_input("Parameter q (MA)", 0, 10, 0)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Kembali ke Step 4", on_click=prev_step)
    with col2:
        if st.button("Lanjut ke Step 6", on_click=next_step):
            pass
    with col3:
        st.button("Reset", on_click=lambda: st.session_state.clear())

# Step 6: Pemodelan ARIMA
elif st.session_state.current_step == 6 and st.session_state.data_stationary is not None:
    st.header("Step 6: Pemodelan ARIMA")
    data = st.session_state.data_stationary
    
    # Ambil parameter dari step sebelumnya
    p = st.number_input("Parameter p (AR)", 0, 10, 1, key='arima_p')
    d = st.number_input("Parameter d (I)", 0, 3, 1, key='arima_d')
    q = st.number_input("Parameter q (MA)", 0, 10, 0, key='arima_q')
    
    if st.button("Bangun Model ARIMA"):
        with st.spinner("Membangun model ARIMA..."):
            try:
                model = ARIMA(data, order=(p, d, q))
                model_fit = model.fit()
                st.session_state.model_arima = model_fit
                
                st.success("Model ARIMA berhasil dibangun!")
                st.write(model_fit.summary())
                
                # Prediksi dan evaluasi
                predictions = model_fit.predict()
                actual = data['Jumlah permintaan']
                
                mse, mae, mape, rmse = calculate_metrics(actual, predictions)
                
                st.session_state.arima_results = {
                    'mse': mse,
                    'mae': mae,
                    'mape': mape,
                    'rmse': rmse
                }
                
                st.subheader("Hasil Evaluasi Model ARIMA:")
                st.write(f"MSE: {mse:.4f}")
                st.write(f"MAE: {mae:.4f}")
                st.write(f"MAPE: {mape:.4f}%")
                st.write(f"RMSE: {rmse:.4f}")
                
                # Plot hasil prediksi
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(actual, label='Aktual')
                ax.plot(predictions, label='Prediksi')
                ax.set_title('Perbandingan Aktual vs Prediksi ARIMA')
                ax.legend()
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Kembali ke Step 5", on_click=prev_step)
    with col2:
        if st.session_state.get('model_arima') is not None:
            if st.button("Lanjut ke Step 7", on_click=next_step):
                pass
    with col3:
        st.button("Reset", on_click=lambda: st.session_state.clear())

# Step 7: Residual ARIMA
elif st.session_state.current_step == 7 and st.session_state.model_arima is not None:
    st.header("Step 7: Residual Model ARIMA")
    model = st.session_state.model_arima
    
    residuals = pd.DataFrame(model.resid, columns=['residual'])
    st.session_state.data_anfis = residuals
    
    st.write("Statistik Residual:")
    st.write(residuals.describe())
    
    # Plot residual
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(residuals, label='Residual')
    ax.set_title('Plot Residual ARIMA')
    ax.legend()
    st.pyplot(fig)
    
    # Normalisasi residual untuk ANFIS
    scaler = MinMaxScaler()
    residuals['residual'] = scaler.fit_transform(residuals[['residual']])
    st.session_state.data_anfis = residuals
    
    st.success("Residual telah diproses untuk pemodelan ANFIS")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Kembali ke Step 6", on_click=prev_step)
    with col2:
        if st.button("Lanjut ke Step 8", on_click=next_step):
            pass
    with col3:
        st.button("Reset", on_click=lambda: st.session_state.clear())

# Step 8: Pemodelan ANFIS-ABC
elif st.session_state.current_step == 8 and st.session_state.data_anfis is not None:
    st.header("Step 8: Pemodelan ANFIS dengan Optimasi ABC")
    data_anfis = st.session_state.data_anfis
    
    st.write("Menentukan lag signifikan untuk input ANFIS:")
    
    # Hitung PACF untuk menentukan lag signifikan
    jp = data_anfis['residual']
    pacf_values = pacf(jp, nlags=33)
    n = len(jp)
    ci = 1.96 / np.sqrt(n)
    significant_lags = [i for i, val in enumerate(pacf_values) if abs(val) > ci and i != 0]
    
    st.write("Lag signifikan:", significant_lags)
    
    if len(significant_lags) >= 2:
        lag1, lag2 = significant_lags[:2]
        
        # Buat fitur lag
        data_anfis[f'residual_lag{lag1}'] = data_anfis['residual'].shift(lag1)
        data_anfis[f'residual_lag{lag2}'] = data_anfis['residual'].shift(lag2)
        data_anfis.dropna(inplace=True)
        
        target = data_anfis.iloc[:, 0].values
        lag1_values = data_anfis.iloc[:, 1].values
        lag2_values = data_anfis.iloc[:, 2].values
        
        # Fungsi untuk ANFIS
        def initialize_membership_functions(data, num_clusters=4):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(data.reshape(-1, 1))
            centers = np.sort(kmeans.cluster_centers_.flatten())
            sigma = (centers[1] - centers[0]) / 2
            return centers, sigma
        
        def gaussian_membership(x, c, sigma):
            return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
        
        def firing_strength(lag1, lag2, c_lag1, sigma_lag1, c_lag2, sigma_lag2):
            lag1_low = gaussian_membership(lag1, c_lag1[0], sigma_lag1)
            lag1_high = gaussian_membership(lag1, c_lag1[1], sigma_lag1)
            lag2_low = gaussian_membership(lag2, c_lag2[0], sigma_lag2)
            lag2_high = gaussian_membership(lag2, c_lag2[1], sigma_lag2)
            
            rules = np.array([
                lag1_low * lag2_low,
                lag1_low * lag2_high,
                lag1_high * lag2_low,
                lag1_high * lag2_high,
            ]).T
            
            return rules
        
        if st.button("Mulai Optimasi ANFIS dengan ABC"):
            with st.spinner("Menjalankan optimasi ABC..."):
                # Inisialisasi fungsi keanggotaan
                c_lag1, sigma_lag1 = initialize_membership_functions(lag1_values)
                c_lag2, sigma_lag2 = initialize_membership_functions(lag2_values)
                
                # Hitung firing strength
                rules = firing_strength(lag1_values, lag2_values, c_lag1, sigma_lag1, c_lag2, sigma_lag2)
                normalized_rules = rules / rules.sum(axis=1, keepdims=True)
                
                # Fungsi ANFIS dengan JIT
                @jit(nopython=True)
                def anfis_predict(params, lag1, lag2, rules):
                    n_samples = lag1.shape[0]
                    n_rules = rules.shape[1]
                    p = params[:n_rules]
                    q = params[n_rules:2*n_rules]
                    r = params[2*n_rules:3*n_rules]
                    outputs = np.zeros(n_samples)
                    
                    for i in range(n_samples):
                        weighted_sum = 0.0
                        rule_sum = 0.0
                        
                        for j in range(n_rules):
                            rule_val = rules[i, j]
                            rule_output = p[j] * lag1[i] + q[j] * lag2[i] + r[j]
                            weighted_sum += rule_val * rule_output
                            rule_sum += rule_val
                        
                        outputs[i] = weighted_sum / (rule_sum + 1e-8)
                    
                    return outputs
                
                # Fungsi ABC sederhana
                def abc_optimizer(lag1, lag2, rules, target, num_food_sources=50, max_iter=200):
                    n_rules = rules.shape[1]
                    n_params = 3 * n_rules
                    
                    # Inisialisasi
                    food_sources = np.random.uniform(-1, 1, (num_food_sources, n_params))
                    fitness = np.array([mean_squared_error(target, anfis_predict(fs, lag1, lag2, rules)) 
                                      for fs in food_sources])
                    best_idx = np.argmin(fitness)
                    best_params = food_sources[best_idx].copy()
                    best_fitness = fitness[best_idx]
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for iteration in range(max_iter):
                        # Employed bees
                        for i in range(num_food_sources):
                            k = np.random.randint(0, num_food_sources)
                            while k == i:
                                k = np.random.randint(0, num_food_sources)
                            
                            phi = np.random.uniform(-1, 1, n_params)
                            candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                            candidate = np.clip(candidate, -1, 1)
                            
                            candidate_fitness = mean_squared_error(target, anfis_predict(candidate, lag1, lag2, rules))
                            
                            if candidate_fitness < fitness[i]:
                                food_sources[i] = candidate
                                fitness[i] = candidate_fitness
                                
                                if candidate_fitness < best_fitness:
                                    best_params = candidate.copy()
                                    best_fitness = candidate_fitness
                        
                        # Update progress
                        progress = (iteration + 1) / max_iter
                        progress_bar.progress(progress)
                        status_text.text(f"Iterasi {iteration + 1}/{max_iter}, Best Loss: {best_fitness:.6f}")
                    
                    return best_params, best_fitness
                
                # Jalankan optimasi
                best_params, best_loss = abc_optimizer(lag1_values, lag2_values, normalized_rules, target)
                st.session_state.best_params = best_params
                st.session_state.best_loss = best_loss
                
                st.success("Optimasi ABC selesai!")
                st.write(f"Best Loss: {best_loss:.6f}")
                
                # Prediksi dengan model ANFIS
                pred_anfis = anfis_predict(best_params, lag1_values, lag2_values, normalized_rules)
                
                # Plot hasil
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(target, label='Aktual')
                ax.plot(pred_anfis, label='Prediksi ANFIS')
                ax.set_title('Perbandingan Aktual vs Prediksi ANFIS')
                ax.legend()
                st.pyplot(fig)
                
                # Hitung metrik
                mse, mae, mape, rmse = calculate_metrics(target, pred_anfis)
                st.write(f"MSE ANFIS: {mse:.4f}")
                st.write(f"MAE ANFIS: {mae:.4f}")
                st.write(f"MAPE ANFIS: {mape:.4f}%")
                st.write(f"RMSE ANFIS: {rmse:.4f}")
    else:
        st.warning("Tidak cukup lag signifikan untuk membangun model ANFIS")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Kembali ke Step 7", on_click=prev_step)
    with col2:
        if st.session_state.get('best_params') is not None:
            if st.button("Lanjut ke Step 9", on_click=next_step):
                pass
    with col3:
        st.button("Reset", on_click=lambda: st.session_state.clear())

# Step 9: Hasil Prediksi Hybrid
elif st.session_state.current_step == 9 and st.session_state.model_arima is not None and st.session_state.best_params is not None:
    st.header("Step 9: Hasil Prediksi Hybrid ARIMA-ANFIS")
    
    # Prediksi ARIMA
    model_arima = st.session_state.model_arima
    pred_arima = model_arima.predict()
    
    # Prediksi ANFIS
    data_anfis = st.session_state.data_anfis
    jp = data_anfis['residual'].values
    lag1, lag2 = [int(col.split('_')[1][3:]) for col in data_anfis.columns if 'lag' in col][:2]
    
    lag1_values = data_anfis.iloc[:, 1].values
    lag2_values = data_anfis.iloc[:, 2].values
    
    # Inisialisasi fungsi keanggotaan
    c_lag1, sigma_lag1 = initialize_membership_functions(lag1_values)
    c_lag2, sigma_lag2 = initialize_membership_functions(lag2_values)
    
    # Hitung firing strength
    rules = firing_strength(lag1_values, lag2_values, c_lag1, sigma_lag1, c_lag2, sigma_lag2)
    normalized_rules = rules / rules.sum(axis=1, keepdims=True)
    
    # Prediksi ANFIS
    pred_anfis = anfis_predict(st.session_state.best_params, lag1_values, lag2_values, normalized_rules)
    
    # Gabungkan prediksi ARIMA dan ANFIS
    pred_hybrid = pred_arima[lag2:] + pred_anfis
    
    # Ambil data aktual
    actual = st.session_state.data_stationary['Jumlah permintaan'].values[lag2:]
    
    # Hitung metrik
    mse, mae, mape, rmse = calculate_metrics(actual, pred_hybrid)
    
    st.subheader("Hasil Evaluasi Model Hybrid:")
    st.write(f"MSE: {mse:.4f}")
    st.write(f"MAE: {mae:.4f}")
    st.write(f"MAPE: {mape:.4f}%")
    st.write(f"RMSE: {rmse:.4f}")
    
    # Plot hasil
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(actual, label='Aktual')
    ax.plot(pred_hybrid, label='Prediksi Hybrid')
    ax.set_title('Perbandingan Aktual vs Prediksi Hybrid ARIMA-ANFIS')
    ax.legend()
    st.pyplot(fig)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Kembali ke Step 8", on_click=prev_step)
    with col2:
        if st.button("Lanjut ke Step 10", on_click=next_step):
            pass
    with col3:
        st.button("Reset", on_click=lambda: st.session_state.clear())

# Step 10: Prediksi 6 Bulan Kedepan
elif st.session_state.current_step == 10 and st.session_state.model_arima is not None and st.session_state.best_params is not None:
    st.header("Step 10: Prediksi 6 Bulan Kedepan")
    
    n_steps = 6
    
    # Prediksi ARIMA
    model_arima = st.session_state.model_arima
    forecast_arima = model_arima.forecast(steps=n_steps)
    
    # Prediksi ANFIS
    data_anfis = st.session_state.data_anfis
    jp = data_anfis['residual'].values
    lag1, lag2 = [int(col.split('_')[1][3:]) for col in data_anfis.columns if 'lag' in col][:2]
    
    # Fungsi prediksi satu langkah ANFIS
    def predict_next_step(lag1_val, lag2_val):
        rules_new = firing_strength(np.array([lag1_val]), np.array([lag2_val]), c_lag1, sigma_lag1, c_lag2, sigma_lag2)
        normalized_rules_new = rules_new / rules_new.sum(axis=1, keepdims=True)
        pred = anfis_predict(st.session_state.best_params, np.array([lag1_val]), np.array([lag2_val]), normalized_rules_new)
        return pred[0]
    
    # Inisialisasi nilai lag terakhir
    last_lag1 = data_anfis.iloc[-1, 1]
    last_lag2 = data_anfis.iloc[-1, 2]
    
    forecast_anfis = []
    for _ in range(n_steps):
        pred = predict_next_step(last_lag1, last_lag2)
        forecast_anfis.append(pred)
        last_lag2 = last_lag1
        last_lag1 = pred
    
    # Gabungkan prediksi
    forecast_hybrid = forecast_arima + np.array(forecast_anfis)
    
    # Buat DataFrame hasil prediksi
    last_date = st.session_state.data_stationary.index[-1]
    future_dates = pd.date_range(start=last_date, periods=n_steps+1, freq='M')[1:]
    
    result_df = pd.DataFrame({
        'Tanggal': future_dates,
        'Prediksi ARIMA': forecast_arima,
        'Prediksi ANFIS': forecast_anfis,
        'Prediksi Hybrid': forecast_hybrid
    })
    
    st.write("Hasil Prediksi 6 Bulan Kedepan:")
    st.dataframe(result_df)
    
    # Plot hasil prediksi
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(st.session_state.data_stationary.index, st.session_state.data_stationary['Jumlah permintaan'], label='Data Historis')
    ax.plot(result_df['Tanggal'], result_df['Prediksi Hybrid'], label='Prediksi Hybrid', marker='o')
    ax.set_title('Prediksi 6 Bulan Kedepan')
    ax.legend()
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("Kembali ke Step 9", on_click=prev_step)
    with col2:
        st.button("Selesai", on_click=lambda: st.session_state.clear())
