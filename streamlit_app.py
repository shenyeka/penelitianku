import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from pywt import wavedec, waverec
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from numba import jit
from statsmodels.graphics.tsaplots import plot_pacf

# Set halaman Streamlit
st.set_page_config(layout="centered")

# Inisialisasi state untuk menyimpan data dan status aplikasi
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

# Fungsi untuk menampilkan PACF plot
def plot_pacf_chart(jp):
    fig, ax = plt.subplots()
    plot_pacf(jp, lags=33, method='ywm', alpha=0.05, ax=ax)
    st.pyplot(fig)

# Fungsi untuk normalisasi residual
def normalize_residual(data_anfis):
    scaler_residual = MinMaxScaler()
    data_anfis['residual'] = scaler_residual.fit_transform(data_anfis[['residual']])
    return data_anfis, scaler_residual

# Fungsi untuk menentukan lag signifikan
def get_significant_lags(data_anfis):
    jp = data_anfis['residual']
    pacf_values = plot_pacf(jp, nlags=33, method='ywm')
    n = len(jp)  # jumlah data
    ci = 1.96 / np.sqrt(n)  # Batas interval kepercayaan 95% untuk PACF
    significant_lags = [i for i, val in enumerate(pacf_values) if abs(val) > ci and i != 0]
    return significant_lags, pacf_values

# Fungsi untuk ANFIS prediction
@jit(nopython=True)
def anfis_predict(params, lag32, lag33, rules):
    n_samples = lag32.shape[0]
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
            rule_output = p[j] * lag32[i] + q[j] * lag33[i] + r[j]
            weighted_sum += rule_val * rule_output
            rule_sum += rule_val
        outputs[i] = weighted_sum / (rule_sum + 1e-8)  # avoid division by zero
    return outputs

# Fungsi untuk optimisasi menggunakan Artificial Bee Colony (ABC)
def abc_optimizer(lag32, lag33, rules, target, num_food_sources=200, num_onlooker_bees=200, max_iter=2000, limit=150, param_bounds=(-1, 1)):
    n_rules = rules.shape[1]
    n_params = 3 * n_rules
    food_sources = np.random.uniform(param_bounds[0], param_bounds[1], (num_food_sources, n_params))
    fitness = np.zeros(num_food_sources)
    trial = np.zeros(num_food_sources, dtype=np.int32)
    for i in range(num_food_sources):
        fitness[i] = loss_function(food_sources[i], lag32, lag33, rules, target)
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_params = food_sources[best_idx].copy()
    candidate = np.zeros(n_params)
    for iteration in range(max_iter):
        # --- EMPLOYED BEES ---
        for i in range(num_food_sources):
            k = i
            while k == i:
                k = np.random.randint(0, num_food_sources)
            phi = np.random.uniform(-1, 1, n_params)
            candidate[:] = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, param_bounds[0], param_bounds[1])
            candidate_fit = loss_function(candidate, lag32, lag33, rules, target)
            if candidate_fit < fitness[i]:
                food_sources[i] = candidate.copy()
                fitness[i] = candidate_fit
                trial[i] = 0
                if candidate_fit < best_fitness:
                    best_fitness = candidate_fit
                    best_params = candidate.copy()
            else:
                trial[i] += 1
        # --- ONLOOKER BEES ---
        fitness_inv = 1.0 / (1.0 + fitness)
        probs = fitness_inv / np.sum(fitness_inv)
        for _ in range(num_onlooker_bees):
            i = np.random.choice(num_food_sources, p=probs)
            k = i
            while k == i:
                k = np.random.randint(0, num_food_sources)
            phi = np.random.uniform(-1, 1, n_params)
            candidate[:] = food_sources[i] + phi * (food_sources[i] - food_sources[k])
            candidate = np.clip(candidate, param_bounds[0], param_bounds[1])
            candidate_fit = loss_function(candidate, lag32, lag33, rules, target)
            if candidate_fit < fitness[i]:
                food_sources[i] = candidate.copy()
                fitness[i] = candidate_fit
                trial[i] = 0
                if candidate_fit < best_fitness:
                    best_fitness = candidate_fit
                    best_params = candidate.copy()
            else:
                trial[i] += 1
        # --- SCOUT BEES ---
        for i in range(num_food_sources):
            if trial[i] > limit:
                food_sources[i] = np.random.uniform(param_bounds[0], param_bounds[1], n_params)
                fitness[i] = loss_function(food_sources[i], lag32, lag33, rules, target)
                trial[i] = 0
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_params = food_sources[i].copy()
        if iteration % 100 == 0 or iteration == max_iter - 1:
            print(f"Iteration {iteration}, Best Loss: {best_fitness:.6f}")
    return best_params, best_fitness

# Fungsi untuk memprediksi langkah berikutnya ANFIS
def predict_next_step(lag32_val, lag33_val, c_lag32, sigma_lag32, c_lag33, sigma_lag33, best_params, rules):
    rules_new = firing_strength(lag32_val, lag33_val, c_lag32, sigma_lag32, c_lag33, sigma_lag33)
    lag32_arr = np.array([lag32_val])
    lag33_arr = np.array([lag33_val])
    rules_arr = rules_new.reshape(1, -1)
    pred = anfis_predict(best_params, lag32_arr, lag33_arr, rules_arr)
    return pred[0]

# Fungsi utama Streamlit untuk menjalankan proses ARIMA-ANFIS-ABC
def run_arima_anfis_abc_optimization(data_anfis, test):
    st.title("Pemodelan ARIMA-ANFIS-ABC")
    
    # --- Langkah 1: Pemodelan ARIMA ---
    st.subheader("Langkah 1: Pemodelan ARIMA")
    arima_model = ARIMA(data_anfis['Jumlah permintaan'], order=(1,1,1))  # Model ARIMA(p,d,q)
    arima_fitted = arima_model.fit()
    predictions_arima = arima_fitted.predict(start=len(data_anfis), end=len(data_anfis)+len(test)-1, typ='levels')
    
    # Menampilkan hasil prediksi ARIMA
    st.write("Prediksi ARIMA: ")
    st.write(predictions_arima)
    
    # --- Langkah 2: Pemodelan ANFIS + ABC ---
    st.subheader("Langkah 2: Pemodelan ANFIS + ABC")
    
    # Normalisasi data
    data_anfis, scaler_residual = normalize_residual(data_anfis)
    
    # Tentukan lag signifikan berdasarkan PACF
    significant_lags, pacf_values = get_significant_lags(data_anfis)
    
    st.subheader("Lag Signifikan Berdasarkan PACF")
    st.write(significant_lags)
    
    # Lakukan optimisasi dengan ABC
    target = data_anfis.iloc[:, 0].values  # residual
    lag32 = data_anfis.iloc[:, 1].values  # lag 32
    lag33 = data_anfis.iloc[:, 2].values  # lag 33

    best_params, best_loss = abc_optimizer(lag32, lag33, data_anfis[significant_lags].values, target)
    st.subheader("Hasil Optimisasi")
    st.write("Best Loss:", best_loss)
    st.write("Optimized Parameters:", best_params)
    
    # Prediksi ANFIS
    n_forecast = len(test['Jumlah permintaan'])
    forecast_anfis = []
    
    lag32_future = lag32[-1]
    lag33_future = lag33[-2]
    
    for _ in range(n_forecast):
        pred = predict_next_step(lag32_future, lag33_future, significant_lags, best_params)
        forecast_anfis.append(pred)
        lag33_future = lag32_future
        lag32_future = pred
    
    forecast_df_anfis = pd.DataFrame({
        'Tanggal': pd.date_range(start="2021-10-01", periods=n_forecast, freq='MS'),
        'Prediksi ANFIS (Denormalized)': forecast_anfis
    })
    
    st.subheader("Hasil Prediksi ANFIS (Denormalisasi)")
    st.dataframe(forecast_df_anfis)
    
    # --- Langkah 3: Evaluasi Model ---
    st.subheader("Evaluasi Model Hibrid ARIMA + ANFIS + ABC")
    
    pred_hybrid_test_abc = predictions_arima.values + forecast_df_anfis['Prediksi ANFIS (Denormalized)'].values
    mse_final = mean_squared_error(test['Jumlah permintaan'], pred_hybrid_test_abc)
    rmse_final = np.sqrt(mse_final)
    mape_final = np.mean(np.abs((test['Jumlah permintaan'] - pred_hybrid_test_abc) / test['Jumlah permintaan'])) * 100
    
    st.write(f"**MSE**: {mse_final:.4f}")
    st.write(f"**RMSE**: {rmse_final:.4f}")
    st.write(f"**MAPE**: {mape_final:.4f}%")
    
    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 6
