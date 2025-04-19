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

                    # Fungsi untuk inisialisasi fungsi keanggotaan
def initialize_membership_functions(data_anfis, num_clusters=4):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(data_anfis.reshape(-1, 1))
    centers = np.sort(kmeans.cluster_centers_.flatten())
    sigma = (centers[1] - centers[0]) / 2
    return centers, sigma

# Fungsi keanggotaan Gaussian
def gaussian_membership(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

# Fungsi untuk menghitung kekuatan pemicu
def firing_strength(lag32, lag33, c_lag32, sigma_lag32, c_lag33, sigma_lag33):
    ''' Layar 1 '''
    lag32_low = gaussian_membership(lag32, c_lag32[0], sigma_lag32)
    lag32_high = gaussian_membership(lag32, c_lag32[1], sigma_lag32)
    lag33_low = gaussian_membership(lag33, c_lag33[0], sigma_lag33)
    lag33_high = gaussian_membership(lag33, c_lag33[1], sigma_lag33)

    ''' Layar 2 '''
    rules = np.array([
        lag32_low * lag33_low,   # A1
        lag32_low * lag33_high,  # A2
        lag32_high * lag33_low,  # B1
        lag32_high * lag33_high, # B2
    ]).T  # Transpose agar shape sesuai (n_samples, n_rules)

    return rules

# Fungsi ANFIS untuk prediksi
@jit(nopython=True)
def anfis_predict(params, lag32, lag33, rules):
    n_samples = lag32.shape[0]
    n_rules = rules.shape[1]

    # Split parameters
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

# Fungsi loss MSE untuk ANFIS
@jit(nopython=True)
def loss_function(params, lag32, lag33, rules, target):
    preds = anfis_predict(params, lag32, lag33, rules)
    error = 0.0
    for i in range(len(target)):
        diff = target[i] - preds[i]
        error += diff * diff
    return error / len(target)

# Fungsi untuk optimasi dengan Artificial Bee Colony
def abc_optimizer(lag32, lag33, rules, target,
                  num_food_sources=200,
                  num_onlooker_bees=200,
                  max_iter=2000,
                  limit=150,
                  param_bounds=(-1, 1)):

    n_rules = rules.shape[1]
    n_params = 3 * n_rules

    food_sources = np.random.uniform(param_bounds[0], param_bounds[1],
                                     (num_food_sources, n_params))
    fitness = np.zeros(num_food_sources)
    trial = np.zeros(num_food_sources, dtype=np.int32)

    for i in range(num_food_sources):
        fitness[i] = loss_function(food_sources[i], lag32, lag33, rules, target)

    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_params = food_sources[best_idx].copy()
    candidate = np.zeros(n_params)

    for iteration in range(max_iter):
        # === EMPLOYED BEES ===
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

        # === ONLOOKER BEES ===
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

        # === SCOUT BEES ===
        for i in range(num_food_sources):
            if trial[i] > limit:
                food_sources[i] = np.random.uniform(param_bounds[0], param_bounds[1], n_params)
                fitness[i] = loss_function(food_sources[i], lag32, lag33, rules, target)
                trial[i] = 0

                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_params = food_sources[i].copy()

        # === LOGGING ===
        if iteration % 100 == 0 or iteration == max_iter - 1:
            print(f"Iteration {iteration}, Best Loss: {best_fitness:.6f}")

    return best_params, best_fitness

# Fungsi untuk prediksi satu langkah ANFIS
def predict_next_step(lag32_val, lag33_val):
    rules_new = firing_strength(lag32_val, lag33_val, c_lag32, sigma_lag32, c_lag33, sigma_lag33)

    # Karena input berupa satu data, ubah jadi array 1 sample
    lag32_arr = np.array([lag32_val])
    lag33_arr = np.array([lag33_val])
    rules_arr = rules_new.reshape(1, -1)

    pred = anfis_predict(best_params, lag32_arr, lag33_arr, rules_arr)
    return pred[0]

# Fungsi utama aplikasi Streamlit
def main():
    st.title("PREDIKSI PERMINTAAN DARAH MENGGUNAKAN ARIMA + ANFIS")

    # Upload file
    uploaded_file = st.file_uploader("Upload Data", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Proses data
        # (Lakukan preprocessing dan bagi data menjadi train dan test di sini)

        if 'train' not in st.session_state:
            train, test = train_test_split(data, test_size=0.2, shuffle=False)
            st.session_state['train'] = train
            st.session_state['test'] = test
            st.success("Data telah dibagi menjadi training dan testing.")
        else:
            train = st.session_state['train']
            test = st.session_state['test']

        # Implementasi model ARIMA dan ANFIS
        if st.button("Latih Model ARIMA dan ANFIS"):
            # Langkah-langkah pelatihan ARIMA, ANFIS, dan optimasi ABC
            pass

if __name__ == "__main__":
    main()



