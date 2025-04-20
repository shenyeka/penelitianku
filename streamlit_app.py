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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Creative CSS Styling with Glassmorphism and Neumorphism
st.markdown("""
    <style>
        /* Base Styling */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #fff5f5 0%, #f8f0ff 100%);
            color: #3a3a3a;
        }
        
        /* Main Container */
        .main {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            box-shadow: 0 8px 32px 0 rgba(150, 0, 50, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 2rem;
            margin: 1rem;
        }
        
        /* Header with Glassmorphism Effect */
        .header-container {
            background: linear-gradient(135deg, rgba(200, 50, 80, 0.8) 0%, rgba(128, 0, 64, 0.9) 100%);
            color: white;
            padding: 2.5rem;
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 30px rgba(200, 50, 80, 0.3);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.2);
            position: relative;
            overflow: hidden;
            z-index: 1;
            transition: all 0.5s ease;
        }
        
        .header-container:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 40px rgba(200, 50, 80, 0.4);
        }
        
        .header-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, 
                          rgba(255,255,255,0.1) 0%, 
                          rgba(255,255,255,0.3) 50%, 
                          rgba(255,255,255,0.1) 100%);
            z-index: -1;
            opacity: 0.6;
        }
        
        /* Content Cards */
        .content {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 5px 5px 15px rgba(200, 100, 120, 0.1),
                       -5px -5px 15px rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.5);
            transition: all 0.3s ease;
        }
        
        .content:hover {
            transform: translateY(-5px);
            box-shadow: 8px 8px 20px rgba(200, 100, 120, 0.15),
                       -8px -8px 20px rgba(255, 255, 255, 0.9);
        }
        
        /* Buttons with Neumorphic Effect */
        .stButton>button {
            background: linear-gradient(135deg, #c83264 0%, #801040 100%);
            color: white;
            border: none;
            padding: 0.8rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 12px;
            box-shadow: 4px 4px 8px rgba(200, 50, 80, 0.2),
                       -4px -4px 8px rgba(255, 255, 255, 0.8);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 6px 6px 12px rgba(200, 50, 80, 0.3),
                       -6px -6px 12px rgba(255, 255, 255, 0.9);
            color: white;
        }
        
        .stButton>button:active {
            transform: translateY(1px);
            box-shadow: 2px 2px 4px rgba(200, 50, 80, 0.2),
                       -2px -2px 4px rgba(255, 255, 255, 0.8);
        }
        
        .stButton>button::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, 
                          rgba(255,255,255,0.1) 0%, 
                          rgba(255,255,255,0.3) 50%, 
                          rgba(255,255,255,0.1) 100%);
            z-index: -1;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .stButton>button:hover::before {
            opacity: 0.6;
        }
        
        /* Sidebar Styling */
        .sidebar .sidebar-content {
            background: rgba(255, 245, 245, 0.8);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 220, 220, 0.5);
            box-shadow: 5px 0 15px rgba(200, 100, 120, 0.1);
        }
        
        /* Radio Buttons */
        .stRadio>div {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 12px;
            padding: 0.5rem;
            box-shadow: inset 3px 3px 6px rgba(200, 100, 120, 0.1),
                       inset -3px -3px 6px rgba(255, 255, 255, 0.8);
        }
        
        .stRadio>div>div>label {
            color: #5a2a27;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            transition: all 0.2s ease;
        }
        
        .stRadio>div>div>label:hover {
            background: rgba(200, 50, 80, 0.1);
        }
        
        /* Input Fields */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select,
        .stMultiselect>div>div>div {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: inset 3px 3px 6px rgba(200, 100, 120, 0.1),
                       inset -3px -3px 6px rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(255, 220, 220, 0.8);
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus,
        .stSelectbox>div>div>select:focus,
        .stMultiselect>div>div>div:focus {
            box-shadow: inset 4px 4px 8px rgba(200, 100, 120, 0.15),
                       inset -4px -4px 8px rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(200, 50, 80, 0.5);
        }
        
        /* Alert Boxes */
        .stAlert {
            border-radius: 12px;
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
        }
        
        .stSuccess {
            background: rgba(210, 255, 220, 0.7);
            border: 1px solid rgba(100, 200, 120, 0.3);
        }
        
        .stWarning {
            background: rgba(255, 245, 200, 0.7);
            border: 1px solid rgba(200, 180, 80, 0.3);
        }
        
        .stError {
            background: rgba(255, 220, 220, 0.7);
            border: 1px solid rgba(200, 80, 80, 0.3);
        }
        
        /* Floating Animation */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
            100% { transform: translateY(0px); }
        }
        
        .floating {
            animation: float 4s ease-in-out infinite;
        }
        
        /* Pulse Animation */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.03); }
            100% { transform: scale(1); }
        }
        
        .pulse {
            animation: pulse 3s ease-in-out infinite;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 240, 240, 0.5);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#c83264, #801040);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(#d84274, #902050);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu with floating effect
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #c83264;' class='floating'>MENU NAVIGASI</h2>
    </div>
    """, unsafe_allow_html=True)
    menu = st.radio("", ["HOME", "DATA PREPROCESSING", "STASIONERITAS DATA", "DATA SPLITTING", "PREDIKSI"])

# ======================== HOME ========================
if menu == "HOME":
    st.markdown("<div class='header-container pulse'>PREDIKSI PERMINTAAN DARAH<br>MENGGUNAKAN ARIMA-ANFIS DENGAN OPTIMASI ABC</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content">
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <svg width="80" height="80" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin: 0 auto;">
                <path d="M7 12L10 15L17 8" stroke="#c83264" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="12" cy="12" r="10" stroke="#c83264" stroke-width="2"/>
            </svg>
        </div>
        <p style="text-align: center; font-size: 1.1rem; line-height: 1.8;">
        Antarmuka ini menggunakan metode hybrid <b style="color: #c83264;">ARIMA-ANFIS</b> dengan optimasi <b style="color: #c83264;">Artificial Bee Colony</b> (ABC)
        untuk memprediksi permintaan darah pada Unit Transfusi Darah (UTD).<br><br>
        Silakan mulai dengan mengunggah data pada menu <b style="color: #c83264;">DATA PREPROCESSING</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("KEMBALI", disabled=True):
            pass
    with col2:
        if st.button("LANJUT KE DATA PREPROCESSING"):
            menu = "DATA PREPROCESSING"
            st.experimental_rerun()

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
    
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("KEMBALI KE HOME"):
            menu = "HOME"
            st.experimental_rerun()
    with col2:
        if st.button("LANJUT KE STASIONERITAS DATA"):
            menu = "STASIONERITAS DATA"
            st.experimental_rerun()

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
                st.warning("⚠ Data tidak stasioner. Melakukan differencing...")

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
    
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("KEMBALI KE DATA PREPROCESSING"):
            menu = "DATA PREPROCESSING"
            st.experimental_rerun()
    with col2:
        if st.button("LANJUT KE DATA SPLITTING"):
            menu = "DATA SPLITTING"
            st.experimental_rerun()

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
            st.warning("⚠ Data harus hanya memiliki 1 kolom target untuk proses split time series.")

    else:
        st.info("Silakan unggah data yang ingin Anda split.")
    
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("KEMBALI KE STASIONERITAS DATA"):
            menu = "STASIONERITAS DATA"
            st.experimental_rerun()
    with col2:
        if st.button("LANJUT KE PREDIKSI"):
            menu = "PREDIKSI"
            st.experimental_rerun()

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
                st.warning("⚠ Mohon pilih target dan minimal satu input untuk menyimpan dataset ANFIS.")
    
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("KEMBALI KE DATA SPLITTING"):
            menu = "DATA SPLITTING"
            st.experimental_rerun()
    with col2:
        if st.button("SELESAI"):
            st.balloons()
            st.success("Proses prediksi selesai!")
