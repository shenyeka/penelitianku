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

# Elegant Pink-Maroon Theme with Enhanced Interactivity
st.markdown("""
    <style>
        /* Base Styling */
        html, body, [class*="css"] {
            font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #fff0f5 0%, #ffe6ee 100%);
            color: #4a2a2a;
        }
        
        /* Main Container */
        .main {
            background: rgba(255, 245, 248, 0.9);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 18px;
            box-shadow: 0 12px 40px 0 rgba(150, 0, 50, 0.12);
            border: 1px solid rgba(255, 230, 240, 0.25);
            padding: 2.5rem;
            margin: 1.2rem;
        }
        
        /* Elegant Header with Gradient */
        .header-container {
            background: linear-gradient(135deg, rgba(190, 60, 90, 0.9) 0%, rgba(120, 20, 60, 0.95) 100%);
            color: white;
            padding: 3rem;
            text-align: center;
            font-size: 2.7rem;
            font-weight: 700;
            border-radius: 18px;
            margin-bottom: 2.5rem;
            box-shadow: 0 8px 35px rgba(190, 60, 90, 0.25);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            border: 1px solid rgba(255, 230, 240, 0.35);
            text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.15);
            position: relative;
            overflow: hidden;
            z-index: 1;
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }
        
        .header-container:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 45px rgba(190, 60, 90, 0.35);
        }
        
        .header-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, 
                          rgba(255,255,255,0.15) 0%, 
                          rgba(255,255,255,0.25) 50%, 
                          rgba(255,255,255,0.15) 100%);
            z-index: -1;
            opacity: 0.7;
            transition: opacity 0.5s ease;
        }
        
        .header-container:hover::before {
            opacity: 0.9;
        }
        
        /* Content Cards with Soft Glow */
        .content {
            background: rgba(255, 248, 250, 0.85);
            border-radius: 16px;
            padding: 2.2rem;
            margin: 1.8rem 0;
            box-shadow: 6px 6px 18px rgba(200, 120, 140, 0.08),
                       -6px -6px 18px rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(255, 230, 240, 0.4);
            transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        }
        
        .content:hover {
            transform: translateY(-6px);
            box-shadow: 10px 10px 25px rgba(200, 120, 140, 0.12),
                       -10px -10px 25px rgba(255, 255, 255, 1);
        }
        
        /* Buttons with Elegant Hover Effect */
        .stButton>button {
            background: linear-gradient(135deg, #c04070 0%, #902050 100%);
            color: white;
            border: none;
            padding: 0.9rem 2.2rem;
            font-size: 1.05rem;
            font-weight: 600;
            border-radius: 14px;
            box-shadow: 5px 5px 12px rgba(190, 80, 100, 0.2),
                       -5px -5px 12px rgba(255, 255, 255, 0.9);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 8px 8px 16px rgba(190, 80, 100, 0.3),
                       -8px -8px 16px rgba(255, 255, 255, 1);
            color: white;
        }
        
        .stButton>button:active {
            transform: translateY(1px);
            box-shadow: 3px 3px 8px rgba(190, 80, 100, 0.2),
                       -3px -3px 8px rgba(255, 255, 255, 0.9);
        }
        
        .stButton>button::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(45deg, 
                          rgba(255,255,255,0.2) 0%, 
                          rgba(255,255,255,0.3) 50%, 
                          rgba(255,255,255,0.2) 100%);
            z-index: -1;
            opacity: 0;
            transition: opacity 0.4s ease;
        }
        
        .stButton>button:hover::after {
            opacity: 0.8;
        }
        
        /* Sidebar with Elegant Transparency */
        .sidebar .sidebar-content {
            background: rgba(255, 240, 245, 0.88);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-right: 1px solid rgba(255, 220, 230, 0.6);
            box-shadow: 8px 0 20px rgba(190, 100, 120, 0.1);
        }
        
        /* Radio Buttons with Soft Selection */
        .stRadio>div {
            background: rgba(255, 248, 250, 0.8);
            border-radius: 14px;
            padding: 0.6rem;
            box-shadow: inset 4px 4px 8px rgba(200, 120, 140, 0.08),
                       inset -4px -4px 8px rgba(255, 255, 255, 0.9);
        }
        
        .stRadio>div>div>label {
            color: #5a2a3a;
            font-weight: 600;
            padding: 0.6rem 1.2rem;
            border-radius: 10px;
            transition: all 0.25s ease;
        }
        
        .stRadio>div>div>label:hover {
            background: rgba(200, 80, 100, 0.12);
            box-shadow: 2px 2px 6px rgba(200, 120, 140, 0.1);
        }
        
        /* Input Fields with Soft Focus */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select,
        .stMultiselect>div>div>div {
            background: rgba(255, 250, 252, 0.9);
            border-radius: 12px;
            box-shadow: inset 4px 4px 10px rgba(200, 120, 140, 0.08),
                       inset -4px -4px 10px rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(255, 220, 230, 0.9);
            transition: all 0.35s ease;
        }
        
        .stTextInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus,
        .stSelectbox>div>div>select:focus,
        .stMultiselect>div>div>div:focus {
            box-shadow: inset 5px 5px 12px rgba(200, 120, 140, 0.12),
                       inset -5px -5px 12px rgba(255, 255, 255, 1);
            border: 1px solid rgba(200, 80, 100, 0.6);
        }
        
        /* Enhanced Alert Boxes */
        .stAlert {
            border-radius: 14px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        
        .stSuccess {
            background: rgba(220, 255, 230, 0.8);
            border: 1px solid rgba(120, 200, 140, 0.4);
        }
        
        .stWarning {
            background: rgba(255, 240, 210, 0.8);
            border: 1px solid rgba(200, 160, 90, 0.4);
        }
        
        .stError {
            background: rgba(255, 230, 230, 0.8);
            border: 1px solid rgba(200, 90, 90, 0.4);
        }
        
        /* Floating Animation */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-6px); }
            100% { transform: translateY(0px); }
        }
        
        .floating {
            animation: float 4.5s ease-in-out infinite;
        }
        
        /* Pulse Animation */
        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.9; }
            50% { transform: scale(1.03); opacity: 1; }
            100% { transform: scale(1); opacity: 0.9; }
        }
        
        .pulse {
            animation: pulse 3.5s ease-in-out infinite;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 240, 245, 0.6);
            border-radius: 12px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(#c04070, #902050);
            border-radius: 12px;
            border: 2px solid rgba(255, 240, 245, 0.6);
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(#d05080, #a03060);
        }
        
        /* Tooltip Styling */
        .stTooltip {
            background: rgba(120, 20, 60, 0.95) !important;
            border-radius: 10px !important;
            padding: 0.8rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 240, 245, 0.8) !important;
            border-radius: 12px !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #c04070 0%, #902050 100%) !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(190, 80, 100, 0.2) !important;
        }
        
        /* Progress Bar Styling */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #c04070 0%, #902050 100%) !important;
        }
        
        /* Metric Cards */
        .stMetric {
            background: rgba(255, 248, 250, 0.9) !important;
            border-radius: 14px !important;
            border-left: 4px solid #c04070 !important;
            box-shadow: 4px 4px 12px rgba(200, 120, 140, 0.1) !important;
        }
        
        /* Dataframe Styling */
        .stDataFrame {
            border-radius: 14px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu with enhanced floating effect
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2.5rem;'>
        <h2 style='color: #c04070; text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.1);' class='floating'>🩸 MENU NAVIGASI</h2>
    </div>
    """, unsafe_allow_html=True)
    menu = st.radio("", ["HOME", "DATA PREPROCESSING", "STASIONERITAS DATA", "DATA SPLITTING", "PREDIKSI"],
                   label_visibility="collapsed")

# ======================== HOME ========================
if menu == "HOME":
    st.markdown("""
    <div class='header-container pulse'>
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M7 12L10 15L17 8" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="12" cy="12" r="10" stroke="white" stroke-width="2"/>
            </svg>
            PREDIKSI PERMINTAAN DARAH<br>ARIMA-ANFIS OPTIMASI ABC
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content">
        <div style="text-align: center; margin-bottom: 2rem;">
            <svg width="100" height="100" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin: 0 auto;">
                <path d="M12 21C16.9706 21 21 16.9706 21 12C21 7.02944 16.9706 3 12 3C7.02944 3 3 7.02944 3 12C3 16.9706 7.02944 21 12 21Z" stroke="#c04070" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M12 8V16" stroke="#c04070" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M8 12H16" stroke="#c04070" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <p style="text-align: center; font-size: 1.2rem; line-height: 1.8; color: #5a2a3a;">
        Sistem ini menggunakan metode hybrid <span style="color: #c04070; font-weight: 600;">ARIMA-ANFIS</span> dengan optimasi 
        <span style="color: #c04070; font-weight: 600;">Artificial Bee Colony</span> (ABC) untuk memprediksi permintaan darah 
        pada Unit Transfusi Darah (UTD) dengan akurasi tinggi.
        </p>
        
        <div style="background: rgba(200, 80, 100, 0.1); border-left: 4px solid #c04070; padding: 1.2rem; border-radius: 10px; margin-top: 2rem;">
            <h4 style="color: #c04070; margin-top: 0;">📌 Panduan Penggunaan:</h4>
            <ol style="color: #5a2a3a;">
                <li>Unggah dataset di menu <b>DATA PREPROCESSING</b></li>
                <li>Periksa stasioneritas data di menu <b>STASIONERITAS DATA</b></li>
                <li>Bagi data menjadi training & testing di <b>DATA SPLITTING</b></li>
                <li>Lakukan prediksi di menu <b>PREDIKSI</b></li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)

# [Rest of your code remains exactly the same...]
# ==================== DATA PREPROCESSING ====================
elif menu == "DATA PREPROCESSING":
    st.markdown("<div class='header-container'>📊 DATA PREPROCESSING</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="content">
            <h4 style="color: #c04070; margin-top: 0;">Unggah Dataset Anda</h4>
            <p style="color: #5a2a3a;">Silakan unggah file CSV yang berisi data time series permintaan darah.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"], 
                                       help="Format harus berupa CSV dengan kolom tanggal dan nilai permintaan darah")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        with st.expander("🔍 Preview Data", expanded=True):
            st.dataframe(data.head().style.set_properties(**{'background-color': 'rgba(255, 240, 245, 0.8)',
                                                          'color': '#5a2a3a',
                                                          'border': '1px solid rgba(200, 120, 140, 0.2)'}))
            
            # Pilih kolom waktu
            time_col = st.selectbox("🗓️ Pilih Kolom Waktu sebagai Index", 
                                  options=data.columns,
                                  help="Pilih kolom yang berisi informasi tanggal/waktu")
            
            if time_col:
                data[time_col] = pd.to_datetime(data[time_col])
                data.set_index(time_col, inplace=True)
                
                st.success("✅ Index waktu berhasil ditetapkan")
                st.dataframe(data.head().style.set_properties(**{'background-color': 'rgba(255, 240, 245, 0.8)'}))

                # Tangani missing values
                missing = data.isnull().sum()
                if missing.any():
                    st.warning(f"⚠ Data memiliki {missing.sum()} missing values")
                    if st.button("Hapus Baris dengan Missing Values"):
                        data.dropna(inplace=True)
                        st.success(f"✅ {missing.sum()} baris dengan missing values telah dihapus")

                # Tampilkan plot interaktif
                st.markdown("""
                <div class="content">
                    <h4 style="color: #c04070; margin-top: 0;">📈 Visualisasi Data</h4>
                </div>
                """, unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=data, ax=ax, color='#c04070', linewidth=2.5)
                ax.set_title("Time Series Data Permintaan Darah", pad=20, fontsize=14, color='#5a2a3a')
                ax.set_facecolor('rgba(255, 240, 245, 0.5)')
                fig.patch.set_facecolor('rgba(255, 248, 250, 0)')
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)

                # Simpan data ke session_state
                st.session_state["data"] = data
                st.session_state["time_col"] = time_col

                st.success("🎉 Preprocessing selesai! Silakan lanjut ke menu 'STASIONERITAS DATA'.")

# ================== STASIONERITAS DATA =====================
elif menu == "STASIONERITAS DATA":
    st.markdown("<div class='header-container'>📈 STASIONERITAS DATA</div>", unsafe_allow_html=True)

    if "data" in st.session_state:
        data = st.session_state["data"]
        time_col = st.session_state.get("time_col", "")
        
        with st.container():
            st.markdown(f"""
            <div class="content">
                <h4 style="color: #c04070; margin-top: 0;">Analisis Stasioneritas</h4>
                <p style="color: #5a2a3a;">Data yang digunakan: <b>{time_col}</b> sebagai index waktu</p>
            </div>
            """, unsafe_allow_html=True)
            
            col = st.selectbox("📌 Pilih kolom untuk diuji stasioneritas:", 
                             data.columns,
                             help="Pilih kolom yang akan dianalisis stasioneritasnya")

        if col:
            # Uji ADF awal
            st.markdown("""
            <div class="content">
                <h4 style="color: #c04070; margin-top: 0;">🔍 Uji Augmented Dickey-Fuller (ADF)</h4>
                <p style="color: #5a2a3a;">Uji ini menentukan apakah data bersifat stasioner atau tidak</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner('Menghitung uji ADF...'):
                adf_result = adfuller(data[col])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
                    st.metric("P-Value", f"{adf_result[1]:.4f}")
                    
                with col2:
                    for key, val in adf_result[4].items():
                        st.metric(f"Critical Value ({key})", f"{val:.4f}")

                if adf_result[1] < 0.05:
                    st.success("✅ Data sudah stasioner (p-value < 0.05)")
                else:
                    st.warning("⚠ Data tidak stasioner. Melakukan differencing...")

                    # Differencing
                    data_diff = data[col].diff().dropna()
                    st.session_state["data_diff"] = data_diff

                    # Plot hasil differencing
                    st.markdown("""
                    <div class="content">
                        <h4 style="color: #c04070; margin-top: 0;">📉 Data Setelah Differencing</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(x=data_diff.index, y=data_diff.values, ax=ax, color='#902050', linewidth=2)
                    ax.set_title("Data Setelah Differencing", pad=20, fontsize=14, color='#5a2a3a')
                    ax.set_facecolor('rgba(255, 240, 245, 0.5)')
                    fig.patch.set_facecolor('rgba(255, 248, 250, 0)')
                    ax.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig)

                    # Uji ADF ulang setelah differencing
                    st.markdown("""
                    <div class="content">
                        <h4 style="color: #c04070; margin-top: 0;">🔍 Uji ADF - Setelah Differencing</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner('Menghitung uji ADF setelah differencing...'):
                        adf_diff_result = adfuller(data_diff)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("ADF Statistic", f"{adf_diff_result[0]:.4f}", delta="Differenced")
                            st.metric("P-Value", f"{adf_diff_result[1]:.4f}")
                            
                        with col2:
                            for key, val in adf_diff_result[4].items():
                                st.metric(f"Critical Value ({key})", f"{val:.4f}")

                        if adf_diff_result[1] < 0.05:
                            st.success("✅ Data sudah stasioner setelah differencing (p-value < 0.05)")
                        else:
                            st.error("❌ Data masih belum stasioner setelah differencing")

            # Plot ACF dan PACF
            st.markdown("""
            <div class="content">
                <h4 style="color: #c04070; margin-top: 0;">📊 Autocorrelation dan Partial Autocorrelation</h4>
                <p style="color: #5a2a3a;">Plot ini membantu menentukan parameter ARIMA (p,d,q)</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"])
            
            with tab1:
                fig_acf, ax_acf = plt.subplots(figsize=(12, 4))
                plot_acf(data[col].dropna(), lags=40, ax=ax_acf, color='#c04070')
                ax_acf.set_title('Autocorrelation Function (ACF)', pad=15, fontsize=14, color='#5a2a3a')
                ax_acf.set_facecolor('rgba(255, 240, 245, 0.5)')
                fig_acf.patch.set_facecolor('rgba(255, 248, 250, 0)')
                st.pyplot(fig_acf)

            with tab2:
                fig_pacf, ax_pacf = plt.subplots(figsize=(12, 4))
                plot_pacf(data[col].dropna(), lags=40, ax=ax_pacf, color='#902050')
                ax_pacf.set_title('Partial Autocorrelation Function (PACF)', pad=15, fontsize=14, color='#5a2a3a')
                ax_pacf.set_facecolor('rgba(255, 240, 245, 0.5)')
                fig_pacf.patch.set_facecolor('rgba(255, 248, 250, 0)')
                st.pyplot(fig_pacf)

    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu di menu 'DATA PREPROCESSING'")

# =================== DATA SPLITTING ===================
elif menu == "DATA SPLITTING":
    st.markdown("<div class='header-container'>✂️ DATA SPLITTING</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div class="content">
            <h4 style="color: #c04070; margin-top: 0;">Bagi Data Menjadi Training dan Testing</h4>
            <p style="color: #5a2a3a;">Silakan unggah data yang akan dibagi dengan rasio 80:20</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_split_file = st.file_uploader("Pilih file CSV", type=["csv"], 
                                             help="Format harus berupa CSV dengan data time series")

    if uploaded_split_file is not None:
        df = pd.read_csv(uploaded_split_file)
        
        with st.expander("🔍 Preview Data", expanded=True):
            st.dataframe(df.head().style.set_properties(**{'background-color': 'rgba(255, 240, 245, 0.8)',
                                                        'color': '#5a2a3a',
                                                        'border': '1px solid rgba(200, 120, 140, 0.2)'}))

            time_column = st.selectbox("🗓️ Pilih Kolom Waktu (jika ada)", 
                                     ["Tidak Ada"] + list(df.columns),
                                     help="Pilih kolom yang berisi informasi tanggal/waktu")

            if time_column != "Tidak Ada":
                df[time_column] = pd.to_datetime(df[time_column])
                df.set_index(time_column, inplace=True)
                st.success(f"✅ {time_column} berhasil dijadikan index")

            if len(df.columns) == 1:
                col_name = df.columns[0]
                
                # Slider untuk menentukan rasio split
                split_ratio = st.slider("📊 Rasio Pembagian Data", 
                                      min_value=0.6, max_value=0.9, 
                                      value=0.8, step=0.05,
                                      help="Rasio training:testing (default 80:20)")
                
                if st.button("🚀 Bagi Data", help="Klik untuk membagi data sesuai rasio yang ditentukan"):
                    train_size = int(len(df) * split_ratio)
                    train_data = df.iloc[:train_size]
                    test_data = df.iloc[train_size:]

                    st.session_state["train_data"] = train_data
                    st.session_state["test_data"] = test_data
                    st.session_state["split_ratio"] = split_ratio

                    st.success(f"✅ Data berhasil di-split dengan rasio {int(split_ratio*100)}% training dan {int((1-split_ratio)*100)}% testing")

                    # Visualisasi pembagian data
                    st.markdown("""
                    <div class="content">
                        <h4 style="color: #c04070; margin-top: 0;">📊 Visualisasi Pembagian Data</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(train_data.index, train_data[col_name], label='Training Data', color='#c04070')
                    ax.plot(test_data.index, test_data[col_name], label='Testing Data', color='#902050')
                    ax.axvline(x=test_data.index[0], color='#5a2a3a', linestyle='--', label='Pemisah')
                    ax.set_title(f"Pembagian Data ({int(split_ratio*100)}/{int((1-split_ratio)*100)})", pad=20, fontsize=14, color='#5a2a3a')
                    ax.set_facecolor('rgba(255, 240, 245, 0.5)')
                    fig.patch.set_facecolor('rgba(255, 248, 250, 0)')
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.legend()
                    st.pyplot(fig)

                    # Tampilkan statistik
                    st.markdown("""
                    <div class="content">
                        <h4 style="color: #c04070; margin-top: 0;">📋 Statistik Data</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Jumlah Data Training", len(train_data))
                    with col2:
                        st.metric("Jumlah Data Testing", len(test_data))

            else:
                st.warning("⚠ Data harus hanya memiliki 1 kolom target untuk proses split time series")

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
