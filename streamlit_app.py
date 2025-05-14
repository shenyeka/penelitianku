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
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numba import jit
import io

# Set page config must be the first command
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    page_icon="➕",  # Atau emoji yang diinginkan
    layout="wide"
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
    menu = st.radio("", ["HOME", "INPUT DATA", "DATA PREPROCESSING", "STASIONERITAS DATA", "DATA SPLITTING", "PEMODELAN ARIMA", "PEMODELAN ARIMA-ANFIS", "PREDIKSI"],
                label_visibility="collapsed")


# ======================== HOME ========================
if menu == "HOME":
    st.markdown("""
    <div class='header-container pulse'>
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
            <span style="font-size: 24px; font-weight: bold; color: white;">
                PREDIKSI PERMINTAAN DARAH<br>ARIMA-ANFIS OPTIMASI ABC
            </span>
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
        Sistem ini menggunakan metode <span style="color: #c04070; font-weight: 600;">ARIMA-ANFIS</span> dengan optimasi 
        <span style="color: #c04070; font-weight: 600;">Artificial Bee Colony</span> (ABC) untuk memprediksi permintaan darah 
        pada Unit Transfusi Darah (UTD).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Panduan Penggunaan GUI
    st.markdown("""
    <div class="guide" style="background-color: #fff0f5; padding: 1.5rem; border-radius: 1rem; box-shadow: 0 0 10px rgba(192, 64, 112, 0.2); margin-top: 2rem;">
        <h3 style="color: #c04070; text-align: center; font-weight: bold;">📘 Panduan Penggunaan Sistem</h3>
        <ul style="font-size: 1.05rem; line-height: 1.8; color: #5a2a3a;">
            <li><strong>HOME</strong>: Halaman utama yang menjelaskan tujuan dan metode prediksi sistem.</li>
            <li><strong>DATA PREPROCESSING</strong>: Lakukan pembersihan dan normalisasi data untuk memastikan kualitas input model.</li>
            <li><strong>STASIONERITAS DATA</strong>: Uji stasioneritas data menggunakan metode statistik sebelum diterapkan pada ARIMA.</li>
            <li><strong>DATA SPLITTING</strong>: Pisahkan data menjadi data latih dan data uji agar proses prediksi lebih akurat.</li>
            <li><strong>PREDIKSI</strong>: Lakukan prediksi permintaan darah menggunakan model ARIMA-ANFIS dengan optimasi ABC.</li>
        </ul>
        <p style="text-align: center; color: #5a2a3a;"><em>Pastikan untuk mengikuti alur dari atas ke bawah agar proses prediksi berjalan optimal.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
# ==================== INPUT DATA ====================
elif menu == "INPUT DATA":
    # CSS untuk gaya glassmorphism
    st.markdown("""
        <style>
        .glass-box {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            padding: 20px;
            margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Menampilkan panduan kriteria dataset
    st.markdown("""
        <style>
            .note-box {
                background-color: #f8f9fa;
                border-left: 5px solid #e74c3c;
                padding: 15px;
                margin: 10px 0;
                border-radius: 0 8px 8px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .note-title {
                color: #e74c3c;
                font-weight: 600;
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            .note-list {
                padding-left: 20px;
            }
            .note-list li {
                margin-bottom: 8px;
            }
            .highlight {
                background-color: #fffde7;
                padding: 2px 4px;
                border-radius: 4px;
                font-weight: 500;
            }
        </style>

        <div class="note-box">
            <div class="note-title">📋 Panduan Kriteria Dataset</div>
            <ul class="note-list">
                <li>Dataset harus berupa <span class="highlight">data deret waktu (time series)</span> dengan kolom waktu sebagai indeks</li>
                <li>Dataset harus bersifat <span class="highlight">univariat</span> (hanya satu variabel target)</li>
                <li>Disarankan dataset memiliki lebih dari <span class="highlight">100 baris</span> untuk analisis yang lebih optimal</li>
                <li>Ukuran file maksimal adalah <span class="highlight">200 MB</span></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    # Menambahkan cek ukuran file
    max_file_size_mb = 200  # Maksimal 200 MB
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)  # Ukuran file dalam MB

        if file_size_mb > max_file_size_mb:
            st.error(f"Ukuran file terlalu besar! Ukuran file maksimal adalah {max_file_size_mb} MB.")
        else:
            data = pd.read_csv(uploaded_file)

            # Informasi jumlah baris
            num_rows = data.shape[0]
            st.markdown(f"""
            <div class="glass-box">
                <h4>📊 Informasi Data:</h4>
                <p><b>Jumlah Baris:</b> {num_rows}</p>
            </div>
            """, unsafe_allow_html=True)

            # Preview data
            st.markdown("""
            <div class="glass-box">
                <h4>🔍 Preview Data (5 baris pertama):</h4>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(data.head())

            # Simpan data ke session_state untuk digunakan di menu DATA PREPROCESSING
            st.session_state["data"] = data  # Pastikan "data" disimpan di session_state

# ==================== DATA PREPROCESSING ====================
elif menu == "DATA PREPROCESSING":
    st.markdown("<div class='header-container'>DATA PREPROCESSING</div>", unsafe_allow_html=True)

    st.markdown("""
        <style>
            .note-box {
                background-color: #f8f9fa;
                border-left: 5px solid #e74c3c;
                padding: 15px;
                margin: 10px 0;
                border-radius: 0 8px 8px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .note-title {
                color: #e74c3c;
                font-weight: 600;
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            .note-list {
                padding-left: 20px;
            }
            .note-list li {
                margin-bottom: 8px;
            }
            .highlight {
                background-color: #fffde7;
                padding: 2px 4px;
                border-radius: 4px;
                font-weight: 500;
            }
        </style>
    """, unsafe_allow_html=True)  # This is where the CSS block ends

    # Check if data exists in session_state
    if "data" in st.session_state:
        data = st.session_state["data"]
        st.write("Preview Data:")
        st.write(data.head())

        # Pilih kolom waktu sebagai index
        time_col = st.selectbox("Pilih Kolom Waktu sebagai Index", options=data.columns)

        if st.button("Periksa missing value"):
            if time_col:
                try:
                    # Mengubah kolom waktu menjadi datetime dan set sebagai index
                    data[time_col] = pd.to_datetime(data[time_col])
                    data.set_index(time_col, inplace=True)

                    # Tangani missing values
                    missing = data.isnull().sum()
                    if missing.any():
                        st.warning("Data memiliki missing values. Menghapus baris dengan nilai kosong.")
                        data.dropna(inplace=True)
                    else:
                        st.info("Data tidak memiliki missing values.")

                    # Tampilkan plot data
                    st.write("Plot Data Setelah Preprocessing:")
                    fig, ax = plt.subplots()
                    sns.lineplot(data=data, ax=ax)
                    ax.set_title("Data Time Series")
                    st.pyplot(fig)

                    # Simpan data yang sudah diproses ke session_state
                    st.session_state["data"] = data

                    st.success("Preprocessing selesai. Silakan lanjut ke menu 'STASIONERITAS DATA'.")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat preprocessing: {e}")
    else:
        st.warning("Data belum diunggah, silahkan kembali ke menu 'INPUT DATA'.")


# ================== STASIONERITAS DATA =====================

elif menu == "STASIONERITAS DATA":
    st.markdown("<div class='header-container'>STASIONERITAS DATA</div>", unsafe_allow_html=True)

    if "data" in st.session_state:
        data = st.session_state["data"]
        st.write("Menggunakan data hasil preprocessing.")

        col = st.selectbox("Pilih kolom untuk diuji stasioneritas:", data.columns)

        if st.button("Uji Stasioneritas"):
            if col:
                # Uji ADF awal
                st.subheader("Uji ADF Awal")
                adf_result = adfuller(data[col])
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"P-Value: {adf_result[1]:.4f}")
                for key, val in adf_result[4].items():
                    st.write(f"Critical Value ({key}): {val:.4f}")

                if adf_result[1] < 0.05:
                    st.success("✅ Data sudah stasioner.")
                    st.markdown(
                        "- **P-Value < 0.05** → menolak H0 → **data stasioner**\n"
                        "- ✅ **d = 0** (tidak perlu differencing)."
                    )

                    # ACF & PACF langsung jika data stasioner
                    st.subheader("Plot ACF dan PACF:")
                    fig_acf, ax_acf = plt.subplots()
                    plot_acf(data[col].dropna(), lags=40, ax=ax_acf)
                    st.pyplot(fig_acf)

                    fig_pacf, ax_pacf = plt.subplots()
                    plot_pacf(data[col].dropna(), lags=40, ax=ax_pacf)
                    st.pyplot(fig_pacf)

                    # Panduan Pembacaan ACF dan PACF (Improved Version)
                    st.markdown("""
                    ### 📊 Panduan Membaca ACF dan PACF untuk ARIMA

                    **🔍 Konsep Dasar:**
                    - **ACF (Autocorrelation Function)**: Mengukur korelasi antara observasi dengan lag-nya
                    - **PACF (Partial Autocorrelation Function)**: Mengukur korelasi antara observasi dengan lag-nya setelah menghilangkan pengaruh lag lainnya

                    **🎯 Cara Menentukan Parameter ARIMA:**
                    1. **Identifikasi Orde AR (p) dari PACF:**
                       - Cari lag terakhir di PACF yang melebihi batas signifikan (garis biru)
                       - Contoh: Jika signifikan di lag 1 dan 2 → p=2

                    2. **Identifikasi Orde MA (q) dari ACF:**
                       - Cari lag terakhir di ACF yang melebihi batas signifikan
                       - Contoh: Jika signifikan di lag 1 → q=1

                    **💡 Tips Interpretasi:**
                    1. Fokus pada lag awal (10-15 lag pertama)
                    2. Nilai di dalam area biru (confidence interval) tidak signifikan
                    """, unsafe_allow_html=True)

                else:
                    st.warning("⚠ Data tidak stasioner, lakukan differencing...")

                    # Differencing
                    data_diff = data[col].diff().dropna()
                    st.session_state["data_diff"] = data_diff

                    st.subheader("Plot Setelah Differencing:")
                    fig, ax = plt.subplots()
                    sns.lineplot(x=data_diff.index, y=data_diff.values, ax=ax)
                    ax.set_title("Data Setelah Differencing")
                    st.pyplot(fig)

                    # Uji ADF ulang
                    st.subheader("Uji ADF - Setelah Differencing")
                    adf_diff_result = adfuller(data_diff)
                    st.write(f"ADF Statistic: {adf_diff_result[0]:.4f}")
                    st.write(f"P-Value: {adf_diff_result[1]:.4f}")
                    for key, val in adf_diff_result[4].items():
                        st.write(f"Critical Value ({key}): {val:.4f}")

                    if adf_diff_result[1] < 0.05:
                        st.success("✅ Data sudah stasioner setelah differencing.")
                        st.markdown(
                            "- **P-Value < 0.05** → menolak H0 → **data stasioner** setelah differencing\n"
                            "- ✅ **d = 1** (butuh 1x differencing)."
                        )
                    else:
                        st.error("❌ Data masih belum stasioner setelah differencing.")
                        st.markdown(
                            "- Pertimbangkan melakukan differencing ke-2 (**d = 2**)."
                        )

                    # ACF dan PACF untuk data differencing
                    st.subheader("Plot ACF dan PACF:")
                    fig_acf, ax_acf = plt.subplots()
                    plot_acf(data_diff, lags=40, ax=ax_acf)
                    st.pyplot(fig_acf)

                    fig_pacf, ax_pacf = plt.subplots()
                    plot_pacf(data_diff, lags=40, ax=ax_pacf)
                    st.pyplot(fig_pacf)

                    # Panduan Pembacaan ACF dan PACF (Improved Version)
                    st.markdown("""
                    ### 📊 Panduan Membaca ACF dan PACF untuk ARIMA

                    **🔍 Konsep Dasar:**
                    - **ACF (Autocorrelation Function)**: Mengukur korelasi antara observasi dengan lag-nya
                    - **PACF (Partial Autocorrelation Function)**: Mengukur korelasi antara observasi dengan lag-nya setelah menghilangkan pengaruh lag lainnya

                    **🎯 Cara Menentukan Parameter ARIMA:**
                    1. **Identifikasi Orde AR (p) dari PACF:**
                       - Cari lag terakhir di PACF yang melebihi batas signifikan (garis biru)
                       - Contoh: Jika signifikan di lag 1 dan 2 → p=2

                    2. **Identifikasi Orde MA (q) dari ACF:**
                       - Cari lag terakhir di ACF yang melebihi batas signifikan
                       - Contoh: Jika signifikan di lag 1 → q=1
                       
                    **💡 Tips Interpretasi:**
                    1. Fokus pada lag awal (10-15 lag pertama)
                    2. Nilai di dalam area biru (confidence interval) tidak signifikan
                    """, unsafe_allow_html=True)

    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu di menu 'DATA PREPROCESSING'.")
        
# =================== DATA SPLITTING ===================
elif menu == "DATA SPLITTING":
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    st.markdown("<div class='header-container'>DATA SPLITTING</div>", unsafe_allow_html=True)

    # Check if preprocessing data exists in session state
    if "data" in st.session_state:
        df = st.session_state["data"]
        st.write("Menggunakan data hasil preprocessing.")
        
        st.write("Preview Data:")
        st.write(df.head())

        # Pastikan hanya satu kolom target
        if len(df.columns) == 1:
            col_name = df.columns[0]

            # Tambahkan slider untuk memilih rasio train dan test
            split_ratio = st.slider("Pilih rasio pembagian data (Training:Testing)", 0.1, 0.9, 0.8, 0.05)
            train_size = int(len(df) * split_ratio)
            train_data = df.iloc[:train_size].copy()
            test_data = df.iloc[train_size:].copy()

            # Simpan untuk proses berikutnya
            st.session_state["train_data"] = train_data
            st.session_state["test_data"] = test_data

            st.success(f"✅ Data berhasil di-split dengan rasio {split_ratio*100}% training dan {(1-split_ratio)*100}% testing.")

            st.subheader("Data Training:")
            st.write(train_data)
            st.line_chart(train_data)

            st.subheader("Data Testing:")
            st.write(test_data)
            st.line_chart(test_data)
        else:
            st.warning("⚠ Data harus hanya memiliki 1 kolom target untuk proses split time series.")
    else:
        st.info("Silakan lakukan preprocessing data terlebih dahulu.")


# ========PEMODELAN ARIMA=====
elif menu == "PEMODELAN ARIMA":
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    st.title("PREDIKSI PERMINTAAN DARAH MENGGUNAKAN ARIMA")

    train = st.session_state.get('train_data')
    test = st.session_state.get('test_data')

    if train is not None and test is not None:
        st.subheader("Tentukan Parameter ARIMA (p, d, q)")
        
        # Input parameter ARIMA
        p = st.number_input("Masukkan nilai p (autoregressive term)", min_value=0, value=1, step=1)
        d = st.number_input("Masukkan nilai d (differencing)", min_value=0, value=0, step=1)
        q = st.number_input("Masukkan nilai q (moving average term)", min_value=0, value=1, step=1)

        if st.button("Latih Model ARIMA"):
            model_arima = ARIMA(train, order=(p, d, q))
            model_arima = model_arima.fit()
            st.success("Model ARIMA berhasil dilatih.")
            st.write(model_arima.summary())

            start_test = len(train)
            pred_train = model_arima.predict(start=0, end=start_test-1)
            pred_test = model_arima.forecast(steps=len(test))

            # Tambahkan prediksi ke data
            test['prediksi'] = pred_test

            st.subheader("4. Evaluasi Model dengan MAPE dan RMSE")

            # Hitung MAPE dan RMSE untuk training dan testing
            mape_train = mean_absolute_percentage_error(train, pred_train) * 100
            mape_test = mean_absolute_percentage_error(test.iloc[:, 0], test['prediksi']) * 100

            st.write(f"MAPE Training: {mape_train:.2f}%")
            st.write(f"MAPE Testing: {mape_test:.2f}%")

            st.line_chart({"Data Aktual": test.iloc[:, 0], "Prediksi ARIMA": test['prediksi']})

            # Simpan model & residual ke session_state
            st.session_state['model_arima'] = model_arima
            st.session_state['residual_arima'] = model_arima.resid


# ==========Menu ARIMA-ANFIS===============
elif menu == "PEMODELAN ARIMA-ANFIS":
    st.markdown("<div class='header-container'>PEMODELAN ARIMA-ANFIS</div>", unsafe_allow_html=True)

    if 'model_arima' in st.session_state:
        st.subheader("Input ANFIS")

        # Tombol untuk menampilkan residual ARIMA
        if st.button("Lihat Residual ARIMA"):
            residual = st.session_state['residual_arima']
            st.line_chart(residual)

            # Simpan residual ke DataFrame
            data_anfis = pd.DataFrame({'residual': residual})
            st.session_state['data_anfis_raw'] = data_anfis

        # Tombol normalisasi residual
        if st.button("Lanjutkan ke Normalisasi Residual"):
            if 'data_anfis_raw' in st.session_state:
                data_anfis = st.session_state['data_anfis_raw']
                scaler_residual = MinMaxScaler()
                data_anfis['residual'] = scaler_residual.fit_transform(data_anfis[['residual']])
                st.session_state['data_anfis'] = data_anfis
                st.session_state['scaler_residual'] = scaler_residual
                st.success("Residual berhasil dinormalisasi.")
                st.write(data_anfis.head())
                st.info("Silakan tentukan input ANFIS dari PACF.")
            else:
                st.warning("Residual belum tersedia. Klik 'Lihat Residual ARIMA' terlebih dahulu.")

        # Tombol PACF untuk menentukan input ANFIS
        if st.button("Tentukan Input ANFIS dari PACF"):
            if 'data_anfis' in st.session_state:
                data_anfis = st.session_state['data_anfis']

                jp = data_anfis['residual']
                if len(jp) > 1:
                    from statsmodels.tsa.stattools import pacf
                    from statsmodels.graphics.tsaplots import plot_pacf
                    import numpy as np
                    import matplotlib.pyplot as plt

                    pacf_values = pacf(jp, nlags=12)
                    n = len(jp)
                    ci = 1.96 / np.sqrt(n)

                    # Tentukan lag signifikan
                    significant_lags = [i for i, val in enumerate(pacf_values) if abs(val) > ci and i != 0]
                    st.write(f"Lag signifikan (berdasarkan interval kepercayaan): {significant_lags}")

                    # Tambahkan lag signifikan ke DataFrame
                    for lag in significant_lags:
                        data_anfis[f'residual_lag{lag}'] = data_anfis['residual'].shift(lag)
                    data_anfis.dropna(inplace=True)

                    # Simpan hasilnya
                    st.session_state['data_anfis_with_lags'] = data_anfis
                    st.success("Lag signifikan berhasil ditambahkan.")
                    st.write(data_anfis.head())

                    # Plot PACF
                    st.subheader("Plot Partial Autocorrelation Function (PACF)")
                    plt.figure(figsize=(10, 6))
                    plot_pacf(jp, lags=12, method='ywm', alpha=0.05)
                    plt.title('Partial Autocorrelation Function (PACF) residual ARIMA')
                    st.pyplot(plt)

                    # Pilih 2 input teratas dari lag signifikan
                    if len(significant_lags) >= 2:
                        input_lags = [f'residual_lag{lag}' for lag in significant_lags[:2]]
                        st.session_state['anfis_input_lags'] = input_lags
                        st.session_state['anfis_target'] = data_anfis['residual'].values
                        st.session_state['anfis_input'] = data_anfis[input_lags].values
                        st.info(f"Dua input ANFIS terpilih: {input_lags}")
                    else:
                        st.warning("Jumlah lag signifikan kurang dari 2. Tidak cukup untuk input ANFIS.")
                else:
                    st.warning("Data terlalu sedikit untuk menghitung PACF. Pastikan dataset memiliki lebih dari satu data.")
            else:
                st.warning("Data residual belum dinormalisasi. Klik 'Lanjutkan ke Normalisasi Residual' terlebih dahulu.")

