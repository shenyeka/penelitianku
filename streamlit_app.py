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

# [Rest of your code remains exactly the same...]
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

# [Continue with all your existing code exactly as before...]
# Just replace the remaining header containers with the new styled ones where needed
