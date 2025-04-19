# HALAMAN UTAMA

import streamlit as st
from PIL import Image

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    layout="wide"
)

# Inject CSS agar styling muncul
st.markdown("""
    <style>
        /* Ubah background seluruh halaman */
        body {
            background-color: #FADADD;  /* soft pink */
        }

        /* Header judul utama */
        .header-container {
            background-color: #800000;  /* maroon */
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        /* Navigasi menu */
        .nav-bar {
            display: flex;
            justify-content: center;
            background-color: white;
            padding: 10px 0;
            border-top: 3px solid #800000;
            border-bottom: 3px solid #800000;
            margin-bottom: 20px;
        }

        .nav-bar a {
            margin: 0 25px;
            font-weight: bold;
            color: #800000;
            text-decoration: none;
            font-size: 18px;
        }

        .nav-bar a:hover {
            color: #a83232;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Tampilkan header utama
st.markdown(
    """
    <div class="header-container">
        PREDIKSI PERMINTAAN DARAH MENGGUNAKAN METODE<br>
        ARIMA-ANFIS DENGAN OPTIMASI ARTIFICIAL BEE COLONY
    </div>
    """,
    unsafe_allow_html=True
)

# Tampilkan menu navigasi
st.markdown(
    """
    <div class="nav-bar">
        <a href="#">HOME</a>
        <a href="#">DATA PREPROCESSING</a>
        <a href="#">STASIONERITAS DATA</a>
        <a href="#">PREDIKSI</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Konten halaman HOME
st.markdown("## Selamat Datang 👋")
st.markdown("Ini adalah aplikasi prediksi permintaan darah menggunakan metode ARIMA-ANFIS dengan optimasi Artificial Bee Colony. Silakan pilih menu di atas untuk mulai.")

# MENU HOME
import streamlit as st

# Set konfigurasi halaman (WAJIB baris pertama sebelum apa pun)
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    layout="wide"
)

# Sidebar navigasi menu
menu = st.sidebar.radio("Menu", ["HOME", "DATA PREPROCESSING", "STASIONERITAS DATA", "PREDIKSI"])

# Tampilan menu HOME
if menu == "HOME":
    st.markdown("<h1 style='text-align: center; color: maroon;'>PREDIKSI PERMINTAAN DARAH</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>MENGGUNAKAN METODE ARIMA-ANFIS DENGAN OPTIMASI ARTIFICIAL BEE COLONY</h3>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div style='text-align: justify; font-size:18px'>
    Prediksi permintaan darah menggunakan metode <b>ARIMA-ANFIS</b> dengan optimasi <b>Artificial Bee Colony (ABC)</b> merupakan pendekatan hybrid yang menggabungkan kekuatan model statistik dan kecerdasan buatan untuk meningkatkan akurasi peramalan. <br><br>
    
    Metode <b>ARIMA (AutoRegressive Integrated Moving Average)</b> digunakan untuk menangkap pola linier dan tren jangka panjang dari data historis permintaan darah, sedangkan <b>ANFIS (Adaptive Neuro-Fuzzy Inference System)</b> digunakan untuk memodelkan hubungan non-linier yang kompleks. <br><br>
    
    Namun, agar parameter ANFIS dapat bekerja secara optimal, algoritma <b>Artificial Bee Colony</b> diterapkan sebagai metode optimasi berbasis perilaku lebah madu dalam mencari solusi terbaik. <br><br>
    
    Dengan menggabungkan ketiga metode ini, prediksi permintaan darah menjadi lebih akurat dan adaptif terhadap dinamika data, sehingga sangat membantu dalam pengambilan keputusan yang tepat di <b>Unit Transfusi Darah (UTD)</b>.
    </div>
    """, unsafe_allow_html=True)
