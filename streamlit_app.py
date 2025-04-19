import streamlit as st

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
