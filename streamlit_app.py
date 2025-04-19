import streamlit as st

# Atur konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    layout="wide"
)

# Warna tema
soft_pink = "#FADADD"
maroon = "#800000"

# CSS untuk styling
st.markdown(
    f"""
    <style>
        .main {{
            background-color: {soft_pink};
        }}
        .header {{
            background-color: {maroon};
            color: white;
            padding: 1rem;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            border-radius: 8px;
        }}
        .nav-bar {{
            display: flex;
            justify-content: center;
            background-color: white;
            padding: 10px;
            border-bottom: 2px solid {maroon};
            border-top: 2px solid {maroon};
            margin-top: 10px;
            margin-bottom: 20px;
        }}
        .nav-bar a {{
            margin: 0 25px;
            font-weight: bold;
            color: {maroon};
            text-decoration: none;
            font-size: 18px;
        }}
        .nav-bar a:hover {{
            color: #a83232;
            text-decoration: underline;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Judul utama
st.markdown('<div class="header">PREDIKSI PERMINTAAN DARAH MENGGUNAKAN METODE<br>ARIMA-ANFIS DENGAN OPTIMASI ARTIFICIAL BEE COLONY</div>', unsafe_allow_html=True)

# Navigasi menu
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

# Konten halaman utama
st.markdown("## Selamat datang di sistem prediksi permintaan darah!")
st.markdown("Silakan pilih menu di atas untuk mulai menggunakan aplikasi.")
