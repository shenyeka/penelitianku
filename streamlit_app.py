import streamlit as st

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    layout="wide"
)

# Inject CSS agar styling muncul
st.markdown("""
    <style>
        /* Background halaman dengan gradasi */
        body {
            background: linear-gradient(135deg, #FADADD, #E6A6B9);
            font-family: 'Arial', sans-serif;
        }

        /* Header utama */
        .header-container {
            background: #800000;
            color: white;
            padding: 40px;
            text-align: center;
            font-size: 30px;
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

        /* Navigasi menu */
        .nav-bar {
            display: flex;
            justify-content: center;
            background-color: #fff;
            padding: 15px 0;
            border-top: 3px solid #800000;
            border-bottom: 3px solid #800000;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .nav-bar a {
            margin: 0 20px;
            font-weight: bold;
            color: #800000;
            text-decoration: none;
            font-size: 20px;
            padding: 10px;
            border-radius: 8px;
            transition: background-color 0.3s, color 0.3s;
        }

        .nav-bar a:hover {
            background-color: #800000;
            color: #fff;
            text-decoration: none;
        }

        /* Footer (Optional) */
        .footer {
            text-align: center;
            padding: 20px;
            background-color: #800000;
            color: white;
            border-radius: 10px;
            position: fixed;
            bottom: 0;
            width: 100%;
        }

        /* Konten */
        .content {
            text-align: justify;
            font-size: 18px;
            line-height: 1.6;
            margin: 20px 10%;
        }

        /* Efek hover pada konten paragraf */
        .content:hover {
            background-color: #F5F5F5;
            border-radius: 10px;
            padding: 10px;
            transition: background-color 0.3s ease;
        }

        /* Styling tombol */
        .btn {
            background-color: #800000;
            color: white;
            padding: 12px 25px;
            font-size: 18px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .btn:hover {
            background-color: #a83232;
        }

    </style>
""", unsafe_allow_html=True)

# Tampilan menu navigasi dengan sidebar
menu = st.sidebar.radio("Menu", ["HOME", "DATA PREPROCESSING", "STASIONERITAS DATA", "PREDIKSI"])

# Tampilan konten berdasarkan menu yang dipilih
if menu == "HOME":
    # Header utama
    st.markdown("<div class='header-container'>PREDIKSI PERMINTAAN DARAH MENGGUNAKAN METODE<br> ARIMA-ANFIS DENGAN OPTIMASI ARTIFICIAL BEE COLONY</div>", unsafe_allow_html=True)
    
    # Konten HOME
    st.markdown("## Selamat Datang 👋")
    st.markdown("Ini adalah aplikasi prediksi permintaan darah menggunakan metode ARIMA-ANFIS dengan optimasi Artificial Bee Colony. Silakan pilih menu di atas untuk mulai.")
    st.markdown("---")
    st.markdown("""
    <div class="content">
    Prediksi permintaan darah menggunakan metode <b>ARIMA-ANFIS</b> dengan optimasi <b>Artificial Bee Colony (ABC)</b> merupakan pendekatan hybrid yang menggabungkan kekuatan model statistik dan kecerdasan buatan untuk meningkatkan akurasi peramalan. <br><br>
    
    Metode <b>ARIMA (AutoRegressive Integrated Moving Average)</b> digunakan untuk menangkap pola linier dan tren jangka panjang dari data historis permintaan darah, sedangkan <b>ANFIS (Adaptive Neuro-Fuzzy Inference System)</b> digunakan untuk memodelkan hubungan non-linier yang kompleks. <br><br>
    
    Namun, agar parameter ANFIS dapat bekerja secara optimal, algoritma <b>Artificial Bee Colony</b> diterapkan sebagai metode optimasi berbasis perilaku lebah madu dalam mencari solusi terbaik. <br><br>
    
    Dengan menggabungkan ketiga metode ini, prediksi permintaan darah menjadi lebih akurat dan adaptif terhadap dinamika data, sehingga sangat membantu dalam pengambilan keputusan yang tepat di <b>Unit Transfusi Darah (UTD)</b>.
    </div>
    """, unsafe_allow_html=True)

elif menu == "DATA PREPROCESSING":
    st.markdown("## Data Preprocessing")
    # Implementasikan bagian ini sesuai kebutuhan

elif menu == "STASIONERITAS DATA":
    st.markdown("## Stasioneritas Data")
    # Implementasikan bagian ini sesuai kebutuhan

elif menu == "PREDIKSI":
    st.markdown("## Prediksi Permintaan Darah")
    # Implementasikan bagian ini sesuai kebutuhan
