import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman harus diletakkan di bagian paling atas
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    layout="wide"
)

# Inject CSS agar styling muncul
st.markdown("""
    <style>
        /* Background halaman dengan warna soft pink */
        body {
            background-color: #FADADD;  /* soft pink */
            font-family: 'Arial', sans-serif;
            font-size: 18px;
        }

        /* Header utama */
        .header-container {
            background: #800000;
            color: white;
            padding: 40px;
            text-align: center;
            font-size: 40px;  /* Memperbesar ukuran font header */
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
            padding: 20px 0;
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
            font-size: 22px;  /* Memperbesar ukuran font pada link */
            padding: 12px;
            border-radius: 8px;
            transition: background-color 0.3s, color 0.3s;
        }

        .nav-bar a:hover {
            background-color: #800000;
            color: #fff;
            text-decoration: none;
        }

        /* Konten */
        .content {
            text-align: justify;
            font-size: 20px;  /* Memperbesar ukuran font konten */
            line-height: 1.8;
            margin: 20px 10%;
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .content:hover {
            background-color: #F5F5F5;
            border-radius: 10px;
            padding: 15px;
            transition: background-color 0.3s ease;
        }

        /* Styling tombol */
        .btn {
            background-color: #800000;
            color: white;
            padding: 15px 30px;
            font-size: 20px;  /* Memperbesar ukuran font tombol */
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
    st.markdown("## Selamat Datang")
    st.markdown("Ini adalah antarmuka prediksi permintaan darah menggunakan metode ARIMA-ANFIS dengan optimasi Artificial Bee Colony. Silakan pilih menu di atas untuk mulai.")
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
    st.markdown("<h1 style='text-align: center;'>DATA PREPROCESSING</h1>", unsafe_allow_html=True)
    st.markdown("### Langkah-langkah Preprocessing Data")
    st.markdown("""
    1. Unggah dataset Anda menggunakan form di bawah ini. <br>
    2. Tentukan kolom waktu yang akan menjadi indeks data. <br>
    3. Cek dan tangani missing values. <br>
    4. Tampilkan plot data setelah preprocessing.
    """, unsafe_allow_html=True)
    
    # Upload file dataset
    uploaded_file = st.file_uploader("Unggah Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Membaca file dataset
        data = pd.read_csv(uploaded_file)

        # Menampilkan preview data
        st.write("Preview Data:")
        st.write(data.head())
        
        # Pilih kolom waktu yang akan dijadikan indeks
        time_column = st.selectbox("Pilih Kolom Waktu", options=data.columns)
        
        if time_column:
            # Mengatur kolom waktu sebagai indeks
            data.set_index(time_column, inplace=True)
            st.write(f"Data setelah menetapkan {time_column} sebagai indeks:")
            st.write(data.head())
        
            # Mengecek missing values
            st.write("Mengecek Missing Values:")
            missing_values = data.isnull().sum()
            st.write(missing_values)

            # Menangani missing values dengan menghapus baris yang memiliki nilai null
            if missing_values.any():
                st.write("Menghapus baris dengan missing values...")
                data = data.dropna()
                st.write("Data setelah menghapus missing values:")
                st.write(data.head())

            # Menampilkan plot data setelah preprocessing
            st.write("Plot Data Setelah Preprocessing:")
            sns.lineplot(data=data)
            plt.title("Data Setelah Preprocessing")
            st.pyplot()
    
elif menu == "STASIONERITAS DATA":
    st.markdown("## Stasioneritas Data")
    # Implementasikan bagian ini sesuai kebutuhan

elif menu == "PREDIKSI":
    st.markdown("## Prediksi Permintaan Darah")
    # Implementasikan bagian ini sesuai kebutuhan
