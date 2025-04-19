import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Konfigurasi halaman harus diletakkan di bagian paling atas
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    layout="wide"
)

# Inisialisasi variabel global untuk data
data_global = None

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
    st.markdown("<div class='header-container'>DATA PREPROCESSING</div>", unsafe_allow_html=True)
    
    st.markdown("""
        <ul>
            <li>Unggah dataset Anda menggunakan form di bawah ini.</li>
            <li>Tentukan kolom waktu yang akan menjadi indeks data.</li>
            <li>Cek dan tangani missing values.</li>
            <li>Tampilkan plot data setelah preprocessing.</li>
        </ul>
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
            fig, ax = plt.subplots()  # Membuat figure dan axis
            sns.lineplot(data=data, ax=ax)  # Menambahkan plot ke axis
            ax.set_title("Data Setelah Preprocessing")  # Menambahkan judul pada plot
            st.pyplot(fig)  # Menampilkan plot
            
            # Simpan data yang sudah diproses ke dalam data_global
            data_global = data  # Simpan data untuk digunakan di menu lain
            
            # Opsi untuk menyimpan dataset yang telah diproses
            st.markdown("<h3>Simpan Dataset Setelah Preprocessing</h3>", unsafe_allow_html=True)
            
            # Tombol untuk menyimpan dataset sebagai CSV
            csv_data = data.to_csv(index=True)
            st.download_button(
                label="Download Dataset (CSV)",
                data=csv_data,
                file_name="data_preprocessed.csv",
                mime="text/csv"
            )

elif menu == "STASIONERITAS DATA":
    st.markdown("<div class='header-container'>STASIONERITAS DATA</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content">
    ### Uji Stasioneritas Data Menggunakan Uji ADF
    <br>
    Pada langkah ini, kita akan melakukan Uji Augmented Dickey-Fuller (ADF) untuk menguji apakah data stasioner. Jika data tidak stasioner, kita akan melakukan differencing.
    </div>
    """, unsafe_allow_html=True)
    
    if data_global is not None and not data_global.empty:
        # Lakukan Uji Augmented Dickey-Fuller (ADF)
        adf_test = adfuller(data_global.iloc[:, 0])  # Menggunakan kolom pertama sebagai contoh
        
        st.write("Hasil Uji Augmented Dickey-Fuller (ADF):")
        st.write(f"ADF Statistic: {adf_test[0]}")
        st.write(f"P-Value: {adf_test[1]}")
        st.write("Kritikal Nilai:")
        for key, value in adf_test[4].items():
            st.write(f"{key}: {value}")
        
        # Interpretasi hasil
        alpha = 0.05
        if adf_test[1] < alpha:
            st.write("Data stasioner berdasarkan uji ADF.")
        else:
            st.write("Data tidak stasioner berdasarkan uji ADF.")
            
            # Lakukan differencing jika data tidak stasioner
            st.write("Melakukan Differencing untuk membuat data stasioner...")
            data_diff = data_global.diff().dropna()
            
            st.write("Data Setelah Differencing:")
            st.write(data_diff.head())
            
            # Menampilkan plot data setelah differencing
            st.write("Plot Data Setelah Differencing:")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=data_diff, ax=ax)
            ax.set_title("Data Setelah Differencing")
            st.pyplot(fig)
            
        # Plot ACF dan PACF
        st.write("Plot ACF dan PACF:")
        fig_acf, ax_acf = plt.subplots(figsize=(10, 6))
        plot_acf(data_global, lags=40, ax=ax_acf)
        st.pyplot(fig_acf)
        
        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 6))
        plot_pacf(data_global, lags=40, ax=ax_pacf)
        st.pyplot(fig_pacf)

    else:
        st.write("Data belum diproses, silakan kembali ke menu 'DATA PREPROCESSING'.")

elif menu == "PREDIKSI":
    st.markdown("<div class='header-container'>PREDIKSI</div>", unsafe_allow_html=True)
    st.write("Menu prediksi akan muncul di sini.")
