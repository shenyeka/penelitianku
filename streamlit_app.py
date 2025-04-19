import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Konfigurasi halaman harus diletakkan di bagian paling atas
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    layout="wide"
)

# Inject CSS agar styling muncul
st.markdown("""
    <style>
        /* Styling untuk konten data preprocessing */
        .preprocessing-content {
            background-color: #FFF0F5;  /* Light pink background */
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }

        .preprocessing-content h2 {
            text-align: center;
            color: #800000;
        }

        .preprocessing-content ul {
            padding-left: 20px;
            font-size: 18px;
        }

        /* Styling untuk tombol simpan */
        .save-btn {
            background-color: #800000;
            color: white;
            font-size: 18px;
            padding: 12px 30px;
            border-radius: 8px;
            cursor: pointer;
            text-align: center;
            transition: background-color 0.3s;
        }

        .save-btn:hover {
            background-color: #a83232;
        }
    </style>
""", unsafe_allow_html=True)

# Tampilan menu navigasi dengan sidebar
menu = st.sidebar.radio("Menu", ["HOME", "DATA PREPROCESSING", "STASIONERITAS DATA", "PREDIKSI"])

# Tampilan konten berdasarkan menu yang dipilih
if menu == "DATA PREPROCESSING":
    # Tampilan bagian data preprocessing dengan CSS yang lebih menarik
    st.markdown("<div class='preprocessing-content'>", unsafe_allow_html=True)
    
    st.markdown("<h2>DATA PREPROCESSING</h2>", unsafe_allow_html=True)
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
            fig, ax = plt.subplots()
            sns.lineplot(data=data, ax=ax)
            ax.set_title("Data Setelah Preprocessing")
            st.pyplot(fig)

            # Menambahkan tombol untuk menyimpan dataset setelah preprocessing
            st.markdown("<h3 style='text-align: center;'>Simpan Dataset Setelah Preprocessing</h3>", unsafe_allow_html=True)
            if st.button("Simpan Dataset"):
                # Menyimpan dataset ke dalam format CSV
                csv_data = data.to_csv(index=True)
                st.download_button(
                    label="Download Dataset (CSV)",
                    data=csv_data,
                    file_name="data_preprocessed.csv",
                    mime="text/csv"
                )
    
    st.markdown("</div>", unsafe_allow_html=True)

# Bagian lainnya tetap seperti sebelumnya    
elif menu == "STASIONERITAS DATA":
    st.markdown("## Stasioneritas Data")
    # Implementasikan bagian ini sesuai kebutuhan

elif menu == "PREDIKSI":
    st.markdown("## Prediksi Permintaan Darah")
    # Implementasikan bagian ini sesuai kebutuhan
