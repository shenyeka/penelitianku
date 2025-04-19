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
import io

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Permintaan Darah",
    layout="wide"
)

# Styling CSS
st.markdown("""
    <style>
        body {
            background-color: #FADADD;
            font-family: 'Arial', sans-serif;
            font-size: 18px;
        }

        .header-container {
            background: #800000;
            color: white;
            padding: 40px;
            text-align: center;
            font-size: 40px;
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

        .content {
            text-align: justify;
            font-size: 20px;
            line-height: 1.8;
            margin: 20px 10%;
            padding: 20px;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.radio("Menu", ["HOME", "DATA PREPROCESSING", "STASIONERITAS DATA", "DATA SPLITTING", "PREDIKSI"])

# ======================== HOME ========================
if menu == "HOME":
    st.markdown("<div class='header-container'>PREDIKSI PERMINTAAN DARAH<br>MENGGUNAKAN MENGGUNAKAN ARIMA-ANFIS DENGAN OPTIMASI ABC</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content">
    Antarmuka ini menggunakan metode hybrid <b>ARIMA-ANFIS</b> dengan optimasi <b>Artificial Bee Colony</b> (ABC)
    untuk memprediksi permintaan darah pada Unit Transfusi Darah (UTD).<br><br>
    Silakan mulai dengan mengunggah data pada menu <b>DATA PREPROCESSING</b>.
    </div>
    """, unsafe_allow_html=True)

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

# ================== STASIONERITAS DATA =====================
elif menu == "STASIONERITAS DATA":
    st.markdown("<div class='header-container'>STASIONERITAS DATA</div>", unsafe_allow_html=True)

    if "data" in st.session_state:
        data = st.session_state["data"]
        st.write("Menggunakan data hasil preprocessing.")

        col = st.selectbox("Pilih kolom untuk diuji stasioneritas:", data.columns)

        if col:
            # Uji ADF awal
            st.subheader("Uji ADF - Sebelum Differencing")
            adf_result = adfuller(data[col])
            st.write(f"ADF Statistic: {adf_result[0]:.4f}")
            st.write(f"P-Value: {adf_result[1]:.4f}")
            for key, val in adf_result[4].items():
                st.write(f"Critical Value ({key}): {val:.4f}")

            if adf_result[1] < 0.05:
                st.success("✅ Data sudah stasioner.")
            else:
                st.warning("⚠️ Data tidak stasioner. Melakukan differencing...")

                # Differencing
                data_diff = data[col].diff().dropna()

                # Simpan data differencing ke session
                st.session_state["data_diff"] = data_diff

                # Plot hasil differencing
                st.subheader("Plot Setelah Differencing:")
                fig, ax = plt.subplots()
                sns.lineplot(x=data_diff.index, y=data_diff.values, ax=ax)
                ax.set_title("Data Setelah Differencing")
                st.pyplot(fig)

                # Uji ADF ulang setelah differencing
                st.subheader("Uji ADF - Setelah Differencing")
                adf_diff_result = adfuller(data_diff)
                st.write(f"ADF Statistic: {adf_diff_result[0]:.4f}")
                st.write(f"P-Value: {adf_diff_result[1]:.4f}")
                for key, val in adf_diff_result[4].items():
                    st.write(f"Critical Value ({key}): {val:.4f}")

                if adf_diff_result[1] < 0.05:
                    st.success("✅ Data sudah stasioner setelah differencing.")
                else:
                    st.error("❌ Data masih belum stasioner setelah differencing.")

            # Plot ACF dan PACF
            st.subheader("Plot ACF dan PACF:")
            fig_acf, ax_acf = plt.subplots()
            plot_acf(data[col].dropna(), lags=40, ax=ax_acf)
            st.pyplot(fig_acf)

            fig_pacf, ax_pacf = plt.subplots()
            plot_pacf(data[col].dropna(), lags=40, ax=ax_pacf)
            st.pyplot(fig_pacf)

    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu di menu 'DATA PREPROCESSING'.")

# =================== DATA SPLITTING ===================
elif menu == "DATA SPLITTING":
    st.markdown("<div class='header-container'>DATA SPLITTING</div>", unsafe_allow_html=True)

    uploaded_split_file = st.file_uploader("Unggah Data yang Akan Di-Split (CSV)", type=["csv"])

    if uploaded_split_file is not None:
        df = pd.read_csv(uploaded_split_file)

        st.write("Preview Data:")
        st.write(df.head())

        time_column = st.selectbox("Pilih Kolom Waktu (jika ada)", ["Tidak Ada"] + list(df.columns))

        if time_column != "Tidak Ada":
            df[time_column] = pd.to_datetime(df[time_column])
            df.set_index(time_column, inplace=True)

        if len(df.columns) == 1:
            col_name = df.columns[0]

            train_size = int(len(df) * 0.8)
            train_data = df.iloc[:train_size]
            test_data = df.iloc[train_size:]

            st.session_state["train_data"] = train_data
            st.session_state["test_data"] = test_data

            st.success("✅ Data berhasil di-split dengan rasio 80% training dan 20% testing.")

            st.subheader("Data Training:")
            st.write(train_data.tail())
            st.line_chart(train_data)

            st.subheader("Data Testing:")
            st.write(test_data.head())
            st.line_chart(test_data)

        else:
            st.warning("⚠️ Data harus hanya memiliki 1 kolom target untuk proses split time series.")

    else:
        st.info("Silakan unggah data yang ingin Anda split.")

# =================== PREDIKSI ======================
elif menu == "PREDIKSI":
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.preprocessing import MinMaxScaler

    st.title("PREDIKSI PERMINTAAN DARAH MENGGUNAKAN ARIMA")

    train = st.session_state.get('train')
    test = st.session_state.get('test')

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

            # =========================
            # TAHAP LANJUT: RESIDUAL
            # =========================
            if st.button("Lanjutkan ke Residual ARIMA untuk ANFIS"):
                st.subheader("5. Residual dari Model ARIMA")

                # Ambil residual
                residual = model_arima.resid
                data_anfis = pd.DataFrame({'residual': residual})

                # Simpan ke session_state untuk ANFIS
                st.session_state['data_anfis_raw'] = data_anfis

                st.write("Residual ARIMA:")
                st.line_chart(data_anfis['residual'])

                st.subheader("6. Normalisasi Residual")
                scaler_residual = MinMaxScaler()
                data_anfis['residual'] = scaler_residual.fit_transform(data_anfis[['residual']])

                st.session_state['data_anfis'] = data_anfis
                st.session_state['scaler_residual'] = scaler_residual

                st.success("Residual berhasil dinormalisasi dan siap untuk tahap ANFIS.")

                # Preview hasil normalisasi
                st.write(data_anfis.head())

                # Navigasi manual selanjutnya
                st.info("Silakan lanjut ke menu ANFIS untuk pemodelan hybrid.")
    else:
        st.warning("Silakan lakukan data splitting terlebih dahulu.")


