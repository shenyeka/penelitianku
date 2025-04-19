import streamlit as st
import pandas as pd
import math
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Inisialisasi session_state
if "step" not in st.session_state:
    st.session_state.step = 1

st.title("Prediksi Permintaan Darah - ARIMA ANFIS ABC")

# Fungsi tombol navigasi
def tombol_navigasi():
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.session_state.step > 1:
            if st.button("⬅️ Kembali"):
                st.session_state.step -= 1
    with col2:
        if st.session_state.step < 7:
            if st.button("Lanjut ➡️"):
                st.session_state.step += 1

# STEP 1: Upload Dataset
if st.session_state.step == 1:
    st.subheader("1. Upload Dataset")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.write("Contoh data:")
        st.write(df.head())
    tombol_navigasi()

# STEP 2: Preprocessing Data
elif st.session_state.step == 2:
    st.subheader("2. Preprocessing Data")
    if "df" in st.session_state:
        df = st.session_state["df"]
        if 'Bulan' in df.columns:
            df['Bulan'] = pd.to_datetime(df['Bulan'])
            df = df.sort_values('Bulan')
            df = df.set_index('Bulan')
            st.session_state["df"] = df
            st.write(df.head())
        else:
            st.error("Kolom 'Bulan' tidak ditemukan dalam dataset.")
    else:
        st.warning("Silakan upload dataset terlebih dahulu.")
    tombol_navigasi()

# STEP 3: Plot Data
elif st.session_state.step == 3:
    st.subheader("3. Plot Data Permintaan Darah")
    if "df" in st.session_state:
        df = st.session_state["df"]
        plt.figure(figsize=(10, 4))
        plt.plot(df.index, df[df.columns[0]], label='Permintaan Darah')
        plt.xlabel('Bulan')
        plt.ylabel('Jumlah')
        plt.title('Plot Permintaan Darah')
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.warning("Silakan upload dan preprocessing dataset terlebih dahulu.")
    tombol_navigasi()

# STEP 4: Pemodelan ARIMA
elif st.session_state.step == 4:
    st.subheader("4. Pemodelan ARIMA")
    if "df" in st.session_state:
        df = st.session_state["df"]
        ts = df[df.columns[0]]
        model = sm.tsa.ARIMA(ts, order=(1,1,1))
        model_fit = model.fit()
        st.write(model_fit.summary())
        st.session_state["residual"] = model_fit.resid
    else:
        st.warning("Silakan upload dan preprocessing dataset terlebih dahulu.")
    tombol_navigasi()

# STEP 5: Residual ARIMA
elif st.session_state.step == 5:
    st.subheader("5. Plot Residual ARIMA")
    if "residual" in st.session_state:
        resid = st.session_state["residual"]
        plt.figure(figsize=(10, 4))
        plt.plot(resid)
        plt.title("Residual ARIMA")
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.warning("Model ARIMA belum dijalankan.")
    tombol_navigasi()

# STEP 6: Pemodelan ANFIS + Optimasi ABC
elif st.session_state.step == 6:
    st.subheader("6. Pemodelan ANFIS + Optimasi ABC")
    st.info("Halaman ini akan digunakan untuk implementasi ANFIS + ABC. (Belum diimplementasikan)")
    tombol_navigasi()

# STEP 7: Selesai
elif st.session_state.step == 7:
    st.subheader("7. Selesai")
    st.success("Prediksi selesai. Anda dapat mengulang proses atau mengembangkan lebih lanjut.")
    tombol_navigasi()
