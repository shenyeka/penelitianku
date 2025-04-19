import streamlit as st
import pandas as pd
import math
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Prediksi Permintaan Darah", page_icon="🩸")

st.title("🩸 Prediksi Permintaan Darah")
st.markdown("""
Selamat datang di dashboard prediksi permintaan darah!  
Gunakan menu navigasi di sebelah kiri untuk berpindah halaman:

- Upload & pilih data
- Lihat data aktual
- Lihat hasil prediksi
- Validasi hasil prediksi
- Lihat metrik akurasi

""")

import streamlit as st
import pandas as pd

st.header("📥 Import Data")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File berhasil diunggah!")
    st.dataframe(df)
    st.session_state["data"] = df

import streamlit as st
import pandas as pd

st.header("🧪 Pengujian Validitas")

if "data" in st.session_state:
    df = st.session_state["data"]
    st.write("Contoh hasil prediksi & aktual:")
    st.dataframe(df[['Tahun', 'Permintaan_Prediksi', 'Permintaan_Aktual']])

    # Contoh perhitungan MAPE dan RMSE
    from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

    try:
        mape = mean_absolute_percentage_error(df['Permintaan_Aktual'], df['Permintaan_Prediksi']) * 100
        rmse = mean_squared_error(df['Permintaan_Aktual'], df['Permintaan_Prediksi'], squared=False)

        st.metric("MAPE", f"{mape:.2f} %")
        st.metric("RMSE", f"{rmse:.2f}")
    except:
        st.warning("Data prediksi dan aktual belum lengkap.")
else:
    st.warning("Silakan upload data terlebih dahulu di halaman 'Import Data'.")


