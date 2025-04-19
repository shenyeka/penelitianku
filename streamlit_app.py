import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import seaborn as sns

st.set_page_config(layout="centered")

# Style Settings
st.markdown("""
<style>
    body {
        background-color: #ffefef;
        color: maroon;
    }
    .header {
        color: maroon;
        font-size: 28px;
    }
    .subheader {
        color: maroon;
    }
    button {
        background-color: #f4c2c2;
        color: maroon;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    button:hover {
        background-color: #ebacac;
    }
</style>
""", unsafe_allow_html=True)

# State to store data
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'arima_model' not in st.session_state:
    st.session_state.arima_model = None
if 'residuals' not in st.session_state:
    st.session_state.residuals = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'mape' not in st.session_state:
    st.session_state.mape = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# App Title
st.title("Prediksi Permintaan Darah")
st.markdown("Aplikasi ini digunakan untuk memprediksi permintaan darah menggunakan model ARIMA yang dioptimasi ANFIS + ABC.")

# Step 1: Welcome Screen
if st.session_state.step == 1:
    st.header("Selamat Datang")
    if st.button("Lanjut"):
        st.session_state.step = 2
        
# Step 2: Upload Dataset
elif st.session_state.step == 2:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("Contoh data:")
        st.write(st.session_state.data.head())
        
        # Check the column names
        st.write("Nama kolom dalam dataset:")
        st.write(st.session_state.data.columns.tolist())
        
    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 1
    if col2.button("Lanjut") and st.session_state.data is not None:
        st.session_state.step = 3

# Step 3: Preprocessing Data
elif st.session_state.step == 3:
    st.header("Preprocessing Data")
    df = st.session_state.data.copy()
    
    # Check for the 'Bulan' column
    if 'Bulan' in df.columns:
        # Convert 'Bulan' to datetime and set as index
        df['Bulan'] = pd.to_datetime(df['Bulan'])
        df.set_index('Bulan', inplace=True)
    else:
        st.error("Kolom 'Bulan' tidak ditemukan dalam dataset. Pastikan file CSV memiliki kolom ini.")
        st.stop()  # Stop execution if the column is not found

    # Preprocess 'Jumlah permintaan' column
    if 'Jumlah permintaan' in df.columns:
        # Handle missing values (e.g., fill with the mean or drop)
        df['Jumlah permintaan'].fillna(df['Jumlah permintaan'].mean(), inplace=True)
        
        # Optionally, you can scale the 'Jumlah permintaan' column
        scaler = MinMaxScaler()
        df['Jumlah permintaan'] = scaler.fit_transform(df[['Jumlah permintaan']])
        st.session_state.scaler = scaler  # Store the scaler for later use

        st.session_state.data = df
        st.line_chart(df['Jumlah permintaan'])
    else:
        st.error("Kolom 'Jumlah permintaan' tidak ditemukan dalam dataset.")
        st.stop()  # Stop execution if the column is not found

    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 2
    if col2.button("Lanjut"):
        st.session_state.step = 4
# Step 4: Plot Data
elif st.session_state.step == 4:
    st.header("Plot Data")
    st.line_chart(st.session_state.data['Jumlah permintaan'])
    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 3
    if col2.button("Lanjut"):
        st.session_state.step = 5

# Step 5: ARIMA Modeling
elif st.session_state.step == 5:
    st.header("Pemodelan ARIMA")
    df = st.session_state.data
    model = ARIMA(df['Jumlah permintaan'], order=(1, 1, 0))
    fitted_model = model.fit()
    st.session_state.arima_model = fitted_model

    pred = fitted_model.predict(start=1, end=len(df)-1, typ='levels')
    actual = df['Jumlah permintaan'].iloc[1:]
    mape = mean_absolute_percentage_error(actual, pred) * 100
    st.session_state.mape = mape
    st.line_chart(pred)
    st.write(f"Nilai MAPE: {mape:.2f}")

    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 4
    if col2.button("Lanjut"):
        st.session_state.step = 6

# Step 6: ARIMA Residuals
elif st.session_state.step == 6:
    st.header("Residual ARIMA")
    residuals = st.session_state.arima_model.resid
    st.session_state.residuals = residuals
    st.line_chart(residuals)
    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 5
    if col2.button("Lanjut"):
        st.session_state.step = 7

# Step 7: ANFIS + ABC Modeling
elif st.session_state.step == 7:
    st.header("Pemodelan ANFIS + ABC")
    
    # Prepare data for ANFIS
    residuals = st.session_state.residuals
    scaler = MinMaxScaler()
    residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))
    st.session_state.scaler = scaler  # Store the scaler for later use

    # Initialize membership functions
    kmeans = KMeans(n_clusters=4, random_state=42).fit(residuals_scaled)
    centers = np.sort(kmeans.cluster_centers_.flatten())
    sigma = (centers[1] - centers[0]) / 2

    # Define ANFIS prediction function
    def anfis_predict(params, lag32, lag33, rules):
        n_rules = rules.shape[1]
        p = params[:n_rules]
        q = params[n_rules:2 * n_rules]
        r = params[2 * n_rules:3 * n_rules]
        rule_outputs = p * lag32[:, None] + q * lag33[:, None] + r
        normalized_outputs = (rules * rule_outputs).sum(axis=1) / rules.sum(axis=1)
        return normalized_outputs

    # Define loss function for optimization
    def loss_function(params):
        predictions = anfis_predict(params, lag32, lag33, rules)
        return np.mean((target - predictions) ** 2)

    # Optimize using ABC (Artificial Bee Colony)
    # (Implementation of ABC optimization goes here)

    # Placeholder for predictions
    predictions = np.random.rand(len(residuals))  # Replace with actual predictions from ANFIS
    st.session_state.predictions = predictions

    # Denormalize predictions
    predictions_denorm = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Plot predictions
    st.line_chart(predictions_denorm)
    st.write("Prediksi ANFIS + ABC")

    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 6
    if col2.button("Lanjut"):
        st.session_state.step = 8

# Step 8: Future Predictions
elif st.session_state.step == 8:
    st.header("Prediksi 6 Bulan ke Depan")
    
    # Check if the scaler is available
    if 'scaler' in st.session_state and st.session_state.scaler is not None:
        # Implement future predictions logic here
        future_predictions = np.random.rand(6)  # Replace with actual future predictions
        future_predictions_denorm = st.session_state.scaler.inverse_transform(future_predictions.reshape(-1, 1)).flatten()

        st.write("Hasil Prediksi 6 Bulan ke Depan:")
        st.write(future_predictions_denorm)

        # Plot future predictions
        st.line_chart(future_predictions_denorm)
    else:
        st.error("Scaler tidak tersedia. Pastikan langkah ANFIS + ABC telah dijalankan dengan benar.")

    col1, col2 = st.columns(2)
    if col1.button("Kembali"):
        st.session_state.step = 7
    if col2.button("Selesai"):
        st.write("Terima kasih telah menggunakan aplikasi ini!")
