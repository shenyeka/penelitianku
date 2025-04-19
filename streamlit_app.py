import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# Fungsi untuk menampilkan plot data
def plot_data(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Bulan'], data['Jumlah permintaan'], label='Jumlah Permintaan')
    plt.title('Data Jumlah Permintaan Darah')
    plt.xlabel('Bulan')
    plt.ylabel('Jumlah Permintaan')
    plt.legend()
    st.pyplot(plt)

# Fungsi untuk membagi data menjadi train dan test
def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

# Fungsi untuk melatih model ARIMA
def train_arima(train_data):
    model_arima = ARIMA(train_data['Jumlah permintaan'], order=(1, 1, 0)).fit()
    return model_arima

# Fungsi untuk melakukan prediksi pada data testing
def predict_testing(model_arima, test_data):
    predictions = model_arima.forecast(steps=len(test_data))
    mape = mean_absolute_percentage_error(test_data['Jumlah permintaan'], predictions) * 100
    return predictions, mape

# Fungsi untuk menampilkan residual dari training ARIMA
def show_residual(model_arima):
    residuals = model_arima.resid
    plt.figure(figsize=(10, 5))
    plt.plot(residuals, label='Residual ARIMA', color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residual dari Model ARIMA')
    plt.xlabel('Index')
    plt.ylabel('Residual')
    plt.legend()
    st.pyplot(plt)

# Fungsi untuk melatih model ARIMA-ANFIS dengan optimasi ABC
def train_arima_anfis_abc(train_data):
    # Implementasi training ARIMA-ANFIS+ABC
    pass

# Fungsi untuk melakukan prediksi sesuai jumlah data testing
def predict_testing_arima_anfis_abc(model_arima_anfis_abc, test_data):
    # Implementasi prediksi ARIMA-ANFIS+ABC
    pass

# Fungsi untuk melakukan prediksi 6 bulan ke depan
def predict_6_bulan_ke_depan(model_arima_anfis_abc):
    # Implementasi prediksi 6 bulan ke depan ARIMA-ANFIS+ABC
    pass

# Main aplikasi
def main():
    st.title("Sistem Prediksi Permintaan Darah")
    
    # Menu
    menu = st.sidebar.selectbox("Menu", ["Start", "Upload File", "Tampilkan Plot Data", 
                                         "Splitting Data Train Test 80:20", "Lakukan Training dg ARIMA 1,1,0", 
                                         "Prediksi Data Testing dan Tampilkan MAPE beserta Plot", 
                                         "Tampilkan Residual dari Training ARIMA", "Lakukan Trainng dengan ARIMA-ANFIS+ABC", 
                                         "Lakukan Prediksi Sesuai Jumlah Data Testing Tampilkan MAPE dan Plot", 
                                         "Lakukan Prediksi 6 Bulan ke Depan dan Tampilkan Plot"])
    
    if menu == "Start":
        st.write("Selamat datang di Sistem Prediksi Permintaan Darah!")
    
    elif menu == "Upload File":
        file = st.file_uploader("Pilih file CSV", type=["csv"])
        if file is not None:
            data = pd.read_csv(file, sep=";")
            st.session_state.data = data  # Simpan data ke session state
            st.write("File berhasil diupload!")
    
    elif menu == "Tampilkan Plot Data":
        if 'data' in st.session_state:
            plot_data(st.session_state.data)
        else:
            st.write("Silakan upload file terlebih dahulu.")
    
    elif menu == "Splitting Data Train Test 80:20":
        if 'data' in st.session_state:
            train_data, test_data = split_data(st.session_state.data)
            st.session_state.train_data = train_data
            st.session_state.test_data = test_data
            st.write("Data berhasil dibagi menjadi train dan test.")
        else:
            st.write("Silakan upload file terlebih dahulu.")
    
    elif menu == "Lakukan Training dg ARIMA 1,1,0":
        if 'train_data' in st.session_state:
            model_arima = train_arima(st.session_state.train_data)
            st.session_state.model_arima = model_arima
            st.write("Model ARIMA berhasil dilatih.")
        else:
            st.write("Silakan bagi data terlebih dahulu.")
    
    elif menu == "Prediksi Data Testing dan Tampilkan MAPE beserta Plot":
        if 'model_arima' in st.session_state and 'test_data' in st.session_state:
            predictions, mape = predict_testing(st.session_state.model_arima, st.session_state.test_data)
            st.write(f"MAPE: {mape:.2f}%")
            plt.figure(figsize=(10, 5))
            plt.plot(st.session_state.test_data['Bulan'], st.session_state.test_data['Jumlah permintaan'], label='Data Aktual')
            plt.plot(st.session_state.test_data['Bulan'], predictions, label='Prediksi ARIMA', linestyle='--')
            plt.title('Prediksi Data Testing')
            plt.xlabel('Bulan')
            plt.ylabel('Jumlah Permintaan')
            plt.legend()
            st.pyplot(plt)
        else:
            st.write("Silakan latih model ARIMA terlebih dahulu.")
    
    elif menu == "Tampilkan Residual dari Training ARIMA":
        if 'model_arima' in st.session_state:
            show_residual(st.session_state.model_arima)
        else:
            st.write("Silakan latih model ARIMA terlebih dahulu.")
    
    elif menu == "Lakukan Trainng dengan ARIMA-ANFIS+ABC":
        if 'train_data' in st.session_state:
            model_arima_anfis_abc = train_arima_anfis_abc(st.session_state.train_data)
            st.session_state.model_arima_anfis_abc = model_arima_anfis_abc
            st.write("Model ARIMA-ANFIS+ABC berhasil dilatih.")
        else:
            st.write("Silakan bagi data terlebih dahulu.")
    
    elif menu == "Lakukan Prediksi Sesuai Jumlah Data Testing Tampilkan MAPE dan Plot":
        if 'model_arima_anfis_abc' in st.session_state and 'test_data' in st.session_state:
            predictions, mape = predict_testing_arima_anfis_abc(st.session_state.model_arima_anfis_abc, st.session_state.test_data)
            st.write(f"MAPE: {mape:.2f}%")
            plt.figure(figsize=(10, 5))
            plt.plot(st.session_state.test_data['Bulan'], st.session_state.test_data['Jumlah permintaan'], label='Data Aktual')
            plt.plot(st.session_state.test_data['Bulan'], predictions, label='Prediksi ARIMA-ANFIS+ABC', linestyle='--')
            plt.title('Prediksi Data Testing')
            plt.xlabel('Bulan')
            plt.ylabel('Jumlah Permintaan')
            plt.legend()
            st.pyplot(plt)
        else:
            st.write("Silakan latih model ARIMA-ANFIS+ABC terlebih dahulu.")
    
    elif menu == "Lakukan Prediksi 6 Bulan ke Depan dan Tampilkan Plot":
        if 'model_arima_anfis_abc' in st.session_state:
            predictions = predict_6_bulan_ke_depan(st.session_state.model_arima_anfis_abc)
            plt.figure(figsize=(10, 5))
            plt.plot(predictions, label='Prediksi 6 Bulan ke Depan', linestyle='--')
            plt.title('Prediksi 6 Bulan ke Depan')
            plt.xlabel('Bulan')
            plt.ylabel('Jumlah Permintaan')
            plt.legend()
            st.pyplot(plt)
        else:
            st.write("Silakan latih model ARIMA-ANFIS+ABC terlebih dahulu.")

if __name__ == "__main__":
    main()
