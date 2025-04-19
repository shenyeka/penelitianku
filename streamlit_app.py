import streamlit as st
import pandas as pd
import math
from pathlib import Path

import streamlit as st
import pandas as pd
import math
from pathlib import Path
import matplotlib.pyplot as plt

# Set tab title and favicon
st.set_page_config(
    page_title='Prediksi Permintaan Darah',
    page_icon='🩸'
)

# -----------------------------------------------------------------------------
# Fungsi ambil data
@st.cache_data
def get_blood_demand_data():
    DATA_FILENAME = Path(__file__).parent / 'data/blood_demand.csv'
    df = pd.read_csv(DATA_FILENAME)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])  # Pastikan kolom waktu dalam datetime
    return df

df = get_blood_demand_data()

# -----------------------------------------------------------------------------
# Judul halaman
st.markdown("""
# 🩸 Prediksi Permintaan Darah

Dashboard ini menampilkan hasil prediksi jumlah permintaan darah menggunakan model hybrid **ARIMA-ANFIS** yang dioptimasi dengan algoritma **Artificial Bee Colony (ABC)**.

Silakan pilih rentang waktu untuk melihat tren dan hasil prediksi.
""")

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Aplikasi Prediksi Permintaan Darah - ARIMA-ANFIS + ABC")
        self.geometry("800x600")
        self.frames = {}
        self.dataset = None
        self.create_menu()
        self.create_frames()
        self.show_frame("Welcome")

    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        menu = tk.Menu(menubar, tearoff=0)
        menu.add_command(label="Selamat Datang", command=lambda: self.show_frame("Welcome"))
        menu.add_command(label="Upload Dataset", command=lambda: self.show_frame("Upload"))
        menu.add_command(label="Preprocessing Data", command=lambda: self.show_frame("Preprocessing"))
        menu.add_command(label="Plot Data", command=lambda: self.show_frame("Plot"))
        menubar.add_cascade(label="Menu", menu=menu)

    def create_frames(self):
        self.frames["Welcome"] = WelcomeFrame(self)
        self.frames["Upload"] = UploadFrame(self)
        self.frames["Preprocessing"] = PreprocessingFrame(self)
        self.frames["Plot"] = PlotFrame(self)
        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, name):
        frame = self.frames[name]
        frame.tkraise()

class WelcomeFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        label = tk.Label(self, text="SELAMAT DATANG DI APLIKASI PREDIKSI PERMINTAAN DARAH", font=("Helvetica", 16), wraplength=600)
        label.pack(pady=200)

class UploadFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        label = tk.Label(self, text="Upload Dataset CSV", font=("Helvetica", 14))
        label.pack(pady=10)
        upload_btn = tk.Button(self, text="Pilih File", command=self.upload_dataset)
        upload_btn.pack(pady=10)
        self.filename_label = tk.Label(self, text="")
        self.filename_label.pack()

    def upload_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.master.dataset = pd.read_csv(file_path)
                self.filename_label.config(text=f"Berhasil memuat: {file_path}")
                messagebox.showinfo("Sukses", "Dataset berhasil dimuat!")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat dataset: {e}")

class PreprocessingFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        label = tk.Label(self, text="Preprocessing Data", font=("Helvetica", 14))
        label.pack(pady=10)
        process_btn = tk.Button(self, text="Proses Data", command=self.preprocess_data)
        process_btn.pack(pady=10)
        self.result_label = tk.Label(self, text="")
        self.result_label.pack()

    def preprocess_data(self):
        if self.master.dataset is not None:
            try:
                data = self.master.dataset.copy()
                data['Bulan'] = pd.to_datetime(data['Bulan'])
                data = data.set_index('Bulan')
                nulls = data.isnull().sum().sum()
                self.master.dataset = data
                self.result_label.config(text=f"Jumlah data null: {nulls}")
                messagebox.showinfo("Sukses", "Preprocessing selesai")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal preprocessing: {e}")
        else:
            messagebox.showwarning("Peringatan", "Silakan upload dataset terlebih dahulu")

class PlotFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        label = tk.Label(self, text="Plot Data Jumlah Permintaan Darah", font=("Helvetica", 14))
        label.pack(pady=10)
        plot_btn = tk.Button(self, text="Tampilkan Plot", command=self.plot_data)
        plot_btn.pack(pady=10)

    def plot_data(self):
        if self.master.dataset is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self.master.dataset, label='Jumlah permintaan darah')
            ax.set_title('Data Jumlah Permintaan Darah 2011-2024')
            ax.set_xlabel('Tahun')
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=self)
            canvas.draw()
            canvas.get_tk_widget().pack(pady=10)
        else:
            messagebox.showwarning("Peringatan", "Silakan upload dan preprocessing dataset terlebih dahulu")

if __name__ == "__main__":
    app = App()
    app.mainloop()
