import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='BD Predictions',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_gdp_data():
    """Grab GDP data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """

    # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
    DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
    raw_gdp_df = pd.read_csv(DATA_FILENAME)

    MIN_YEAR = 1960
    MAX_YEAR = 2022

    # The data above has columns like:
    # - Country Name
    # - Country Code
    # - [Stuff I don't care about]
    # - GDP for 1960
    # - GDP for 1961
    # - GDP for 1962
    # - ...
    # - GDP for 2022
    #
    # ...but I want this instead:
    # - Country Name
    # - Country Code
    # - Year
    # - GDP
    #
    # So let's pivot all those year-columns into two: Year and GDP
    gdp_df = raw_gdp_df.melt(
        ['Country Code'],
        [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
        'Year',
        'GDP',
    )

    # Convert years from string to integers
    gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

    return gdp_df

gdp_df = get_gdp_data()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :earth_americas: GDP dashboard

Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
But it's otherwise a great (and did I mention _free_?) source of data.
'''

# Add some spacing
''
''

min_value = gdp_df['Year'].min()
max_value = gdp_df['Year'].max()

from_year, to_year = st.slider(
    'Which years are you interested in?',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

countries = gdp_df['Country Code'].unique()

if not len(countries):
    st.warning("Select at least one country")

selected_countries = st.multiselect(
    'Which countries would you like to view?',
    countries,
    ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

''
''
''

# Filter the data
filtered_gdp_df = gdp_df[
    (gdp_df['Country Code'].isin(selected_countries))
    & (gdp_df['Year'] <= to_year)
    & (from_year <= gdp_df['Year'])
]

st.header('GDP over time', divider='gray')

''

st.line_chart(
    filtered_gdp_df,
    x='Year',
    y='GDP',
    color='Country Code',
)

''
''


first_year = gdp_df[gdp_df['Year'] == from_year]
last_year = gdp_df[gdp_df['Year'] == to_year]

st.header(f'GDP in {to_year}', divider='gray')

''

cols = st.columns(4)

for i, country in enumerate(selected_countries):
    col = cols[i % len(cols)]

    with col:
        first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
        last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

        if math.isnan(first_gdp):
            growth = 'n/a'
            delta_color = 'off'
        else:
            growth = f'{last_gdp / first_gdp:,.2f}x'
            delta_color = 'normal'

        st.metric(
            
            label=f'{country} GDP',
            value=f'{last_gdp:,.0f}B',
            delta=growth,
            delta_color=delta_color
        )

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
