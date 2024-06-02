import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import streamlit as st

# Atur Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('data-monitoring-424622-f89eeef34709.json', scope)
client = gspread.authorize(creds)

# Memuat model dan scaler yang sudah disimpan
knn = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

def prediksi_status(suhu, tegangan):
    data_input = pd.DataFrame([[suhu, tegangan]], columns=['Temperature', 'Voltage'])
    data_input_scaled = scaler.transform(data_input)
    status_terprediksi = knn.predict(data_input_scaled)
    return status_terprediksi[0]

def bersihkan_data(data):
    data = data.dropna()
    data['Temperature'] = data['Temperature'].astype(float)
    data['Voltage'] = data['Voltage'].astype(float)
    data = data.drop_duplicates()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return data

def generate_future_data(current_data, days=30):
    suhu_mean = current_data['Temperature'].mean()
    suhu_std = current_data['Temperature'].std()
    tegangan_mean = current_data['Voltage'].mean()
    tegangan_std = current_data['Voltage'].std()
    future_suhu = np.random.normal(suhu_mean, suhu_std, days)
    future_tegangan = np.random.normal(tegangan_mean, tegangan_std, days)
    return future_suhu, future_tegangan

def proses_spreadsheet(data):
    suhu_list = []
    tegangan_list = []
    status_list = []
    data = bersihkan_data(data)
    if len(data) > 50:
        data = data.iloc[-50:]
    for index, row in data.iterrows():
        suhu = row['Temperature']
        tegangan = row['Voltage']
        status = prediksi_status(suhu, tegangan)
        suhu_list.append(suhu)
        tegangan_list.append(tegangan)
        status_list.append(status)
        print(f"Suhu: {suhu}, Tegangan: {tegangan}, Status Terprediksi: {status}")
    return suhu_list, tegangan_list, status_list

def plot_grafik(suhu_list, tegangan_list, status_list, chart_placeholder):
    chart_placeholder.empty()
    
    # Plot Suhu dan Tegangan
    with chart_placeholder.container():
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Suhu', color='tab:blue')
        ax1.plot(suhu_list, label='Suhu', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Tegangan', color='tab:red')
        ax2.plot(tegangan_list, label='Tegangan', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        fig.tight_layout()
        plt.title('Grafik Suhu dan Tegangan')
        st.pyplot(fig)
        plt.close(fig)
        
        st.empty()  # Menambahkan spasi antara grafik
        
        # Plot Status
        fig_status, ax_status = plt.subplots(figsize=(12, 6))
        ax_status.plot(status_list, label='Status', color='tab:green')
        ax_status.set_xlabel('Index')
        ax_status.set_ylabel('Status', color='tab:green')
        ax_status.set_title('Status Terprediksi')
        ax_status.legend()
        ax_status.grid(True)
        st.pyplot(fig_status)
        plt.close(fig_status)

def perbarui_visualisasi(sheet, chart_placeholder, last_data):
    data = pd.DataFrame(sheet.get_all_records())
    if not data.equals(last_data):
        suhu_list, tegangan_list, status_list = proses_spreadsheet(data)
        plot_grafik(suhu_list, tegangan_list, status_list, chart_placeholder)
    else:
        # Update grafik meskipun tidak ada perubahan data
        suhu_list, tegangan_list, status_list = proses_spreadsheet(data)
        plot_grafik(suhu_list, tegangan_list, status_list, chart_placeholder)
    return data

def plot_prediksi_30_hari(data, chart_placeholder):
    future_suhu, future_tegangan = generate_future_data(data)
    future_status_list = []
    for suhu, tegangan in zip(future_suhu, future_tegangan):
        status = prediksi_status(suhu, tegangan)
        future_status_list.append(status)
        print(f"Prediksi 30 Hari - Suhu: {suhu}, Tegangan: {tegangan}, Status Terprediksi: {status}")
    
    chart_placeholder.empty()
    
    # Plot Suhu dan Tegangan Prediksi
    with chart_placeholder.container():
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Hari ke-')
        ax1.set_ylabel('Suhu', color='tab:blue')
        ax1.plot(future_suhu, label='Suhu', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Tegangan', color='tab:red')
        ax2.plot(future_tegangan, label='Tegangan', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        fig.tight_layout()
        plt.title('Grafik Prediksi Suhu dan Tegangan 30 Hari ke Depan')
        st.pyplot(fig)
        plt.close(fig)
        
        st.empty()  # Menambahkan spasi antara grafik
        
        # Plot Status Prediksi
        fig_status, ax_status = plt.subplots(figsize=(12, 6))
        ax_status.plot(future_status_list, label='Status', color='tab:green')
        ax_status.set_xlabel('Hari ke-')
        ax_status.set_ylabel('Status', color='tab:green')
        ax_status.set_title('Status Terprediksi 30 Hari ke Depan')
        ax_status.legend()
        ax_status.grid(True)
        st.pyplot(fig_status)
        plt.close(fig_status)

# Fungsi utama untuk aplikasi Streamlit
def main():
    st.title("Aplikasi Monitoring Suhu dan Tegangan")
    menu = ["Home", "Monitoring", "Prediksi 30 Hari"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.write("Selamat datang di aplikasi monitoring suhu dan tegangan.")
    elif choice == "Monitoring":
        st.subheader("Monitoring")
        sheet_url = 'https://docs.google.com/spreadsheets/d/1t3iwJI4UICYilpjplZ2KbGwJ4MQEsbCWL2AGaXvX_mQ/edit#gid=0'
        sheet = client.open_by_url(sheet_url).sheet1
        chart_placeholder = st.empty()
        last_data = pd.DataFrame()

        # Placeholder untuk memulai/menghentikan pembaruan otomatis
        auto_update = st.checkbox('Mulai Pembaruan Otomatis', value=True)
        
        while auto_update:
            with chart_placeholder.container():
                last_data = perbarui_visualisasi(sheet, chart_placeholder, last_data)
            time.sleep(12)  # Check for updates every 5 seconds
        
        st.write("Pembaruan otomatis dihentikan.")
    
    elif choice == "Prediksi 30 Hari":
        st.subheader("Prediksi 30 Hari ke Depan")
        sheet_url = 'https://docs.google.com/spreadsheets/d/1t3iwJI4UICYilpjplZ2KbGwJ4MQEsbCWL2AGaXvX_mQ/edit#gid=0'
        sheet = client.open_by_url(sheet_url).sheet1
        data = pd.DataFrame(sheet.get_all_records())
        data = bersihkan_data(data)
        chart_placeholder = st.empty()
        plot_prediksi_30_hari(data, chart_placeholder)

if __name__ == "__main__":
    main()
