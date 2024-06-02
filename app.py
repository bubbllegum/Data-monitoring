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
knn_temp = joblib.load('knn_temp_model1.joblib')
knn_volt = joblib.load('knn_volt_model1.joblib')
scaler = joblib.load('scaler1.joblib')

def prediksi_temp_status(suhu, tegangan):
    data_input = pd.DataFrame([[suhu, tegangan]], columns=['Temperature', 'Voltage'])
    data_input_scaled = scaler.transform(data_input)
    status_terprediksi = knn_temp.predict(data_input_scaled)
    return status_terprediksi[0]

def prediksi_volt_status(suhu, tegangan):
    data_input = pd.DataFrame([[suhu, tegangan]], columns=['Temperature', 'Voltage'])
    data_input_scaled = scaler.transform(data_input)
    status_terprediksi = knn_volt.predict(data_input_scaled)
    return status_terprediksi[0]

def interpret_status(status):
    if status == 2:
        return "normal"
    elif status == 1:
        return "low"
    elif status == 3:
        return "high"
    else:
        return "unknown"

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
    temp_status_list = []
    volt_status_list = []
    data = bersihkan_data(data)
    if len(data) > 50:
        data = data.iloc[-50:]
    for index, row in data.iterrows():
        suhu = row['Temperature']
        tegangan = row['Voltage']
        temp_status = prediksi_temp_status(suhu, tegangan)
        volt_status = prediksi_volt_status(suhu, tegangan)
        suhu_list.append(suhu)
        tegangan_list.append(tegangan)
        temp_status_list.append(temp_status)
        volt_status_list.append(volt_status)
        print(f"Suhu: {suhu}, Tegangan: {tegangan}, TempStatus Terprediksi: {temp_status}, VoltStatus Terprediksi: {volt_status}")
    return suhu_list, tegangan_list, temp_status_list, volt_status_list

def plot_grafik(suhu_list, tegangan_list, temp_status_list, volt_status_list, chart_placeholder):
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
        
        # Plot TempStatus
        fig_temp_status, ax_temp_status = plt.subplots(figsize=(12, 6))
        ax_temp_status.plot(temp_status_list, label='TempStatus', color='tab:green')
        ax_temp_status.set_xlabel('Index')
        ax_temp_status.set_ylabel('TempStatus', color='tab:green')
        ax_temp_status.set_title('TempStatus Terprediksi')
        ax_temp_status.legend()
        ax_temp_status.grid(True)
        st.pyplot(fig_temp_status)
        plt.close(fig_temp_status)
        
        # Plot VoltStatus
        fig_volt_status, ax_volt_status = plt.subplots(figsize=(12, 6))
        ax_volt_status.plot(volt_status_list, label='VoltStatus', color='tab:orange')
        ax_volt_status.set_xlabel('Index')
        ax_volt_status.set_ylabel('VoltStatus', color='tab:orange')
        ax_volt_status.set_title('VoltStatus Terprediksi')
        ax_volt_status.legend()
        ax_volt_status.grid(True)
        st.pyplot(fig_volt_status)
        plt.close(fig_volt_status)

def perbarui_visualisasi(sheet, chart_placeholder, last_data):
    data = pd.DataFrame(sheet.get_all_records())
    if not data.equals(last_data):
        suhu_list, tegangan_list, temp_status_list, volt_status_list = proses_spreadsheet(data)
        plot_grafik(suhu_list, tegangan_list, temp_status_list, volt_status_list, chart_placeholder)
    else:
        # Update grafik meskipun tidak ada perubahan data
        suhu_list, tegangan_list, temp_status_list, volt_status_list = proses_spreadsheet(data)
        plot_grafik(suhu_list, tegangan_list, temp_status_list, volt_status_list, chart_placeholder)
    return data

def plot_prediksi_30_hari(data, chart_placeholder):
    future_suhu, future_tegangan = generate_future_data(data)
    future_temp_status_list = []
    future_volt_status_list = []
    for suhu, tegangan in zip(future_suhu, future_tegangan):
        temp_status = prediksi_temp_status(suhu, tegangan)
        volt_status = prediksi_volt_status(suhu, tegangan)
        future_temp_status_list.append(temp_status)
        future_volt_status_list.append(volt_status)
        print(f"Prediksi 30 Hari - Suhu: {suhu}, Tegangan: {tegangan}, TempStatus Terprediksi: {temp_status}, VoltStatus Terprediksi: {volt_status}")
    
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
        
        # Plot TempStatus Prediksi
        fig_temp_status, ax_temp_status = plt.subplots(figsize=(12, 6))
        ax_temp_status.plot(future_temp_status_list, label='TempStatus', color='tab:green')
        ax_temp_status.set_xlabel('Hari ke-')
        ax_temp_status.set_ylabel('TempStatus', color='tab:green')
        ax_temp_status.set_title('TempStatus Terprediksi 30 Hari ke Depan')
        ax_temp_status.legend()
        ax_temp_status.grid(True)
        st.pyplot(fig_temp_status)
        plt.close(fig_temp_status)
        
        # Plot VoltStatus Prediksi
        fig_volt_status, ax_volt_status = plt.subplots(figsize=(12, 6))
        ax_volt_status.plot(future_volt_status_list, label='VoltStatus', color='tab:orange')
        ax_volt_status.set_xlabel('Hari ke-')
        ax_volt_status.set_ylabel('VoltStatus', color='tab:orange')
        ax_volt_status.set_title('VoltStatus Terprediksi 30 Hari ke Depan')
        ax_volt_status.legend()
        ax_volt_status.grid(True)
        st.pyplot(fig_volt_status)
        plt.close(fig_volt_status)

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
            time.sleep(12)  # Check for updates every 12 seconds
        
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
