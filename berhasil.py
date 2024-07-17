import pandas as pd
import joblib
import plotly.graph_objs as go
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
import streamlit as st
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Atur Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('data-monitoring-424622-f89eeef34709.json', scope)
client = gspread.authorize(creds)

# Memuat model dan scaler yang sudah disimpan
rf_temp = joblib.load('temp_model.joblib')
rf_volt = joblib.load('volt_model.joblib')
scaler = joblib.load('scaler_rf.joblib')

def prediksi_status(suhu, tegangan, model):
    data_input = pd.DataFrame([[suhu, tegangan]], columns=['Temperature', 'Voltage'])
    data_input_scaled = scaler.transform(data_input)
    status_terprediksi = model.predict(data_input_scaled)
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
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
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
    time_list = []
    data = bersihkan_data(data)
    if len(data) > 100:
        data = data.iloc[-100:]
    for index, row in data.iterrows():
        suhu = row['Temperature']
        tegangan = row['Voltage']
        time = row['Timestamp']
        # Ubah timestamp agar sesuai dengan waktu sekarang
        time = datetime.now().replace(hour=time.hour, minute=time.minute, second=time.second, microsecond=0)
        temp_status = prediksi_status(suhu, tegangan, rf_temp)
        volt_status = prediksi_status(suhu, tegangan, rf_volt)
        suhu_list.append(suhu)
        tegangan_list.append(tegangan)
        temp_status_list.append(temp_status)
        volt_status_list.append(volt_status)
        time_list.append(time)
        print(f"Waktu: {time}, Suhu: {suhu}, Tegangan: {tegangan}, TempStatus Terprediksi: {temp_status}, VoltStatus Terprediksi: {volt_status}")
    return suhu_list, tegangan_list, temp_status_list, volt_status_list, time_list

def plot_grafik(suhu_list, tegangan_list, temp_status_list, volt_status_list, time_list, chart_placeholder):
    chart_placeholder.empty()
    
    with chart_placeholder.container():
        # Plot Suhu
        fig_suhu = go.Figure()
        fig_suhu.add_trace(go.Scatter(
            x=time_list,
            y=suhu_list,
            mode='lines+markers',
            name='Suhu',
            line=dict(color='blue', shape='spline'),
            marker=dict(size=8, color='blue')
        ))

        fig_suhu.update_layout(
            title='Grafik Suhu',
            xaxis_title='Waktu',
            yaxis_title='Nilai Suhu',
            legend=dict(
                title=dict(text='Parameter', font=dict(size=12, color='white')),
                font=dict(
                    size=12,
                    color='white'
                )
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="white"
            )
        )
        
        st.plotly_chart(fig_suhu)

        # Plot Tegangan
        fig_tegangan = go.Figure()
        fig_tegangan.add_trace(go.Scatter(
            x=time_list,
            y=tegangan_list,
            mode='lines+markers',
            name='Tegangan',
            line=dict(color='red', shape='spline'),
            marker=dict(size=8, color='red')
        ))

        fig_tegangan.update_layout(
            title='Grafik Tegangan',
            xaxis_title='Waktu',
            yaxis_title='Nilai Tegangan',
            legend=dict(
                title=dict(text='Parameter', font=dict(size=12, color='white')),
                font=dict(
                    size=12,
                    color='white'
                )
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="white"
            )
        )
        
        st.plotly_chart(fig_tegangan)

        # Plot TempStatus
        fig_temp_status = go.Figure()
        fig_temp_status.add_trace(go.Scatter(
            x=time_list,
            y=temp_status_list,
            mode='lines+markers',
            name='TempStatus',
            line=dict(color='green', shape='spline'),
            marker=dict(size=8, color='green')
        ))

        fig_temp_status.update_layout(
            title='TempStatus Terprediksi',
            xaxis_title='Waktu',
            yaxis_title='TempStatus',
            legend=dict(
                title=dict(text='Parameter', font=dict(size=12, color='white')),
                font=dict(
                    size=12,
                    color='white'
                )
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="white"
            )
        )
        
        st.plotly_chart(fig_temp_status)
        
        # Plot VoltStatus
        fig_volt_status = go.Figure()
        fig_volt_status.add_trace(go.Scatter(
            x=time_list,
            y=volt_status_list,
            mode='lines+markers',
            name='VoltStatus',
            line=dict(color='purple', shape='spline'),
            marker=dict(size=8, color='purple')
        ))

        fig_volt_status.update_layout(
            title='VoltStatus Terprediksi',
            xaxis_title='Waktu',
            yaxis_title='VoltStatus',
            legend=dict(
                title=dict(text='Parameter', font=dict(size=12, color='white')),
                font=dict(
                    size=12,
                    color='white'
                )
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="white"
            )
        )
        st.plotly_chart(fig_volt_status)

        # Peringatan TempStatus rendah atau tinggi hanya untuk data terbaru
        if temp_status_list[-1] == 1:
            st.warning(f"Warning: TempStatus is LOW at {time_list[-1]}")
        elif temp_status_list[-1] == 3:
            st.error(f"Alert: TempStatus is HIGH at {time_list[-1]}")
        else:
            st.success(f"TempStatus is NORMAL at {time_list[-1]}")

        # Peringatan VoltStatus rendah atau tinggi hanya untuk data terbaru
        if volt_status_list[-1] == 1:
            st.warning(f"Warning: VoltStatus is LOW at {time_list[-1]}")
        elif volt_status_list[-1] == 3:
            st.error(f"Alert: VoltStatus is HIGH at {time_list[-1]}")
        else:
            st.success(f"VoltStatus is NORMAL at {time_list[-1]}")

def plot_prediksi_30_hari(data, chart_placeholder):
    future_suhu, future_tegangan = generate_future_data(data)
    future_temp_status_list = []
    future_volt_status_list = []
    time_list = []
    start_time = datetime.now() + timedelta(hours=1)  # Menggunakan waktu sekarang untuk prediksi ke depan
    for i, (suhu, tegangan) in enumerate(zip(future_suhu, future_tegangan)):
        temp_status = prediksi_status(suhu, tegangan, rf_temp)
        volt_status = prediksi_status(suhu, tegangan, rf_volt)
        time = start_time + timedelta(days=i)
        time = time.replace(microsecond=0)
        future_temp_status_list.append(temp_status)
        future_volt_status_list.append(volt_status)
        time_list.append(time)
        print(f"Prediksi 30 Hari - Waktu: {time}, Suhu: {suhu}, Tegangan: {tegangan}, TempStatus Terprediksi: {temp_status}, VoltStatus Terprediksi: {volt_status}")
    
    chart_placeholder.empty()
    
    with chart_placeholder.container():
        # Plot Suhu Prediksi
        fig_suhu_prediksi = go.Figure()
        fig_suhu_prediksi.add_trace(go.Scatter(
            x=time_list,
            y=future_suhu,
            mode='lines+markers',
            name='Suhu Prediksi',
            line=dict(color='blue', shape='spline'),
            marker=dict(size=8, color='blue')
        ))

        fig_suhu_prediksi.update_layout(
            title='Grafik Prediksi Suhu 30 Hari ke Depan',
            xaxis_title='Waktu',
            yaxis_title='Nilai Suhu',
            legend=dict(
                title=dict(text='Parameter', font=dict(size=12, color='white')),
                font=dict(
                    size=12,
                    color='white'
                )
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="white"
            )
        )
        
        st.plotly_chart(fig_suhu_prediksi)

        # Plot Tegangan Prediksi
        fig_tegangan_prediksi = go.Figure()
        fig_tegangan_prediksi.add_trace(go.Scatter(
            x=time_list,
            y=future_tegangan,
            mode='lines+markers',
            name='Tegangan Prediksi',
            line=dict(color='red', shape='spline'),
            marker=dict(size=8, color='red')
        ))

        fig_tegangan_prediksi.update_layout(
            title='Grafik Prediksi Tegangan 30 Hari ke Depan',
            xaxis_title='Waktu',
            yaxis_title='Nilai Tegangan',
            legend=dict(
                title=dict(text='Parameter', font=dict(size=12, color='white')),
                font=dict(
                    size=12,
                    color='white'
                )
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="white"
            )
        )
        
        st.plotly_chart(fig_tegangan_prediksi)

        # Plot TempStatus Prediksi
        fig_temp_status_prediksi = go.Figure()
        fig_temp_status_prediksi.add_trace(go.Scatter(
            x=time_list,
            y=future_temp_status_list,
            mode='lines+markers',
            name='TempStatus Prediksi',
            line=dict(color='green', shape='spline'),
            marker=dict(size=8, color='green')
        ))

        fig_temp_status_prediksi.update_layout(
            title='TempStatus Terprediksi 30 Hari ke Depan',
            xaxis_title='Waktu',
            yaxis_title='TempStatus',
            legend=dict(
                title=dict(text='Parameter', font=dict(size=12, color='white')),
                font=dict(
                    size=12,
                    color='white'
                )
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="white"
            )
        )
        
        st.plotly_chart(fig_temp_status_prediksi)
        
        # Plot VoltStatus Prediksi
        fig_volt_status_prediksi = go.Figure()
        fig_volt_status_prediksi.add_trace(go.Scatter(
            x=time_list,
            y=future_volt_status_list,
            mode='lines+markers',
            name='VoltStatus Prediksi',
            line=dict(color='purple', shape='spline'),
            marker=dict(size=8, color='purple')
        ))

        fig_volt_status_prediksi.update_layout(
            title='VoltStatus Terprediksi 30 Hari ke Depan',
            xaxis_title='Waktu',
            yaxis_title='VoltStatus',
            legend=dict(
                title=dict(text='Parameter', font=dict(size=12, color='white')),
                font=dict(
                    size=12,
                    color='white'
                )
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='grey',
                tickfont=dict(color='white'),
                titlefont=dict(color='white')
            ),
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="white"
            )
        )
        
        st.plotly_chart(fig_volt_status_prediksi)

def perbarui_visualisasi(sheet, chart_placeholder, last_data):
    data = pd.DataFrame(sheet.get_all_records())
    if not data.equals(last_data):
        suhu_list, tegangan_list, temp_status_list, volt_status_list, time_list = proses_spreadsheet(data)
        plot_grafik(suhu_list, tegangan_list, temp_status_list, volt_status_list, time_list, chart_placeholder)
    else:
        # Update grafik meskipun tidak ada perubahan data
        suhu_list, tegangan_list, temp_status_list, volt_status_list, time_list = proses_spreadsheet(data)
        plot_grafik(suhu_list, tegangan_list, temp_status_list, volt_status_list, time_list, chart_placeholder)
    return data

def main():
    st.set_page_config(page_title="Aplikasi Monitoring Suhu dan Tegangan", layout="wide", initial_sidebar_state="expanded", page_icon="🐣")
    
    # Sidebar dengan logo dan gambar di atas menu
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <img src="https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/Exifa.gif" alt="logo" style="width: 60px; height: 60px;">
                    <span style="font-family: 'Cursive', sans-serif; font-size: 24px; color: white; animation: fadeIn 2s infinite;">𝔹𝕦𝕚𝕝𝕥 𝕓𝕪 𝕊𝕪𝕒𝕙𝕣𝕦𝕝 𝔸𝕓𝕚𝕕𝕚𝕟 𝔸 𝕐𝕒𝕟𝕚</span>
                </div>
                <div style="margin-top: 20px;">
                    <img src="https://img.freepik.com/premium-photo/sweet-funny-baby-chick-wearing-fashion-sunglasses-generative-ai_666746-909.jpg?w=826" style="width: 100%;">
                </div>
            </div>
            <style>
                @keyframes fadeIn {
                    0% { opacity: 0; }
                    50% { opacity: 1; }
                    100% { opacity: 0; }
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        with st.expander("台 Menu"):
            main_page = st.selectbox("Pilih Halaman Utama", ["Home", "Tentang"])
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        with st.expander("🛠 Model Configuration"):
            model_page = st.selectbox("Pilih Halaman Model", ["Monitoring", "Prediksi 30 Hari"])
        
        # Menambahkan logo WhatsApp, Instagram, dan Email GIF dengan jarak dan posisi
        st.markdown(
            """
            <div style="margin-top: 20px;">
                <a href="https://wa.me/085890243536" target="_blank" style="margin-right: 20px;">
                    <img src="https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/topmate.gif" alt="WhatsApp" style="width: 40px; height: 40px;">
                </a>
                <a href="https://www.instagram.com/oyyrulll" target="_blank" style="margin-right: 20px;">
                    <img src="https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/newsletter.gif" alt="Instagram" style="width: 40px; height: 40px;">
                </a>
                <a href="https://mail.google.com/mail/?view=cm&fs=1&to=syahrul.abidin234@gmail.com" target="_blank">
                    <img src="https://raw.githubusercontent.com/sahirmaharaj/exifa/main/img/email.gif" alt="Email" style="width: 40px; height: 40px;">
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if main_page == "Home":
        st.write("<h1 style='text-align: center; color: white;'>Aplikasi Monitoring Suhu dan Tegangan</h1>", unsafe_allow_html=True)
        st.write("<div style='text-align: center; color: white;'>Aplikasi Ini Memungkinkan Anda Untuk memonitoring dan memprediksi temperature dan voltage untuk 30 hari</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: center; color: white;'>Status yang ditampilkan: 1 untuk low, 2 untuk normal, dan 3 untuk high.</div>", unsafe_allow_html=True)

    elif main_page == "Tentang":
        st.write("<h1 style='text-align: center; color: white;'>Aplikasi monitoring suhu dan tegangan</h1>", unsafe_allow_html=True)
        st.write("""
        ### Tentang Aplikasi Monitoring
        Aplikasi ini dirancang untuk memonitor suhu dan tegangan secara real-time menggunakan data yang diambil dari sensor dan disimpan di Google Sheets.
        
        **Fitur Utama:**
        - Monitoring suhu dan tegangan secara real-time.
        - Prediksi suhu dan tegangan untuk 30 hari ke depan.
        - Visualisasi data dalam bentuk grafik yang mudah dipahami.
        - Pemberitahuan otomatis jika ada kondisi abnormal pada suhu atau tegangan.
        
        **Teknologi yang Digunakan:**
        - Google Sheets API untuk penyimpanan data.
        - Model Machine Learning untuk prediksi status suhu dan tegangan.
        - Streamlit untuk tampilan antarmuka pengguna yang interaktif.
                 
        **Pengembang**
        syahrul abidin a yani
        """)

    if model_page == "Monitoring":
        st.write("<h1 style='text-align: center; color: white;'>Monitoring suhu dan tegangan</h1>", unsafe_allow_html=True)
        sheet_url = 'https://docs.google.com/spreadsheets/d/1t3iwJI4UICYilpjplZ2KbGwJ4MQEsbCWL2AGaXvX_mQ/edit#gid=0'
        sheet = client.open_by_url(sheet_url).sheet1
        chart_placeholder = st.empty()
        last_data = pd.DataFrame()

        # Placeholder untuk memulai/menghentikan pembaruan otomatis
        auto_update = st.checkbox('Mulai Pembaruan Otomatis', value=True)
        
        while auto_update:
            with chart_placeholder.container():
                last_data = perbarui_visualisasi(sheet, chart_placeholder, last_data)
            time.sleep(15)  # Check for updates every 15 seconds
        
        st.write("Pembaruan otomatis dihentikan.")
    
    if model_page == "Prediksi 30 Hari":
        st.write("<h1 style='text-align: center; color: white;'>Prediksi 30 hari</h1>", unsafe_allow_html=True)
        sheet_url = 'https://docs.google.com/spreadsheets/d/1t3iwJI4UICYilpjplZ2KbGwJ4MQEsbCWL2AGaXvX_mQ/edit#gid=0'
        sheet = client.open_by_url(sheet_url).sheet1
        data = pd.DataFrame(sheet.get_all_records())
        data = bersihkan_data(data)
        chart_placeholder = st.empty()
        plot_prediksi_30_hari(data, chart_placeholder)

if __name__ == "__main__":
    main()
