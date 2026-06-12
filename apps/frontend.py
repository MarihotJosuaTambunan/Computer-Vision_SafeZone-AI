import streamlit as st
import os
import requests
import cv2
import numpy as np
import time
import urllib.parse

# 1. Konfigurasi Dasar Halaman Dashboard
st.set_page_config(
    page_title="SafeZone-AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Alamat URL API Backend FastAPI kita
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# 2. Desain Header & Title Dashboard
st.title("🛡️ SafeZone-AI: Real-Time K3 APD Monitoring System")
st.markdown("Sistem monitoring pintar berbasis Computer Vision untuk mendeteksi kepatuhan penggunaan Helm dan Rompi keselamatan pekerja.")
st.divider()

# 3. Pembuatan Sidebar Menu
st.sidebar.header("🕹️ Control Panel CCTV")
st.sidebar.markdown("Silakan pilih mode monitoring di bawah ini:")
app_mode = st.sidebar.selectbox(
    "Pilih Mode Analisis:",
    ["Dashboard Utama & Live CCTV", "Analisis Gambar Tunggal (Inspeksi)"]
)

# Cek Status Koneksi ke Backend FastAPI
try:
    response = requests.get(BACKEND_URL)
    if response.status_code == 200:
        st.sidebar.success("🟢 Status Core Engine: Connected")
except:
    st.sidebar.error("🔴 Status Core Engine: Disconnected (Nyalakan main.py!)")

# ==================== MODE 1: LIVE CCTV STREAMING ====================
if app_mode == "Dashboard Utama & Live CCTV":
    st.subheader("📹 Kamera Pengawas Lapangan - Live Stream")
    
    # Membuat Layout Kolom untuk Summary Cards Statistik
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Total Pekerja Terdeteksi", value="Mengalkulasi...", delta="Live")
    with col2:
        st.metric(label="Pekerja Patuh (Aman)", value="Mengalkulasi...", delta="Proteksi", delta_color="normal")
    with col3:
        st.metric(label="Pelanggaran APD (Bahaya)", value="Mengalkulasi...", delta="Segera Tindak", delta_color="inverse")
        
    st.markdown("---")
    
    # Pilih jenis input stream: Video file (CCTV), Upload Video, atau Live Camera (webcam/RTSP)
    input_type = st.selectbox("Pilih Sumber Input:", ["Video File (CCTV)", "Upload Video", "Live Camera (Webcam/RTSP)"])

    start_stream = st.checkbox("🔄 Aktifkan Stream", value=False)

    if start_stream:
        st.info("Menghubungkan ke jalur video streaming FastAPI...")

        if input_type == "Video File (CCTV)":
            stream_url = f"{BACKEND_URL}/api/v1/stream/video"

        elif input_type == "Upload Video":
            # Uploader untuk video yang di-custom upload user
            uploaded_video = st.file_uploader("Pilih file video (.mp4, .avi, .mov, .mkv):", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_video is not None:
                # Simpan video sementara ke folder temp
                import tempfile
                import os
                
                temp_dir = tempfile.gettempdir()
                temp_video_path = os.path.join(temp_dir, uploaded_video.name)
                
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                st.success(f"Video '{uploaded_video.name}' berhasil diunggah!")
                
                # Kirim path ke backend dengan URL encoding
                quoted_path = urllib.parse.quote_plus(temp_video_path)
                stream_url = f"{BACKEND_URL}/api/v1/stream/video-custom?path={quoted_path}"
            else:
                st.warning("Silakan pilih file video terlebih dahulu.")
                st.stop()

        else:  # Live Camera
            # Opsi source: default '0' (webcam). User bisa memasukkan RTSP/HTTP URL.
            src = st.text_input("Camera source (0 untuk webcam, atau masukkan RTSP/URL):", value="0")
            track = st.checkbox("Gunakan tracking (track IDs)", value=True)
            quoted = urllib.parse.quote_plus(src)
            stream_url = f"{BACKEND_URL}/api/v1/stream/live?source={quoted}&track={str(track).lower()}"

        # Tampilkan MJPEG stream di dalam tag <img>
        st.markdown(
            f'<div style="display: flex; justify-content: center;">'
            f'<img src="{stream_url}" width="85%" style="border-radius: 10px; border: 3px solid #343a40;">'
            f'</div>',
            unsafe_allow_html=True
        )

    else:
        st.warning("Stream dinonaktifkan. Centang boks di atas untuk memulai monitoring.")

# ==================== MODE 2: ANALISIS GAMBAR TUNGGAL ====================
elif app_mode == "Analisis Gambar Tunggal (Inspeksi)":
    st.subheader("📸 Inspeksi Foto Kepatuhan Pekerja")
    st.markdown("Unggah foto kondisi pekerja di lapangan untuk melakukan audit kepatuhan APD instan.")
    
    uploaded_file = st.file_uploader("Pilih file gambar (.jpg, .png, .jpeg):", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Tampilkan gambar asli di sisi kiri
        col_img, col_json = st.columns([2, 1])
        
        with col_img:
            st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)
            
        with col_json:
            st.markdown("### 🔍 Hasil Audit AI Engine")
            with st.spinner("Mengevaluasi koordinat spasial APD..."):
                # Kirim gambar ke FastAPI endpoint HTTP POST
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    res = requests.post(f"{BACKEND_URL}/api/v1/predict/image", files=files)
                    if res.status_code == 200:
                        output_data = res.json()

                        # Tampilkan Ringkasan
                        summary = output_data["status_summary"]
                        st.success(f"Total Pekerja: {summary['total_workers_detected']}")
                        st.info(f"Pekerja Aman: {summary['safe_workers']}")
                        if summary['unsafe_workers'] > 0:
                            st.error(f"Pekerja Bahaya (Melanggar): {summary['unsafe_workers']}")

                        # Buat anotasi pada gambar asli
                        img_bytes = uploaded_file.getvalue()
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is None:
                            st.error("Gagal mendecode gambar untuk anotasi.")
                        else:
                            annotated = img.copy()
                            for worker in output_data.get("results", []):
                                bbox = worker.get("person_bbox", [])
                                analysis = worker.get("k3_analysis", {})
                                wid = worker.get("worker_id", "-")
                                conf = worker.get("confidence", 0.0)
                                if len(bbox) == 4:
                                    x1, y1, x2, y2 = map(int, bbox)
                                    # Warna: hijau = AMAN, merah = BAHAYA
                                    color = (0, 255, 0) if analysis.get("safe_status") == "AMAN" else (0, 0, 255)
                                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                                    label = f"ID:{wid} H:{analysis.get('helmet')} V:{analysis.get('vest')} Conf:{conf:.2f}"
                                    # teks background
                                    cv2.rectangle(annotated, (x1, max(0, y1-28)), (x1+440, y1), color, -1)
                                    cv2.putText(annotated, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

                            # Convert BGR->RGB for Streamlit
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                            # Tampilkan gambar beranotasi di atas JSON
                            st.image(annotated_rgb, caption="Gambar Beranotasi (Hasil AI)", use_column_width=True)

                        # Tampilkan detail data JSON interaktif
                        st.markdown("**Detail Log Spasial:**")
                        st.json(output_data["results"])
                    else:
                        st.error(f"Gagal memproses gambar. Log API: {res.text}")
                except Exception as e:
                    st.error(f"Tidak dapat terhubung ke server API. Detail: {str(e)}")