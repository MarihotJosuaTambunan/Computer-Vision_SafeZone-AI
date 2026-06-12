import multiprocessing
import os
import sys
import uvicorn

def start_fastapi():
    print("⏳ Menyalakan Backend FastAPI pada port 8000...")
    # Menjalankan FastAPI secara terprogram
    uvicorn.run("apps.main:app", host="0.0.0.0", port=8000, log_level="info")

def start_streamlit(port):
    print(f"⏳ Menyalakan Frontend Streamlit pada port {port}...")
    # Menjalankan Streamlit menggunakan command line internal Python
    os.system(f"streamlit run apps/frontend.py --server.port {port} --server.address 0.0.0.0 --server.headless true")

if __name__ == "__main__":
    # Railway akan menyuntikkan variabel PORT secara otomatis untuk aplikasi utama (Streamlit)
    # Jika tidak ada (run lokal), default ke port 8501
    port_streamlit = int(os.getenv("PORT", 8501))

    # Membuat proses terpisah untuk masing-masing aplikasi
    process_fastapi = multiprocessing.Process(target=start_fastapi)
    process_streamlit = multiprocessing.Process(target=start_streamlit, args=(port_streamlit,))

    # Memulai kedua proses secara bersamaan
    process_fastapi.start()
    process_streamlit.start()

    # Menjaga agar skrip utama tetap hidup selama proses anak berjalan
    process_fastapi.join()
    process_streamlit.start()