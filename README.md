# 🛡️ SafeZone-AI: Real-Time K3 APD Monitoring System

Sistem monitoring pintar berbasis **Computer Vision** dan **Deep Learning** untuk mendeteksi kepatuhan penggunaan **Helm** dan **Rompi Keselamatan (APD)** pekerja secara real-time.

## 📋 Fitur Utama

- ✅ **Real-Time Detection**: Deteksi APD (Helm & Rompi) dengan YOLO v10 ONNX
- ✅ **ByteTrack Integration**: Pelacakan individu pekerja lintas frame berbasis ByteTrack
- ✅ **Multi-Input Streaming**:
  - Live Camera (Webcam lokal atau RTSP stream)
  - Video File (Static CCTV recording)
  - Custom Video Upload
  - Single Photo Analysis
- ✅ **K3 Compliance Rules**: Evaluasi aturan keselamatan kerja otomatis
- ✅ **MJPEG Streaming**: Transmisi video real-time via MJPEG
- ✅ **REST API Backend**: FastAPI dengan CORS & multipart support
- ✅ **Streamlit Dashboard**: Frontend interaktif dengan metrics & live preview

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 atau lebih tinggi
- Virtual Environment (venv/conda)
- OpenCV 4.8+
- CUDA/GPU (optional untuk inference lebih cepat)

### 2. Instalasi Dependencies

```bash
# Aktifkan virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# atau
source .venv/bin/activate  # Linux/Mac

# Install semua dependencies
pip install -r requirements.txt
```

### 3. Menjalankan Sistem

**Terminal 1 - Backend API:**
```bash
cd apps
python main.py
# atau
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend Streamlit:**
```bash
cd apps
streamlit run frontend.py
```

Browser akan membuka otomatis di `http://localhost:8501`

---

## 📁 Struktur Proyek

```
Computer-Vision_SafeZone-AI/
├── apps/
│   ├── main.py              # FastAPI backend & streaming endpoints
│   ├── frontend.py          # Streamlit dashboard
│   ├── live_camera.py       # Live camera stream generator
│   ├── engine/
│   │   ├── inference.py     # YOLO inference engine + ByteTrack
│   │   └── rule_engine.py   # K3 compliance evaluation
│   └── yolov10s.pt         # Fallback model (if ONNX not found)
├── data/
│   ├── video_test.mp4      # Sample video untuk testing
│   └── ...
├── weights/
│   └── best_yolov10.onnx   # Production YOLO v10 model
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_modeling_yolov8.ipynb
├── requirements.txt        # Python dependencies
└── README.md              # Dokumentasi (file ini)
```

---

## 📚 Dataset

project ini menggunakan dataset "Personal Protective Equipment (PPE) Dataset" dari Kaggle sebagai sumber data untuk pelatihan dan evaluasi model.

Link dataset: https://www.kaggle.com/datasets/ndomalau/personal-protective-equipment-ppe-dataset



## 🎯 API Endpoints

### Backend FastAPI (port 8000)

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | Health check |
| GET | `/api/v1/stream/video` | Stream video dari file CCTV default |
| GET | `/api/v1/stream/video-custom?path={path}` | Stream custom video file |
| GET | `/api/v1/stream/live?source={src}&track={bool}` | Live camera streaming (webcam/RTSP) |
| POST | `/api/v1/predict/image` | Single image analysis |

### Live Camera Parameters

```
GET /api/v1/stream/live?source=0&track=true

Parameters:
- source: "0" (default webcam) atau URL RTSP (contoh: rtsp://user:pass@host:port/stream)
- track: true/false (gunakan ByteTrack untuk worker ID tracking)
```

### Image Upload Response

```json
{
  "filename": "image.jpg",
  "status_summary": {
    "total_workers_detected": 5,
    "safe_workers": 3,
    "unsafe_workers": 2
  },
  "results": [
    {
      "worker_id": 1,
      "person_bbox": [100, 200, 300, 500],
      "k3_analysis": {
        "helmet": "DETECTED",
        "vest": "NOT_DETECTED",
        "safe_status": "BAHAYA"
      }
    }
  ]
}
```

---

## 🖥️ Streamlit Dashboard

### Mode 1: Live CCTV Monitoring
- Pilih sumber input: **Video File (CCTV)**, **Upload Video**, atau **Live Camera**
- Aktifkan stream checkbox untuk mulai monitoring
- Lihat real-time detection dengan bounding box & status (AMAN/BAHAYA)
- Metrics otomatis untuk total pekerja, pekerja aman, dan pekerja melanggar

### Mode 2: Single Photo Analysis
- Upload foto pekerja (.jpg/.png)
- Lihat hasil deteksi beranotasi dengan bounding box, ID, helmet/vest status, dan **confidence score**
- JSON detail untuk setiap deteksi pekerja

## 📊 Detection Output

Setiap deteksi pekerja menampilkan:
- **ID**: Worker ID dari ByteTrack (konsisten lintas frame)
- **H**: Helmet status (PATUH/PELANGGARAN/TIDAK TERDETEKSI)
- **V**: Vest status (PATUH/PELANGGARAN/TIDAK TERDETEKSI)
- **Conf**: Confidence score dari model YOLO (0.00-1.00, semakin tinggi semakin yakin)

Contoh label di video stream:
```
ID:1 | H:PATUH | V:PATUH | Conf:0.92
```

---

## ⚙️ Konfigurasi

File konfigurasi utama di `apps/main.py`:

```python
MODEL_PATH = "../weights/best_yolov10.onnx"  # Path model produksi
CONF_THRESHOLD = 0.25                        # Confidence threshold YOLO
BACKEND_URL = "http://localhost:8000"        # Backend API URL (di frontend.py)
```

---

## 📊 Model Information

- **Architecture**: YOLOv10 (ONNX optimized)
- **Classes**: 
  - 0: Helmet
  - 1: No Helmet
  - 2: No Vest
  - 3: Person
  - 4: Vest
- **Input Size**: 640x640
- **Tracking**: ByteTrack (multi-object tracker)
- **Inference Speed**: ~30-50ms per frame (GPU accelerated)

---

## 🔧 Troubleshooting

### Model Tidak Ditemukan
```
WARNING: Gagal memuat model, menggunakan fallback (yolov10s.pt)...
```
**Solusi**: Pastikan file `weights/best_yolov10.onnx` ada di folder. Jika tidak, download atau gunakan fallback `yolov10s.pt`.

### Error: "Cannot import name 'YOLOInferenceEngine'"
**Solusi**: Pastikan Anda berada di folder `apps/` saat menjalankan backend:
```bash
cd apps
python main.py
```

### Streamlit Tidak Terhubung ke Backend
**Solusi**: Pastikan backend sudah running di terminal lain:
```bash
# Terminal 1
cd apps
python main.py

# Tunggu hingga melihat "SUCCESS: Berhasil memuat model"
```

### Live Camera Tidak Muncul
**Solusi**:
1. Pastikan webcam terinstall & accessible
2. Gunakan `source=0` untuk default webcam
3. Untuk RTSP, gunakan format: `rtsp://username:password@host:port/stream`
4. Check browser console untuk error message

### Video Upload Gagal di Streamlit
**Solusi**: 
- Pastikan format video support: `.mp4`, `.avi`, `.mov`, `.mkv`
- Ukuran file jangan >200MB untuk stabilitas
- Gunakan format MP4 H.264 codec untuk compatibility terbaik

---

## 📦 Dependencies

Semua dependencies sudah terdaftar di `requirements.txt`:

- **fastapi** — REST API framework
- **uvicorn** — ASGI server
- **opencv-python** — Video processing & image manipulation
- **numpy** — Numerical operations
- **streamlit** — Dashboard UI
- **requests** — HTTP client (untuk frontend)
- **ultralytics** — YOLO library & ByteTrack
- **python-multipart** — File upload support
- **Pillow** — Image processing

---

## 🎓 Development Notes

### Menambah Fitur Baru

1. **Buat endpoint baru di `main.py`**:
   ```python
   @app.post("/api/v1/custom-endpoint")
   async def custom_endpoint(data: dict):
       # Logika di sini
       return {"status": "ok"}
   ```

2. **Panggil dari Streamlit (`frontend.py`)**:
   ```python
   response = requests.post(f"{BACKEND_URL}/api/v1/custom-endpoint", json=data)
   ```

3. **Test dengan curl**:
   ```bash
   curl -X GET "http://localhost:8000/api/v1/stream/live?source=0&track=true"
   ```

### Modifying Inference Engine

Edit `apps/engine/inference.py` untuk:
- Ubah model
- Adjust confidence threshold
- Customize class mapping
- Change tracking algorithm

### Modifying Rules Engine

Edit `apps/engine/rule_engine.py` untuk:
- Tambah/ubah K3 compliance rules
- Customize alert logic
- Change status evaluation

---

## ♻️ Reproducible Evaluation

Use the provided reproducible evaluation script to run deterministic inference over a set of images, save predictions, annotated images, and a manifest that records environment metadata (git commit, tags, package versions, seed).

Script: `scripts/eval_reproducible.py`

Example usage:
```bash
# from project root
python scripts/eval_reproducible.py \
  --model weights/best_yolov10.onnx \
  --data-dir data/images \
  --output eval_out \
  --device cpu \
  --seed 42
```

Outputs written to `eval_out/`:
- `predictions.json` — per-image detections
- `annotated_images/` — images with drawn boxes and labels
- `manifest.json` — environment and run metadata (git commit, tags, package versions, params, seed)

Notes:
- Ensure the repository tag `submission-final` exists on the commit you want to record before running the script.
- For GPU inference set `--device cuda:0` (if available).
- The script attempts to capture package versions using importlib.metadata; make sure dependencies are installed in the environment used to run the script.


## 🔐 Security Notes

⚠️ **Production Deployment**:
- Jangan gunakan `reload=True` di uvicorn (gunakan production ASGI server seperti Gunicorn)
- Set `allow_origins` ke domain spesifik, bukan `"*"`
- Validasi semua input dari user
- Gunakan HTTPS/SSL untuk production
- Implement authentication untuk API endpoints

---

## 📄 License

Proprietary - SafeZone-AI Project

---

**Last Updated**: June 2026  
**Version**: 1.2.0
