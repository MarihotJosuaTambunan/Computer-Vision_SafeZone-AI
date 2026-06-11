import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from engine.inference import YOLOInferenceEngine
from engine.rule_engine import K3RuleEngine

app = FastAPI(
    title="SafeZone-AI Backend Production",
    description="API Server untuk Deteksi Kepatuhan APD Pekerja secara Real-Time",
    version="1.0.0"
)

# Konfigurasi CORS agar bisa diakses oleh frontend (React/Vue/HTML biasa)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inisialisasi engine (Sesuaikan path bobot model YOLOv8 hasil training kamu)
MODEL_PATH = "/workspace/TA/weights/baseline_yolo.onnx"
try:
    inference_engine = YOLOInferenceEngine(model_path=MODEL_PATH)
    rule_engine = K3RuleEngine()
except Exception as e:
    print(f"Peringatan: Gagal memuat model dari {MODEL_PATH}, menggunakan model default fallback.")
    inference_engine = YOLOInferenceEngine(model_path="yolov8s.pt")
    rule_engine = K3RuleEngine()

@app.get("/")
def read_root():
    return {"status": "ONLINE", "system": "SafeZone-AI Backend Engine", "framework": "FastAPI"}

@app.post("/api/v1/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint utama untuk menerima unggahan gambar tunggal dari lapangan
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Format file harus berupa gambar.")
        
    try:
        # Baca file biner gambar ke format OpenCV
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Gagal mendekode gambar.")
            
        # Step 1: Jalankan Core Inference YOLOv8
        raw_detections = inference_engine.predict_frame(frame)
        
        # Step 2: Jalankan Rule Engine Evaluasi K3
        k3_results = rule_engine.evaluate_compliance(raw_detections)
        
        return {
            "filename": file.filename,
            "total_workers_detected": len(k3_results),
            "results": k3_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Jalankan server lokal pada port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)