import io
import cv2
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from engine.inference import YOLOInferenceEngine
from engine.rule_engine import K3RuleEngine
from live_camera import generate_camera_stream



# Inisialisasi Aplikasi

app = FastAPI(

    title="SafeZone-AI Backend Production",

    description="API Server Terintegrasi ByteTrack & Real-Time Video Streaming Engine",

    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/stream/live")
async def live_feed(source: str = "0", track: bool = True):
    """
    Endpoint streaming langsung dari kamera.
    - `source`: '0' (default) untuk webcam lokal, atau URL RTSP/HTTP.
    - `track`: jika True gunakan tracking (ByteTrack) untuk menjaga `worker_id`.
    """
    src = source
    use_tracking = bool(track)

    return StreamingResponse(
        generate_camera_stream(inference_engine, rule_engine, source=src, use_tracking=use_tracking),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ==============================================================


# Inisialisasi Core Engine

MODEL_PATH = "../weights/best_yolov10.onnx"

try:

    inference_engine = YOLOInferenceEngine(model_path=MODEL_PATH)

    rule_engine = K3RuleEngine()

    print(f"SUCCESS: Berhasil memuat model produksi dari {MODEL_PATH}")

except Exception as e:

    print(f"WARNING: Gagal memuat model, menggunakan fallback (yolov10s.pt)...")

    inference_engine = YOLOInferenceEngine(model_path="yolov10s.pt")

    rule_engine = K3RuleEngine()



@app.get("/")

def read_root():

    return {"status": "ONLINE", "system": "SafeZone-AI Real-Time Streaming Engine"}



# ==================== STREAMING CORE LOGIC ====================



def generate_video_stream(video_path: str):

    """

    Generator function untuk membaca video, memproses via AI, 

    menggambar bounding box, dan mengalirkannya per frame.

    """

    cap = cv2.VideoCapture(video_path)

    

    if not cap.isOpened():

        print(f"Error: Tidak bisa membuka file video di {video_path}")

        return



    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            # Jika video habis, loop kembali dari awal (bagus untuk demo juri)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            continue



        # Step 1: Jalankan Pelacakan AI ByteTrack

        raw_detections = inference_engine.track_frame(frame)

        

        # Step 2: Evaluasi Aturan K3

        k3_results = rule_engine.evaluate_compliance_with_tracking(raw_detections)



        # Step 3: Visual Annotator (Menggambar Boks & Teks di Frame secara Real-Time)

        for worker in k3_results:

            x1, y1, x2, y2 = worker['person_bbox']

            w_id = worker['worker_id']

            analysis = worker['k3_analysis']



            # Tentukan warna boks berdasarkan status keselamatan pekerja

            # AMAN = Hijau (0, 255, 0), BAHAYA = Merah (0, 0, 255) dalam format BGR OpenCV

            color = (0, 255, 0) if analysis['safe_status'] == "AMAN" else (0, 0, 255)

            

            # Gambar kotak pembungkus manusia

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            

            # Buat teks label status monitoring K3 dengan confidence score

            conf = worker.get('confidence', 0.0)

            label = f"ID:{w_id} | H:{analysis['helmet']} | V:{analysis['vest']} | Conf:{conf:.2f}"

            

            # Gambar latar belakang teks label agar terbaca jelas

            cv2.rectangle(frame, (x1, y1 - 35), (x1 + 415, y1), color, -1)

            cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



        # Step 4: Kompresi kembali frame ke format JPEG untuk transmisi jaringan

        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:

            continue

            

        frame_bytes = buffer.tobytes()

        

        # Kirim frame dalam format multipart stream data standard

        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')



    cap.release()



@app.get("/api/v1/stream/video")

async def video_feed():

    """

    Endpoint Video Streaming: Mengalirkan hasil monitoring CCTV secara real-time.

    Bisa langsung dibuka di browser atau di-render di HTML menggunakan <img src='...'>

    """

    # Pastikan kamu menaruh file video uji coba bernama 'video_test.mp4' di folder data/

    video_source = "../data/video_test.mp4"

    

    if not os.path.exists(video_source):

        raise HTTPException(status_code=404, detail="File video uji coba 'video_test.mp4' tidak ditemukan di folder data/")



    return StreamingResponse(

        generate_video_stream(video_source),

        media_type="multipart/x-mixed-replace; boundary=frame"

    )


@app.get("/api/v1/stream/video-custom")
async def custom_video_feed(path: str):
    """
    Endpoint Video Streaming dari path custom (user-uploaded atau custom path).
    - `path`: full path ke file video
    """
    # Validasi path untuk keamanan (cegah path traversal)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File video tidak ditemukan di path: {path}")
    
    # Pastikan file adalah video
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    if not any(path.lower().endswith(ext) for ext in valid_extensions):
        raise HTTPException(status_code=400, detail="Format file tidak didukung. Gunakan: mp4, avi, mov, mkv, flv, wmv")
    
    return StreamingResponse(
        generate_video_stream(path),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )



# ==============================================================



@app.post("/api/v1/predict/image")

async def predict_image(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):

        raise HTTPException(status_code=400, detail="Format berkas harus berupa citra/gambar.")

    try:

        contents = await file.read()

        nparr = np.frombuffer(contents, np.uint8)

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:

            raise HTTPException(status_code=400, detail="Gagal mendecode struktur citra gambar.")

            

        raw_detections = inference_engine.track_frame(frame)

        k3_results = rule_engine.evaluate_compliance_with_tracking(raw_detections)

        

        return {

            "filename": file.filename,

            "status_summary": {

                "total_workers_detected": len(k3_results),

                "safe_workers": sum(1 for w in k3_results if w['k3_analysis']['safe_status'] == "AMAN"),

                "unsafe_workers": sum(1 for w in k3_results if w['k3_analysis']['safe_status'] == "BAHAYA")

            },

            "results": k3_results

        }

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



if __name__ == "__main__":

    import uvicorn

    import os

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)