import cv2
import time
from typing import Generator

def _normalize_source(src: str):
    # Jika user memberi '0' atau '1', konversi ke int untuk webcam lokal
    if isinstance(src, str) and src.isdigit():
        return int(src)
    return src

def generate_camera_stream(inference_engine, rule_engine, source: str = "0", use_tracking: bool = True) -> Generator[bytes, None, None]:
    """
    Generator untuk streaming dari kamera (webcam lokal atau RTSP).

    - `source`: string yang dapat berupa indeks kamera ('0') atau URL RTSP.
    - `use_tracking`: jika True gunakan `track_frame`, jika False gunakan `predict_frame`.
    """
    src = _normalize_source(source)
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka source kamera: {source}")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                # Coba ulang sambungan beberapa detik
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(src)
                continue

            # Pilih mode inferensi
            if use_tracking and hasattr(inference_engine, 'track_frame'):
                raw_detections = inference_engine.track_frame(frame)
            else:
                raw_detections = inference_engine.predict_frame(frame)

            # Evaluasi aturan K3
            k3_results = rule_engine.evaluate_compliance_with_tracking(raw_detections)

            # Visualisasi deteksi ke frame (sama format seperti di main.generate_video_stream)
            for worker in k3_results:
                x1, y1, x2, y2 = worker['person_bbox']
                w_id = worker['worker_id']
                analysis = worker['k3_analysis']
                conf = worker.get('confidence', 0.0)
                color = (0, 255, 0) if analysis['safe_status'] == "AMAN" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"ID:{w_id} | H:{analysis['helmet']} | V:{analysis['vest']} | Conf:{conf:.2f}"
                cv2.rectangle(frame, (x1, y1 - 35), (x1 + 415, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        cap.release()
