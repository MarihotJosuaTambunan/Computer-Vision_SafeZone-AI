import cv2
import numpy as np
from ultralytics import YOLO

class YOLOInferenceEngine:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Inisialisasi core model objek deteksi APD SafeZone-AI
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        # Memuat model (Otomatis mendeteksi jika formatnya .pt atau .onnx)
        self.model = YOLO(model_path)
        
        # Mapping kelas sesuai dataset Fransiscus Rolanda Malau
        self.class_names = {
            0: 'helmet',
            1: 'vest',
            2: 'no-helmet',
            3: 'person',
            4: 'no-vest'
        }

    def predict_frame(self, frame: np.ndarray):
        """
        Menerima matriks frame gambar OpenCV dan mengembalikan koordinat boks deteksi
        """
        # Jalankan prediksi dengan ambang batas keyakinan (confidence)
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            # Ambil koordinat piksel absolut [x1, y1, x2, y2]
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            detections.append({
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, "unknown")
            })
            
        return detections