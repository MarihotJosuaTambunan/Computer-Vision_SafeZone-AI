import cv2
import numpy as np
from ultralytics import YOLO

class YOLOInferenceEngine:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Inisialisasi core model objek deteksi APD dengan dukungan ByteTrack Tracking
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        
        # Mapping kelas sesuai dataset Fransiscus Rolanda Malau
        self.class_names = {
            0: 'helmet',
            1: 'vest',
            2: 'no-helmet',
            3: 'person',
            4: 'no-vest'
        }

    def track_frame(self, frame: np.ndarray):
        """
        Menerima frame video OpenCV dan mengembalikan koordinat boks deteksi beserta TRACK ID
        """
        # Menggunakan .track() menggantikan .predict() untuk mengaktifkan ByteTrack
        results = self.model.track(
            source=frame, 
            conf=self.conf_threshold, 
            persist=True,       # Wajib True agar ID pekerja tetap terkunci lintas frame
            tracker="bytetrack.yaml", # Menggunakan konfigurasi algoritma ByteTrack
            verbose=False
        )[0]
        
        detections = []
        
        # Periksa apakah ada objek yang terdeteksi pada frame ini
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Ambil Track ID yang dibuat oleh ByteTrack (jika tidak ada, default ke 0)
                track_id = int(box.id[0].cpu().numpy()) if box.id is not None else 0
                
                detections.append({
                    "track_id": track_id, # ID unik pekerja dari ByteTrack
                    "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": self.class_names.get(cls_id, "unknown")
                })
                
        return detections