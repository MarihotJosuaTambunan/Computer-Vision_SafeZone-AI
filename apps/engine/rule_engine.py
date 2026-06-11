class K3RuleEngine:
    @staticmethod
    def is_inside(inner_box, outer_box) -> bool:
        """
        Memeriksa apakah boks APD (inner) berada di dalam atau beririsan kuat 
        dengan boks manusia (outer).
        """
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box
        
        # Hitung area irisan spasial sederhana
        rx1 = max(ix1, ox1)
        ry1 = max(iy1, oy1)
        rx2 = min(ix2, ox2)
        ry2 = min(iy2, oy2)
        
        width = max(0, rx2 - rx1)
        height = max(0, ry2 - ry1)
        intersection_area = width * height
        
        inner_area = (ix2 - ix1) * (iy2 - iy1)
        
        if inner_area == 0:
            return False
            
        # Jika lebih dari 60% boks APD masuk ke area tubuh manusia, dianggap berpasangan
        return (intersection_area / inner_area) > 0.60

    def evaluate_compliance(self, detections: list) -> list:
        """
        Memetakan kepatuhan APD untuk setiap objek 'person' yang terdeteksi
        """
        persons = [d for d in detections if d['class_name'] == 'person']
        apds = [d for d in detections if d['class_name'] in ['helmet', 'vest', 'no-helmet', 'no-vest']]
        
        compliance_results = []
        
        for idx, person in enumerate(persons):
            p_box = person['bbox']
            
            # Status default sebelum diperiksa
            has_helmet = False
            has_vest = False
            violates_helmet = False
            violates_vest = False
            
            # Cari APD yang menempel pada tubuh person ini
            for apd in apds:
                if self.is_inside(apd['bbox'], p_box):
                    if apd['class_name'] == 'helmet':
                        has_helmet = True
                    elif apd['class_name'] == 'no-helmet':
                        violates_helmet = True
                    elif apd['class_name'] == 'vest':
                        has_vest = True
                    elif apd['class_name'] == 'no-vest':
                        violates_vest = True
            
            # Logika konklusi K3
            helmet_status = "PATUH" if has_helmet else ("PELANGGARAN" if violates_helmet else "TIDAK TERDETEKSI")
            vest_status = "PATUH" if has_vest else ("PELANGGARAN" if violates_vest else "TIDAK TERDETEKSI")
            
            compliance_results.append({
                "worker_index": idx,
                "person_bbox": p_box,
                "confidence": person['confidence'],
                "k3_analysis": {
                    "helmet": helmet_status,
                    "vest": vest_status,
                    "safe_status": "AMAN" if (helmet_status == "PATUH" and vest_status == "PATUH") else "BAHAYA"
                }
            })
            
        return compliance_results