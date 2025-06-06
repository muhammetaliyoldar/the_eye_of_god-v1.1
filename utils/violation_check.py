import cv2
import numpy as np
import os
import time
from datetime import datetime

class ViolationDetector:
    """
    Trafik ihlallerini tespit eden ve kaydeden sınıf.
    """
    def __init__(self, output_dir="outputs/ihlal_kayitlari"):
        """
        Args:
            output_dir (str): İhlal kayıtlarının kaydedileceği dizin
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # İhlal tipi kodları
        self.VIOLATION_TYPES = {
            "RED_LIGHT": 1,  # Kırmızı ışık ihlali
            "YELLOW_LIGHT": 2,  # Sarı ışık ihlali
            "SPEEDING": 3,  # Hız ihlali
        }
        
        # Tespit edilen ihlaller (tekrarlı tespitleri önlemek için)
        # {vehicle_id: {violation_type: timestamp}}
        self.detected_violations = {}
        
        # İhlal sayacı
        self.violation_counter = 0
    
    def check_traffic_light_violation(self, vehicle_id, vehicle_box, cross_line, light_color, plate_text=None):
        """
        Trafik ışığı ihlalini kontrol eder
        
        Args:
            vehicle_id (int): Araç ID'si
            vehicle_box (list): Araç bounding box'ı [x1, y1, x2, y2]
            cross_line (list): Geçiş çizgisi [[x1, y1], [x2, y2]]
            light_color (str): Trafik ışığı rengi ('red', 'yellow', 'green')
            plate_text (str, optional): Araç plakası
            
        Returns:
            bool: İhlal tespit edilirse True, edilmezse False
        """
        # Aracın alt orta noktası
        vehicle_bottom = [
            (vehicle_box[0] + vehicle_box[2]) // 2,  # x ortası
            vehicle_box[3]  # y en altı
        ]
        
        # Aracın çizgiyi geçip geçmediğini kontrol et
        if self._is_point_crossing_line(vehicle_bottom, cross_line):
            # Işık yeşil değilse ihlal vardır
            if light_color.lower() != 'green':
                violation_type = self.VIOLATION_TYPES["RED_LIGHT"] if light_color.lower() == 'red' else self.VIOLATION_TYPES["YELLOW_LIGHT"]
                
                # Bu araç için bu ihlal daha önce tespit edilmişse tekrar işlem yapma
                if vehicle_id in self.detected_violations and violation_type in self.detected_violations[vehicle_id]:
                    return False
                
                # Yeni ihlal tespit edildi
                if vehicle_id not in self.detected_violations:
                    self.detected_violations[vehicle_id] = {}
                
                # İhlali kaydet
                self.detected_violations[vehicle_id][violation_type] = time.time()
                return True
        
        return False
    
    def check_speed_violation(self, vehicle_id, speed, vehicle_class, plate_text=None):
        """
        Hız ihlalini kontrol eder
        
        Args:
            vehicle_id (int): Araç ID'si
            speed (float): Araç hızı (km/saat)
            vehicle_class (str): Araç türü
            plate_text (str, optional): Araç plakası
            
        Returns:
            tuple: (ihlal_var_mi (bool), hız_limiti (int))
        """
        # Araç türüne göre hız limiti
        if vehicle_class.lower() == 'truck' or vehicle_class == '7':  # Kamyon
            speed_limit = 40
        else:
            speed_limit = 50
        
        # Hız limiti aşıldı mı?
        if speed > speed_limit:
            violation_type = self.VIOLATION_TYPES["SPEEDING"]
            
            # Bu araç için bu ihlal daha önce tespit edilmişse tekrar işlem yapma
            if vehicle_id in self.detected_violations and violation_type in self.detected_violations[vehicle_id]:
                return False, speed_limit
            
            # Yeni ihlal tespit edildi
            if vehicle_id not in self.detected_violations:
                self.detected_violations[vehicle_id] = {}
            
            # İhlali kaydet
            self.detected_violations[vehicle_id][violation_type] = time.time()
            return True, speed_limit
        
        return False, speed_limit
    
    def save_violation_evidence(self, frame, violation_type, vehicle_id, vehicle_data):
        """
        İhlal kanıtını kaydeder
        
        Args:
            frame (np.array): İhlal anındaki video karesi
            violation_type (int): İhlal türü kodu
            vehicle_id (int): Araç ID'si
            vehicle_data (dict): Araç bilgileri (plaka, hız, tür vb.)
            
        Returns:
            tuple: (kanıt_resim_dosyası, kanıt_metin_dosyası)
        """
        # İhlal anı için zaman damgası
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.violation_counter += 1
        
        # Plaka klasörü oluştur
        plate_text = vehicle_data.get('plate_text', f"UNKNOWN_{vehicle_id}")
        plate_dir = os.path.join(self.output_dir, plate_text)
        os.makedirs(plate_dir, exist_ok=True)
        
        # Dosya adları
        img_filename = f"ihlal_{violation_type}_{timestamp}.jpg"
        txt_filename = f"ihlal_{violation_type}_{timestamp}.txt"
        
        img_path = os.path.join(plate_dir, img_filename)
        txt_path = os.path.join(plate_dir, txt_filename)
        
        # Görüntüyü kaydet
        cv2.imwrite(img_path, frame)
        
        # İhlal detaylarını metin dosyasına kaydet
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"İhlal ID: {self.violation_counter}\n")
            f.write(f"Araç ID: {vehicle_id}\n")
            f.write(f"Plaka: {plate_text}\n")
            
            # Şehir bilgisi
            if 'city' in vehicle_data and vehicle_data['city']:
                f.write(f"Şehir: {vehicle_data['city']}\n")
            
            # Araç türü
            if 'vehicle_class' in vehicle_data:
                f.write(f"Araç Türü: {vehicle_data['vehicle_class']}\n")
            
            # Hız bilgisi
            if 'speed' in vehicle_data:
                f.write(f"Hız: {vehicle_data['speed']:.1f} km/h\n")
            
            # Hız limiti
            if 'speed_limit' in vehicle_data:
                f.write(f"Hız Limiti: {vehicle_data['speed_limit']} km/h\n")
                
                # Hız aşımı
                if 'speed' in vehicle_data:
                    excess = vehicle_data['speed'] - vehicle_data['speed_limit']
                    if excess > 0:
                        f.write(f"Hız Aşımı: +{excess:.1f} km/h\n")
            
            # Trafik ışığı durumu
            if 'light_color' in vehicle_data:
                f.write(f"Trafik Işığı: {vehicle_data['light_color'].upper()}\n")
            
            # İhlal türü
            violation_name = ""
            if violation_type == self.VIOLATION_TYPES["RED_LIGHT"]:
                violation_name = "KIRMIZI IŞIK İHLALİ"
            elif violation_type == self.VIOLATION_TYPES["YELLOW_LIGHT"]:
                violation_name = "SARI IŞIK İHLALİ"
            elif violation_type == self.VIOLATION_TYPES["SPEEDING"]:
                violation_name = "HIZ İHLALİ"
            
            f.write(f"İhlal Türü: {violation_name}\n")
            f.write(f"İhlal Zamanı: {timestamp.replace('_', ' ')}\n")
            f.write(f"Kanıt Görüntüsü: {img_filename}\n")
        
        return img_path, txt_path
    
    def _is_point_crossing_line(self, point, line):
        """
        Bir noktanın çizgiyi geçip geçmediğini kontrol eder
        
        Args:
            point (list): Kontrol edilecek nokta [x, y]
            line (list): Çizgi [[x1, y1], [x2, y2]]
            
        Returns:
            bool: Nokta çizgiyi geçiyorsa True, geçmiyorsa False
        """
        x, y = point
        [[x1, y1], [x2, y2]] = line
        
        # Çizgi denklemi: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        # Noktanın çizgiye olan uzaklığı
        distance = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
        
        # Çizgiye yeterince yakınsa (5 piksel tolerans)
        if distance < 5:
            return True
        
        return False 