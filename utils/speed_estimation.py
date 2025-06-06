import numpy as np
import cv2

class SpeedEstimator:
    """
    Perspektif dönüşüm ve zaman bilgisi kullanarak araç hızlarını tahmin eden sınıf.
    """
    def __init__(self, source_points, target_width, target_height, fps, meter_per_pixel=0.1):
        """
        Args:
            source_points (np.array): Video üzerindeki 4 kalibrasyon noktası [üst sol, üst sağ, alt sağ, alt sol]
            target_width (int): Çıktı perspektif genişliği (metre cinsinden)
            target_height (int): Çıktı perspektif yüksekliği (metre cinsinden)
            fps (float): Video kare hızı
            meter_per_pixel (float): Bir pikselin metre karşılığı
        """
        self.source_points = source_points.astype(np.float32)
        
        # Hedef perspektif noktaları
        self.target_points = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)
        
        # Perspektif dönüşüm matrisi hesapla
        self.transform_matrix = cv2.getPerspectiveTransform(self.source_points, self.target_points)
        
        self.fps = fps
        self.meter_per_pixel = meter_per_pixel
        self.target_width = target_width
        self.target_height = target_height
    
    def transform_point(self, point):
        """
        Bir noktayı perspektif dönüşüm ile çevirir
        
        Args:
            point (np.array): Dönüştürülecek nokta [x, y]
            
        Returns:
            np.array: Dönüştürülmüş nokta [x, y]
        """
        point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(point, self.transform_matrix)
        return transformed.reshape(-1, 2)[0]
    
    def transform_points(self, points):
        """
        Çoklu noktaları perspektif dönüşüm ile çevirir
        
        Args:
            points (np.array): Dönüştürülecek noktalar, şekli (n, 2)
            
        Returns:
            np.array: Dönüştürülmüş noktalar, şekli (n, 2)
        """
        if len(points) == 0:
            return points
        
        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(points, self.transform_matrix)
        return transformed.reshape(-1, 2)
    
    def calculate_speed(self, points, frame_count):
        """
        Bir aracın belirlenen karelerde alınan pozisyonlarına göre hızını hesaplar
        
        Args:
            points (list): Araç pozisyonları, her biri [x, y] şeklinde
            frame_count (int): Pozisyonlar arasındaki toplam kare sayısı
            
        Returns:
            float: Hesaplanan hız (km/saat)
        """
        if len(points) < 2:
            return 0
        
        # Noktaları dönüştür
        transformed_points = self.transform_points(np.array(points))
        
        # İlk ve son nokta arasındaki mesafeyi hesapla
        distance_pixels = np.linalg.norm(transformed_points[-1] - transformed_points[0])
        
        # Mesafeyi metreye çevir
        distance_meters = distance_pixels * self.meter_per_pixel
        
        # Zaman hesapla (saniye)
        time_seconds = frame_count / self.fps
        
        # Hızı hesapla (m/s)
        speed_ms = distance_meters / time_seconds if time_seconds > 0 else 0
        
        # m/s'den km/saat'e çevir
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh
    
    def get_speed_limit(self, vehicle_class):
        """
        Araç türüne göre hız limitini döndürür
        
        Args:
            vehicle_class (str): Araç türü
            
        Returns:
            int: Hız limiti (km/saat)
        """
        # Şehir içi hız limitleri (km/saat)
        limits = {
            'car': 50,       # Otomobil
            'truck': 40,     # Kamyon
            'motorcycle': 50,  # Motosiklet
            'bus': 50,       # Otobüs
            'van': 50,       # Minibüs/Kamyonet
            
            # YOLOv8 sınıf adları için alternatif eşleştirmeler
            '2': 50,         # Otomobil (YOLOv8)
            '7': 40,         # Kamyon (YOLOv8)
            '3': 50,         # Motosiklet (YOLOv8)
            '5': 50,         # Otobüs (YOLOv8)
            '0': 50,         # İnsan (Dikkate alınmayacak)
        }
        
        return limits.get(vehicle_class.lower(), 50)  # Varsayılan: 50 km/saat

    def is_speeding(self, speed, vehicle_class):
        """
        Bir aracın hız limitini aşıp aşmadığını kontrol eder
        
        Args:
            speed (float): Aracın hızı (km/saat)
            vehicle_class (str): Araç türü
            
        Returns:
            bool: Hız ihlali varsa True, yoksa False
        """
        speed_limit = self.get_speed_limit(vehicle_class)
        return speed > speed_limit 