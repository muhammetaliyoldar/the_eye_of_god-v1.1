import cv2
import numpy as np

class TrafficLightColorDetector:
    """
    Trafik ışığı rengini tespit eden sınıf.
    """
    def __init__(self):
        # HSV renk aralıkları
        # Kırmızı (iki aralık: 0-10 ve 160-180)
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Sarı
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([40, 255, 255])
        
        # Yeşil
        self.lower_green = np.array([45, 100, 100])
        self.upper_green = np.array([90, 255, 255])
    
    def detect_color(self, image):
        """
        Görüntüdeki baskın rengi tespit eder
        
        Args:
            image (np.array): Trafik ışığı görüntüsü
            
        Returns:
            str: Tespit edilen renk ('red', 'yellow', 'green' veya 'unknown')
        """
        # BGR'den HSV'ye dönüştür
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Her renk için maske oluştur
        mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        mask_green = cv2.inRange(hsv, self.lower_green, self.upper_green)
        
        # Her renk için piksel sayısını hesapla
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)
        
        # Renk sayılarını topla
        total_color_pixels = red_pixels + yellow_pixels + green_pixels
        
        # Eşik değeri (görüntünün en az %5'i renkli olmalı)
        min_threshold = 0.05 * (image.shape[0] * image.shape[1])
        
        if total_color_pixels < min_threshold:
            return "unknown"
        
        # En çok piksel sayısına sahip rengi seç
        color_counts = {
            "red": red_pixels,
            "yellow": yellow_pixels,
            "green": green_pixels
        }
        
        dominant_color = max(color_counts, key=color_counts.get)
        
        # Baskın renk toplam renkli piksellerin en az %30'unu oluşturmalı
        if color_counts[dominant_color] < 0.3 * total_color_pixels:
            return "unknown"
        
        return dominant_color
    
    def analyze_traffic_light(self, image, traffic_light_box):
        """
        Trafik ışığı bölgesindeki rengi analiz eder
        
        Args:
            image (np.array): Orijinal görüntü
            traffic_light_box (list): Trafik ışığı bounding box'ı [x1, y1, x2, y2]
            
        Returns:
            tuple: (tespit edilen renk (str), işaretlenmiş görüntü (np.array))
        """
        x1, y1, x2, y2 = traffic_light_box
        
        # Görüntü sınırlarını kontrol et
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Trafik ışığı bölgesini kırp
        light_region = image[y1:y2, x1:x2]
        
        # Bölge boş ise (sınırlar hatalı ise)
        if light_region.size == 0:
            return "unknown", image
        
        # Renk tespiti yap
        color = self.detect_color(light_region)
        
        # Sonuç görüntüsünü hazırla
        result_image = image.copy()
        
        # Renk kodları
        color_codes = {
            "red": (0, 0, 255),     # Kırmızı
            "yellow": (0, 255, 255),  # Sarı
            "green": (0, 255, 0),    # Yeşil
            "unknown": (255, 255, 255)  # Beyaz
        }
        
        # Trafik ışığı bounding box'ını çiz
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color_codes[color], 2)
        
        # Tespit edilen rengi yaz
        cv2.putText(
            result_image, 
            color.upper(), 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color_codes[color], 
            2
        )
        
        return color, result_image
    
    def is_red_yellow_combination(self, image, traffic_light_boxes):
        """
        Trafik ışıklarında kırmızı+sarı kombinasyonunu tespit eder
        
        Args:
            image (np.array): Orijinal görüntü
            traffic_light_boxes (list): Trafik ışığı bounding box'ları listesi
            
        Returns:
            bool: Kırmızı ve sarı ışıklar birlikte yanıyorsa True
        """
        has_red = False
        has_yellow = False
        
        for box in traffic_light_boxes:
            color, _ = self.analyze_traffic_light(image, box)
            
            if color == "red":
                has_red = True
            elif color == "yellow":
                has_yellow = True
        
        # Hem kırmızı hem sarı ışık varsa (geçişte)
        return has_red and has_yellow 