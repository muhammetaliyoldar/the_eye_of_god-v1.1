import numpy as np
import cv2
import supervision as sv

class MockDetection:
    """
    YOLO modellerini simüle eden test sınıfı.
    Gerçek modeller çalışmadığında tüm kodun çalışabilirliğini test etmek için kullanılır.
    """
    def __init__(self, vehicle_classes=True):
        """
        Args:
            vehicle_classes (bool): Araç sınıflarını tanıyan model mi yoksa trafik ışığı/plaka tanıyan model mi
        """
        self.vehicle_classes = vehicle_classes
        
    def __call__(self, frame):
        """
        Bir kare üzerinde tespit yapar ve sonuçları döndürür
        
        Args:
            frame (np.array): İşlenecek video karesi
            
        Returns:
            list: İşlenmiş sonuçları içeren liste
        """
        height, width = frame.shape[:2]
        
        # Örnek sonuç oluştur
        result = MockResult(frame, self.vehicle_classes)
        
        return [result]

class MockResult:
    """
    YOLO model sonuçlarını simüle eden sınıf
    """
    def __init__(self, frame, vehicle_classes=True):
        """
        Args:
            frame (np.array): Video karesi
            vehicle_classes (bool): Araç sınıflarını mı döndürsün
        """
        self.height, self.width = frame.shape[:2]
        self.vehicle_classes = vehicle_classes
        
        # Test verisi oluştur
        self.boxes = self._generate_boxes()
        self.conf = self._generate_confidence()  # conf olarak değiştirildi
        self.cls = self._generate_classes()
        
        # Supervision ile uyumlu format
        self.names = {
            0: "person", 
            1: "license_plate", 
            2: "car", 
            3: "motorcycle", 
            5: "bus", 
            7: "truck",
            9: "traffic_light"
        }
        
        # Doğrudan supervision.Detection sınıfına dönüştür
        self.xyxy = self.boxes
        self.confidence = self.conf
        self.class_id = self.cls
        
    def _generate_boxes(self):
        """
        Rastgele bounding box'lar oluşturur
        """
        # 1-5 arası rastgele sayıda kutu
        num_boxes = np.random.randint(1, 6)
        
        boxes = []
        for _ in range(num_boxes):
            # Rastgele bir konum ve boyut
            x1 = np.random.randint(0, self.width - 100)
            y1 = np.random.randint(0, self.height - 100)
            w = np.random.randint(50, 200)
            h = np.random.randint(50, 200)
            x2 = min(x1 + w, self.width)
            y2 = min(y1 + h, self.height)
            
            boxes.append([x1, y1, x2, y2])
        
        return np.array(boxes)
    
    def _generate_confidence(self):
        """
        Güven değerleri oluşturur
        """
        return np.random.uniform(0.5, 0.95, len(self.boxes))
    
    def _generate_classes(self):
        """
        Sınıf ID'leri oluşturur
        """
        if self.vehicle_classes:
            # Araç modeli: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
            classes = np.random.choice([0, 2, 3, 5, 7], len(self.boxes))
        else:
            # Trafik ışığı/plaka modeli: 1=plate, 9=traffic light
            classes = np.random.choice([1, 9], len(self.boxes))
        
        return classes
        
class MockByteTrack:
    """
    ByteTrack izleme sistemini simüle eden sınıf
    """
    def __init__(self, frame_rate=30, track_activation_threshold=0.3):
        self.next_id = 1
        self.tracked_objects = {}
    
    def update_with_detections(self, detections):
        """
        Tespitlere ID atar
        
        Args:
            detections: Tespit edilen nesneler
            
        Returns:
            Detections: Güncellenmiş tespitler
        """
        # Her bir tespite ID ata
        tracker_ids = []
        
        for i in range(len(detections.xyxy)):
            # Mevcut bir ID varsa onu kullan, yoksa yeni oluştur
            tracker_id = self.next_id
            self.next_id += 1
            tracker_ids.append(tracker_id)
        
        # Detections nesnesine ID'leri ekle
        detections.tracker_id = np.array(tracker_ids)
        
        return detections 