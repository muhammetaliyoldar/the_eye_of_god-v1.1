qimport cv2
import numpy as np
import os
import torch
import logging
import time
from datetime import datetime
from collections import defaultdict, deque
import supervision as sv
import subprocess
import sys

# Kendi modüllerimizi içe aktar
from utils.ocr import read_license_plate
from utils.speed_estimation import SpeedEstimator
from utils.violation_check import ViolationDetector
from utils.homography_utils import HomographyTransformer
from utils.database import TrafficDatabase
from utils.color_detection import TrafficLightColorDetector
from utils.mock_models import MockDetection

# Sabit değerleri tanımla
SOURCE_VIDEO_PATH = r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\video-data\test-video.mp4"
TARGET_VIDEO_PATH = r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\outputs\output.mp4"
VEHICLE_MODEL_PATH = r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\models\yolov8x.pt"
TRAFFIC_LIGHT_MODEL_PATH = r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\models\yolov8y.pt"

# --- KALİBRASYON NOKTALARI (VİDEOYA GÖRE DÜZENLEMEN GEREK!) ---
SOURCE = np.array([(941, 228), (1284, 219), (1905, 647), (451, 718)])
TARGET_WIDTH = 20
TARGET_HEIGHT = 30

# Trafik ışığı ihlal çizgisi
CROSS_LINE = [[856, 247], [1179, 245]]

# Logları yapılandır
os.makedirs("outputs/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("outputs", "logs", "application.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("the_eye_of_god")

def is_gpu_available():
    """
    CUDA GPU'nun kullanılabilir olup olmadığını kontrol eder
    
    Returns:
        bool: GPU kullanılabilirse True, değilse False
    """
    return torch.cuda.is_available()

def get_class_name(class_id):
    """
    COCO dataset sınıf ID'sine göre sınıf adını döndürür
    
    Args:
        class_id (int): YOLOv8 COCO sınıf ID'si
        
    Returns:
        str: Sınıf adı
    """
    # COCO sınıf adları (sadece ilgili sınıflar)
    coco_classes = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
    }
    
    return coco_classes.get(class_id, "unknown")

def verify_dependencies():
    """
    Gerekli bağımlılıkları kontrol eder ve eksikleri yükler
    """
    logger.info("Bağımlılıklar kontrol ediliyor...")
    
    try:
        import torch
        import ultralytics
        import supervision
        import pytesseract
        import cv2
        import numpy
        logger.info(f"PyTorch sürümü: {torch.__version__}")
        logger.info(f"Ultralytics sürümü: {ultralytics.__version__}")
        logger.info(f"OpenCV sürümü: {cv2.__version__}")
        
        # YOLOv8 modellerinin varlığını kontrol et
        if not os.path.exists(VEHICLE_MODEL_PATH):
            logger.error(f"Araç modeli bulunamadı: {VEHICLE_MODEL_PATH}")
            return False
            
        if not os.path.exists(TRAFFIC_LIGHT_MODEL_PATH):
            logger.error(f"Trafik ışığı modeli bulunamadı: {TRAFFIC_LIGHT_MODEL_PATH}")
            return False
            
        # Tesseract OCR kontrolü
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR kurulu.")
        except Exception as e:
            logger.warning(f"Tesseract OCR hatası: {e}")
            logger.warning("Plaka tanıma devre dışı bırakılabilir.")
        
        return True
        
    except ImportError as e:
        logger.error(f"Bağımlılık hatası: {e}")
        logger.info("Eksik bağımlılıkları yüklemeye çalışılıyor...")
        
        # Eksik paketleri yüklemeye çalış
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return False

def main():
    # Başlangıç zamanını kaydet
    start_time = time.time()
    
    logger.info("The Eye of God başlatılıyor...")
    
    # Bağımlılıkları kontrol et
    if not verify_dependencies():
        logger.error("Bağımlılık kontrolü başarısız. Program durduruluyor.")
        return
    
    # GPU kullanılabilirliğini kontrol et
    if is_gpu_available():
        logger.info("CUDA GPU bulundu! GPU desteği ile çalışılacak.")
        device = "cuda"
    else:
        logger.info("GPU bulunamadı. CPU ile çalışılacak.")
        device = "cpu"
    
    # Gerekli klasörlerin varlığını kontrol et
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/ihlal_kayitlari", exist_ok=True)
    os.makedirs("database", exist_ok=True)
    
    # Video bilgisini al
    try:
    video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
        logger.info(f"Video yüklendi: {SOURCE_VIDEO_PATH}")
        logger.info(f"Video boyutu: {video_info.resolution_wh}, FPS: {video_info.fps}")
    except Exception as e:
        logger.error(f"Video yüklenemedi: {e}")
        return
    
    # Modelleri yükle
    try:
        logger.info("YOLOv8 modelleri yükleniyor...")
        
        # Modelleri yükleme işlemini sadeleştirme
        try:
            # İlk olarak ultralytics kütüphanesini doğrudan kullanmayı dene
            from ultralytics import YOLO
            
            vehicle_model = YOLO(VEHICLE_MODEL_PATH)
            traffic_light_model = YOLO(TRAFFIC_LIGHT_MODEL_PATH)
            
            logger.info("YOLOv8 modelleri Ultralytics ile başarıyla yüklendi.")
            USE_MOCK_MODELS = False
            
        except Exception as e1:
            logger.warning(f"Ultralytics ile model yüklenemedi: {e1}")
            
            try:
                # Alternatif yöntem dene
                from ultralytics.models import YOLO as UltralyticsYOLO
                
                vehicle_model = UltralyticsYOLO(VEHICLE_MODEL_PATH)
                traffic_light_model = UltralyticsYOLO(TRAFFIC_LIGHT_MODEL_PATH)
                
                logger.info("YOLOv8 modelleri alternatif yöntemle başarıyla yüklendi.")
                USE_MOCK_MODELS = False
                
            except Exception as e2:
                logger.warning(f"Alternatif yöntemle de model yüklenemedi: {e2}")
                
                try:
                    # Torch modeli olarak yüklemeyi dene
                    vehicle_model = torch.hub.load('ultralytics/yolov8', 'custom', path=VEHICLE_MODEL_PATH)
                    traffic_light_model = torch.hub.load('ultralytics/yolov8', 'custom', path=TRAFFIC_LIGHT_MODEL_PATH)
                    
                    logger.info("YOLOv8 modelleri torch.hub ile başarıyla yüklendi.")
                    USE_MOCK_MODELS = False
                    
                except Exception as e3:
                    logger.warning(f"Torch.hub ile de model yüklenemedi: {e3}")
                    logger.info("Test için mock modeller kullanılıyor!")
                    
                    # Mock modelleri kullan
                    vehicle_model = MockDetection(vehicle_classes=True)
                    traffic_light_model = MockDetection(vehicle_classes=False)
                    
                    USE_MOCK_MODELS = True
                    logger.info("Mock modeller başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        return
    
    # ByteTrack (araç takibi için)
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps
    )
    
    # Annotation parametreleri
    # Manual calculation for thickness and text_scale since these utilities aren't available in supervision 0.16.0
    resolution_width, resolution_height = video_info.resolution_wh
    thickness = max(1, int(min(resolution_width, resolution_height) / 500))
    text_scale = max(0.5, min(resolution_width, resolution_height) / 1000)
    
    # Annotation sınıfları
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_padding=3,
        text_color=(255, 255, 255)
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2
    )

    # Video kareleri oluşturucu
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)
    
    # Poligon bölgesi (tespitleri filtrelemek için)
    polygon_zone = sv.PolygonZone(
        polygon=SOURCE,
        frame_resolution_wh=video_info.resolution_wh
    )
    
    # Yardımcı sınıflar
    speed_estimator = SpeedEstimator(
        source_points=SOURCE,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        fps=video_info.fps
    )
    homography = HomographyTransformer(
        source_points=SOURCE,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT
    )
    violation_detector = ViolationDetector()
    color_detector = TrafficLightColorDetector()
    db = TrafficDatabase()
    
    # Araç koordinatları sözlüğü
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps * 3))  # 3 saniyelik veri sakla
    
    # Araç verileri sözlüğü (plaka, tür vb.)
    vehicle_data = {}
    
    # Trafik ışığı rengi (başlangıçta bilinmiyor)
    current_light_color = "unknown"
    
    # İşlem sayacı
    frame_count = 0
    
    logger.info("Video işleme başlıyor...")
    
    # Video çıktısı için video yazıcı
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame in frame_generator:
            frame_count += 1
            
            # FPS hesapla
            fps_start_time = time.time()
            
            # Kare üzerinde çalışmak için kopya oluştur
            annotated_frame = frame.copy()
            
            # --- ARAÇ TESPİTİ ---
            try:
                # Araç tespiti için modeli çalıştır
                if USE_MOCK_MODELS:
                    # Mock model kullanımı
                    vehicle_results = vehicle_model(frame)
                    vehicle_detections = sv.Detections.empty()
                    
                    if len(vehicle_results) > 0:
                        # Mock modellerden gelen tespitleri Detections nesnesine dönüştür
                        boxes = vehicle_results[0].xyxy
                        confidences = vehicle_results[0].confidence
                        class_ids = vehicle_results[0].class_id
                        
                        if len(boxes) > 0:
                            vehicle_detections = sv.Detections(
                                xyxy=boxes,
                                confidence=confidences,
                                class_id=class_ids
                            )
                else:
                    # Gerçek model - birden fazla yükleme metodunu ele al
                    try:
                        # Standart ultralytics.YOLO
                        vehicle_results = vehicle_model(frame)
                        vehicle_detections = sv.Detections.from_ultralytics(vehicle_results[0])
                    except Exception as e:
                        # torch.hub veya diğer metotlar
                        try:
                            # torch.hub yöntemi için farklı API
                            vehicle_results = vehicle_model(frame)
                            
                            if hasattr(vehicle_results, 'xyxy'):
                                # Eğer sonuçta xyxy attribute'u varsa
                                vehicle_detections = sv.Detections(
                                    xyxy=vehicle_results.xyxy,
                                    confidence=vehicle_results.conf,
                                    class_id=vehicle_results.cls
                                )
                            else:
                                # Pandas DF çıktısı
                                vehicle_df = vehicle_results.pandas().xyxy[0]
                                
                                vehicle_detections = sv.Detections(
                                    xyxy=vehicle_df[['xmin', 'ymin', 'xmax', 'ymax']].values,
                                    confidence=vehicle_df['confidence'].values,
                                    class_id=vehicle_df['class'].values
                                )
                        except Exception as e2:
                            logger.error(f"Araç modeli sonuç dönüştürme hatası: {e2}")
                            vehicle_detections = sv.Detections.empty()
                
                # Sadece yüksek güvenilirlikli tespitler
                vehicle_detections = vehicle_detections[vehicle_detections.confidence > 0.3]
                
                # Sadece poligon bölgesi içindeki tespitler
                vehicle_detections = vehicle_detections[polygon_zone.trigger(vehicle_detections)]
                
                # NMS ile çakışan kutuları filtrele
                vehicle_detections = vehicle_detections.with_nms(threshold=0.7)
                
                # ByteTrack ile araç takibi
                vehicle_detections = byte_track.update_with_detections(detections=vehicle_detections)
            except Exception as e:
                logger.error(f"Araç tespiti hatası (kare {frame_count}): {e}")
                vehicle_detections = sv.Detections.empty()
                continue
            
            # --- TRAFİK IŞIĞI TESPİTİ ---
            try:
                # Trafik ışığı tespiti için modeli çalıştır
                if USE_MOCK_MODELS:
                    # Mock model kullanımı
                    traffic_light_results = traffic_light_model(frame)
                    traffic_light_detections = sv.Detections.empty()
                    
                    if len(traffic_light_results) > 0:
                        # Mock modellerden gelen tespitleri Detections nesnesine dönüştür
                        boxes = traffic_light_results[0].xyxy
                        confidences = traffic_light_results[0].confidence
                        class_ids = traffic_light_results[0].class_id
                        
                        if len(boxes) > 0:
                            traffic_light_detections = sv.Detections(
                                xyxy=boxes,
                                confidence=confidences,
                                class_id=class_ids
                            )
                else:
                    # Gerçek model - birden fazla yükleme metodunu ele al
                    try:
                        # Standart ultralytics.YOLO
                        traffic_light_results = traffic_light_model(frame)
                        traffic_light_detections = sv.Detections.from_ultralytics(traffic_light_results[0])
                    except Exception as e:
                        # torch.hub veya diğer metotlar
                        try:
                            # torch.hub yöntemi için farklı API
                            traffic_light_results = traffic_light_model(frame)
                            
                            if hasattr(traffic_light_results, 'xyxy'):
                                # Eğer sonuçta xyxy attribute'u varsa
                                traffic_light_detections = sv.Detections(
                                    xyxy=traffic_light_results.xyxy,
                                    confidence=traffic_light_results.conf,
                                    class_id=traffic_light_results.cls
                                )
                            else:
                                # Pandas DF çıktısı
                                traffic_light_df = traffic_light_results.pandas().xyxy[0]
                                
                                traffic_light_detections = sv.Detections(
                                    xyxy=traffic_light_df[['xmin', 'ymin', 'xmax', 'ymax']].values,
                                    confidence=traffic_light_df['confidence'].values,
                                    class_id=traffic_light_df['class'].values
                                )
                        except Exception as e2:
                            logger.error(f"Trafik ışığı modeli sonuç dönüştürme hatası: {e2}")
                            traffic_light_detections = sv.Detections.empty()
                
                # Sadece yüksek güvenilirlikli tespitler
                traffic_light_detections = traffic_light_detections[traffic_light_detections.confidence > 0.5]
                
                # Trafik ışıklarını ve plaka bölgelerini filtrele (class 9: trafik ışığı)
                plate_detections = traffic_light_detections[traffic_light_detections.class_id == 1]  # Plaka sınıfı
                traffic_light_detections = traffic_light_detections[traffic_light_detections.class_id == 9]  # Trafik ışığı sınıfı
            except Exception as e:
                logger.error(f"Trafik ışığı tespiti hatası (kare {frame_count}): {e}")
                traffic_light_detections = sv.Detections.empty()
                plate_detections = sv.Detections.empty()
            
            # --- TRAFİK IŞIĞI RENGİ ANALİZİ ---
            if len(traffic_light_detections) > 0:
                try:
                    # İlk trafik ışığını analiz et
                    traffic_light_box = traffic_light_detections.xyxy[0].astype(int)
                    detected_color, _ = color_detector.analyze_traffic_light(frame, traffic_light_box)
                    
                    if detected_color != "unknown":
                        current_light_color = detected_color
                        
                    # Kırmızı+sarı kombinasyonu kontrolü (bu durumda kırmızı olarak kabul edilir)
                    if len(traffic_light_detections) > 1:
                        traffic_light_boxes = traffic_light_detections.xyxy.astype(int)
                        if color_detector.is_red_yellow_combination(frame, traffic_light_boxes):
                            current_light_color = "red"  # Kırmızı+sarı durumu kırmızı kabul edilir
                except Exception as e:
                    logger.error(f"Trafik ışığı renk analizi hatası (kare {frame_count}): {e}")
            
            # --- ARAÇ İŞLEME ---
            # Araç merkezlerini al (altı orta nokta)
            try:
                # In supervision 0.16.0, we need to extract coordinates from xyxy
                if len(vehicle_detections) > 0:
                    # Calculate bottom center points from bounding boxes
                    xyxy = vehicle_detections.xyxy
                    vehicle_points = np.zeros((len(xyxy), 2))
                    for i, box in enumerate(xyxy):
                        x1, y1, x2, y2 = box
                        # Bottom center point
                        vehicle_points[i] = [(x1 + x2) / 2, y2]
                    
                    # Matris dönüşümü için noktaları dönüştür
                    transformed_points = speed_estimator.transform_points(points=vehicle_points).astype(int)
            except Exception as e:
                logger.error(f"Koordinat dönüşüm hatası (kare {frame_count}): {e}")
                continue
            
            # Araç verilerini ve hızlarını takip et
            labels = []
            for i, (tracker_id, class_id, confidence, vehicle_box) in enumerate(zip(
                vehicle_detections.tracker_id, 
                vehicle_detections.class_id,
                vehicle_detections.confidence,
                vehicle_detections.xyxy
            )):
                try:
                    if tracker_id is None:
                        continue
                    
                    # Araç sınıfını belirle
                    vehicle_class = get_class_name(int(class_id))
                    
                    # Sadece belirli araç türlerini kabul et
                    if vehicle_class not in ["car", "truck", "bus", "motorcycle"]:
                        continue
                    
                    # Araç koordinatlarını kaydet
                    if i < len(transformed_points):
                        coordinates[tracker_id].append(transformed_points[i])
                    
                    # Araç veritabanına ekle/güncelle
                    if tracker_id not in vehicle_data:
                        # Yeni araç tespiti
                        vehicle_data[tracker_id] = {
                            'tracker_id': tracker_id,
                            'vehicle_class': vehicle_class,
                            'plate_text': None,
                            'city': None,
                            'plate_confidence': 0,
                            'speed': 0,
                            'db_id': None
                        }
                        
                        # Veritabanına kaydet
                        db_id = db.add_vehicle(
                            tracker_id=tracker_id,
                            vehicle_class=vehicle_class
                        )
                        vehicle_data[tracker_id]['db_id'] = db_id
                    
                    # --- PLAKA OKUMA ---
                    # Aracın plakası henüz okunmadıysa, plaka tespitini dene
                    if vehicle_data[tracker_id]['plate_text'] is None and len(plate_detections) > 0:
                        vehicle_box_int = vehicle_box.astype(int)
                        
                        # Plaka bölgelerini kontrol et
                        for plate_box in plate_detections.xyxy:
                            plate_box_int = plate_box.astype(int)
                            
                            # Plaka bölgesinin araç içinde olup olmadığını kontrol et
                            if (plate_box_int[0] >= vehicle_box_int[0] and 
                                plate_box_int[1] >= vehicle_box_int[1] and
                                plate_box_int[2] <= vehicle_box_int[2] and
                                plate_box_int[3] <= vehicle_box_int[3]):
                                
                                # Plaka bölgesini kırp
                                plate_region = frame[
                                    plate_box_int[1]:plate_box_int[3], 
                                    plate_box_int[0]:plate_box_int[2]
                                ]
                                
                                # OCR ile plaka oku
                                if plate_region.size > 0:
                                    try:
                                        plate_result = read_license_plate(plate_region)
                                        
                                        if plate_result['plate_text'] and plate_result['confidence'] > 50:
                                            # Plaka bilgilerini kaydet
                                            vehicle_data[tracker_id]['plate_text'] = plate_result['plate_text']
                                            vehicle_data[tracker_id]['city'] = plate_result['city']
                                            vehicle_data[tracker_id]['plate_confidence'] = plate_result['confidence']
                                            
                                            # Veritabanını güncelle
                                            db.add_vehicle(
                                                tracker_id=tracker_id,
                                                vehicle_class=vehicle_class,
                                                plate_text=plate_result['plate_text'],
                                                city=plate_result['city'],
                                                confidence=plate_result['confidence']
                                            )
                                            
                                            logger.info(f"Yeni plaka tanındı: {plate_result['plate_text']} - {plate_result['city']}")
                                            break
                                    except Exception as e:
                                        logger.error(f"Plaka okuma hatası (kare {frame_count}, araç {tracker_id}): {e}")
                    
                    # --- HIZ HESAPLAMA ---
                    if len(coordinates[tracker_id]) >= video_info.fps / 2:  # En az yarım saniye takip edildiyse
                        # Hız hesapla
                        speed = speed_estimator.calculate_speed(
                            list(coordinates[tracker_id]), 
                            len(coordinates[tracker_id])
                        )
                        
                        # Araç verilerini güncelle
                        vehicle_data[tracker_id]['speed'] = speed
                        
                        # Plaka bilgisi ve araç türü
                        plate_info = ""
                        if vehicle_data[tracker_id]['plate_text']:
                            plate_info = f" {vehicle_data[tracker_id]['plate_text']}"
                        
                        # Etiket metnini oluştur
                        label = f"#{tracker_id}{plate_info} {vehicle_class} {int(speed)} km/h"
                        
                        # --- HIZ İHLALİ KONTROLÜ ---
                        if vehicle_data[tracker_id]['db_id'] is not None:
                            violation, speed_limit = violation_detector.check_speed_violation(
                                vehicle_id=tracker_id,
                                speed=speed,
                                vehicle_class=vehicle_class,
                                plate_text=vehicle_data[tracker_id]['plate_text']
                            )
                            
                            if violation:
                                # Hız ihlali tespit edildi
                                logger.info(f"Hız ihlali: #{tracker_id} - {speed:.1f} km/h (Limit: {speed_limit} km/h)")
                                
                                # İhlal kanıtını kaydet
                                img_path, txt_path = violation_detector.save_violation_evidence(
                                    frame=frame,
                                    violation_type=violation_detector.VIOLATION_TYPES["SPEEDING"],
                                    vehicle_id=tracker_id,
                                    vehicle_data={
                                        'plate_text': vehicle_data[tracker_id]['plate_text'],
                                        'city': vehicle_data[tracker_id]['city'],
                                        'vehicle_class': vehicle_class,
                                        'speed': speed,
                                        'speed_limit': speed_limit
                                    }
                                )
                                
                                # Veritabanına ihlali kaydet
                                db.add_violation(
                                    vehicle_id=vehicle_data[tracker_id]['db_id'],
                                    violation_type=violation_detector.VIOLATION_TYPES["SPEEDING"],
                                    speed=speed,
                                    speed_limit=speed_limit,
                                    evidence_image=img_path,
                                    evidence_text=txt_path
                                )
                        
                        # --- TRAFİK IŞIĞI İHLALİ KONTROLÜ ---
                        vehicle_box_int = vehicle_box.astype(int)
                        if current_light_color != "unknown" and vehicle_data[tracker_id]['db_id'] is not None:
                            violation = violation_detector.check_traffic_light_violation(
                                vehicle_id=tracker_id,
                                vehicle_box=vehicle_box_int,
                                cross_line=CROSS_LINE,
                                light_color=current_light_color,
                                plate_text=vehicle_data[tracker_id]['plate_text']
                            )
                            
                            if violation:
                                # Trafik ışığı ihlali tespit edildi
                                violation_type = violation_detector.VIOLATION_TYPES["RED_LIGHT"] if current_light_color == "red" else violation_detector.VIOLATION_TYPES["YELLOW_LIGHT"]
                                logger.info(f"Trafik ışığı ihlali: #{tracker_id} - {current_light_color.upper()} ışıkta geçiş")
                                
                                # İhlal kanıtını kaydet
                                img_path, txt_path = violation_detector.save_violation_evidence(
                                    frame=frame,
                                    violation_type=violation_type,
                                    vehicle_id=tracker_id,
                                    vehicle_data={
                                        'plate_text': vehicle_data[tracker_id]['plate_text'],
                                        'city': vehicle_data[tracker_id]['city'],
                                        'vehicle_class': vehicle_class,
                                        'speed': speed,
                                        'light_color': current_light_color
                                    }
                                )
                                
                                # Veritabanına ihlali kaydet
                                db.add_violation(
                                    vehicle_id=vehicle_data[tracker_id]['db_id'],
                                    violation_type=violation_type,
                                    speed=speed,
                                    light_color=current_light_color,
                                    evidence_image=img_path,
                                    evidence_text=txt_path
                                )
                    else:
                        label = f"#{tracker_id} {vehicle_class}"
                    
                    labels.append(label)
                except Exception as e:
                    logger.error(f"Araç işleme hatası (kare {frame_count}, araç {tracker_id}): {e}")
                    continue
            
            # --- GÖRSEL ÇIKTI ---
            try:
                # Trafik ışığı rengini göster
                if current_light_color != "unknown":
                    color_map = {
                        "red": (0, 0, 255),      # Kırmızı
                        "yellow": (0, 255, 255),  # Sarı
                        "green": (0, 255, 0)      # Yeşil
                    }
                    
                    # Ekranın sağ üst köşesine trafik ışığı durumunu yaz
                    cv2.putText(
                        annotated_frame,
                        f"TRAFFIC LIGHT: {current_light_color.upper()}",
                        (annotated_frame.shape[1] - 300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color_map[current_light_color],
                        2
                    )
                    
                    # İhlal çizgisini çiz
                    cv2.line(
                        annotated_frame,
                        tuple(CROSS_LINE[0]),
                        tuple(CROSS_LINE[1]),
                        (0, 0, 255) if current_light_color != "green" else (0, 255, 0),
                        2
                    )
                
                # Matris bölgesini çiz
                annotated_frame = homography.draw_region_on_image(annotated_frame)
                
                # Araçları ve etiketleri çiz
                annotated_frame = trace_annotator.annotate(annotated_frame, vehicle_detections)
                annotated_frame = box_annotator.annotate(annotated_frame, vehicle_detections)
                annotated_frame = label_annotator.annotate(annotated_frame, vehicle_detections, labels)
                
                # FPS hesapla ve göster
                fps_end_time = time.time()
                elapsed_time = fps_end_time - fps_start_time
                
                # Division by zero kontrol et
                if elapsed_time > 0:
                    fps = 1 / elapsed_time
                else:
                    fps = 0  # Sıfıra bölme hatası olmaması için
                
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Kare sayısını göster
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Kareyi video çıktısına yaz
            sink.write_frame(annotated_frame)
                
                # Kareyi ekranda göster
                cv2.imshow("The Eye of God - Trafik İhlali Tespit Sistemi", annotated_frame)
            except Exception as e:
                logger.error(f"Görsel çıktı hatası (kare {frame_count}): {e}")
            
            # 'q' tuşu ile çık
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Her 100 karede bir ilerleme bilgisi
            if frame_count % 100 == 0:
                logger.info(f"{frame_count} kare işlendi. İşlem devam ediyor...")
    
    # İşlem bittiğinde
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Video işleme tamamlandı. Toplam {frame_count} kare {total_time:.2f} saniyede işlendi.")
    
    # Division by zero kontrol et
    if total_time > 0:
        avg_fps = frame_count / total_time
    else:
        avg_fps = 0
    
    logger.info(f"Ortalama FPS: {avg_fps:.2f}")
    
    # Veritabanı bağlantısını kapat
    db.close()
    
    # Açık pencereleri kapat
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
