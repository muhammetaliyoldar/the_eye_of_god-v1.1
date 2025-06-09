#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The Eye of God - AI-based traffic violation detection system
Author: ze'O
License: MIT
Version: 1.1
"""

from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import time
import sqlite3
from datetime import datetime
import pytesseract
import re
import os
import sys
import torch
import traceback
from PIL import Image
from itertools import product
import functools

# Check for required dependencies
print("Checking required dependencies...")
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError:
    print("WARNING: OpenCV (cv2) not installed.")

try:
    import supervision as sv
    print(f"Supervision version: {sv.__version__}")
except ImportError:
    print("WARNING: Supervision not installed.")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
except ImportError:
    print("WARNING: PyTorch not installed.")

try:
    import pytesseract
    print(f"Pytesseract installed. Path: {pytesseract.pytesseract.tesseract_cmd}")
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        print(f"WARNING: Tesseract executable not found at {pytesseract.pytesseract.tesseract_cmd}")
except ImportError:
    print("WARNING: Pytesseract not installed.")

# --- CONFIGURATION ---
# Set pytesseract path - adjust this based on your installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- VIDEO LOCATIONS ---
TARGET_VIDEO_PATH = r"C:\Users\muham\PycharmProjects\the_eye_of_god-v1.1\show\app-output\app-output.mp4"

# --- CALIBRATION POINTS (UPDATED WITH NEW COORDINATES) ---
SOURCE = np.array([(912, 215), (1321, 209), (1918, 608), (424, 610), (653, 430), (1642, 427)])
TARGET_WIDTH = 11
TARGET_HEIGHT = 25
TARGET = np.array(
    [[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]]
)

# Define the two zones - lower zone (identification) and upper zone (speed & type)
LOWER_ZONE = np.array([SOURCE[3], SOURCE[2], SOURCE[5], SOURCE[4]])  # Bottom to middle
UPPER_ZONE = np.array([SOURCE[4], SOURCE[5], SOURCE[1], SOURCE[0]])  # Middle to top

# Speed limit in km/h (city streets)
SPEED_LIMIT = 50

# --- CLASS IDS ---
# Class ID mapping based on YOLOv8y
LICENSE_PLATE_CLASS_ID = 1  # License plate class ID 
PERSON_CLASS_ID = 0  # Person class ID
TRAFFIC_LIGHT_CLASS_ID = 9  # Common traffic light class ID
GREEN_LIGHT_CLASS_ID = 11   # Green light class ID - based on detection output
RED_LIGHT_CLASS_ID = 12     # Red light class ID - assumed
YELLOW_LIGHT_CLASS_ID = 13  # Yellow light class ID - assumed

# Traffic light related class IDs to check
TRAFFIC_LIGHT_CLASS_IDS = [9, 10, 11, 12, 13, 14, 15, 16]

# Debug class ID values - will print all detected class IDs
DEBUG_CLASS_IDS = True

# Define Turkish license plate pattern
# Format: 00 AA 000 or 00 AAA 000
TR_LICENSE_PLATE_PATTERN = r'^\d{2}\s*[A-Z]{1,3}\s*\d{2,4}$'

# Configuration Constants
DEFAULT_VIDEO_SOURCE = "video-data/test-video.mp4"
VEHICLE_MODEL_PATH = "models/yolov8x.pt"
LICENSE_PLATE_MODEL_PATH = "models/yolov8y.pt"
CONFIDENCE_THRESHOLD = 0.3
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class TrafficLightColorDetector:
    def __init__(self):
        # Color thresholds for HSV color space
        self.color_ranges = {
            "red": [
                ((0, 70, 50), (10, 255, 255)),
                ((170, 70, 50), (180, 255, 255))
            ],
            "yellow": [((20, 100, 100), (30, 255, 255))],
            "green": [((40, 50, 50), (80, 255, 255))]
        }
    
    def detect_color(self, image, bbox):
        # Extract traffic light region
        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return "unknown"
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for each color
        for color, ranges in self.color_ranges.items():
            mask = None
            for lower, upper in ranges:
                if mask is None:
                    mask = cv2.inRange(hsv, lower, upper)
                else:
                    mask = mask | cv2.inRange(hsv, lower, upper)
            
            if mask is None:
                continue
                
            # If enough pixels match the color, return it
            pixel_count = cv2.countNonZero(mask)
            total_pixels = roi.shape[0] * roi.shape[1]
            if pixel_count > total_pixels * 0.05:  # At least 5% of pixels
                return color
        
        return "unknown"


def extract_and_read_plate(frame, xyxy):
    """Extract license plate from frame using the given bounding box and read it using OCR"""
    try:
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Add padding to plate region (can improve OCR)
        padding = 5
        h, w = frame.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Extract plate region
        plate_img = frame[y1:y2, x1:x2]
        
        if plate_img.size == 0:
            return None, None
        
        # Preprocess image for better OCR
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        # Apply threshold to get black text on white background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Alternative preprocessing: adaptive threshold
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Check if Tesseract is installed
        if not pytesseract.pytesseract.tesseract_cmd or not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            print("WARNING: Tesseract is not installed or path is incorrect. Using mock plate data.")
            # Generate mock plate as fallback
            return plate_img, f"34ABC{str(int(x1))[:2]}"
        
        # Run OCR with Turkish language
        try:
            config = '--oem 3 --psm 6 -l tur'
            text = pytesseract.image_to_string(thresh, config=config)
            
            # Clean the text
            text = text.strip().replace('\n', ' ').replace('\r', '')
            
            # Format and validate the plate
            plate_text = format_license_plate(text)
            
            return plate_img, plate_text
        except Exception as e:
            print(f"OCR Error: {e}")
            # Generate mock plate as fallback
            return plate_img, f"34ABC{str(int(x1))[:2]}"
            
    except Exception as e:
        print(f"Plate extraction error: {e}")
        return None, None

def format_license_plate(text):
    """Format and validate license plate text"""
    # Remove all non-alphanumeric characters
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Pattern for Turkish plates: 2 digits + 1-3 letters + 2-4 digits
    if len(cleaned) >= 5:
        # Try to extract plate format
        plate_match = re.search(r'(\d{2})([A-Z]{1,3})(\d{2,4})', cleaned)
        if plate_match:
            groups = plate_match.groups()
            # Format as: 00 AA 000
            if len(groups) == 3:
                return f"{groups[0]} {groups[1]} {groups[2]}"
    
    # If can't format properly, return original text
    return text

class DatabaseManager:
    def __init__(self, db_path="traffic_violations.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.setup_database()
        
    def setup_database(self):
        # Check if the vehicles table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vehicles'")
        table_exists = self.cursor.fetchone()
        
        if not table_exists:
            # Create vehicles table with more detailed information
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicles (
                vehicle_id INTEGER PRIMARY KEY,
                license_plate TEXT,
                vehicle_type TEXT,
                confidence REAL,
                first_zone TEXT,
                first_detected TIMESTAMP,
                last_detected TIMESTAMP
            )
            ''')
        else:
            # Check if confidence column exists, add if not
            try:
                self.cursor.execute("SELECT confidence FROM vehicles LIMIT 1")
            except sqlite3.OperationalError:
                self.cursor.execute("ALTER TABLE vehicles ADD COLUMN confidence REAL")
                
            # Check if first_zone column exists, add if not
            try:
                self.cursor.execute("SELECT first_zone FROM vehicles LIMIT 1")
            except sqlite3.OperationalError:
                self.cursor.execute("ALTER TABLE vehicles ADD COLUMN first_zone TEXT")
        
        # Check if violations table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='violations'")
        table_exists = self.cursor.fetchone()
        
        if not table_exists:
            # Create violations table with more detail
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                violation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER,
                license_plate TEXT,
                violation_type TEXT,
                speed REAL,
                timestamp TIMESTAMP,
                processed BOOLEAN DEFAULT 0,
                evidence_path TEXT,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles (vehicle_id)
            )
            ''')
        else:
            # Check if license_plate column exists, add if not
            try:
                self.cursor.execute("SELECT license_plate FROM violations LIMIT 1")
            except sqlite3.OperationalError:
                self.cursor.execute("ALTER TABLE violations ADD COLUMN license_plate TEXT")
                
            # Check if processed column exists, add if not
            try:
                self.cursor.execute("SELECT processed FROM violations LIMIT 1")
            except sqlite3.OperationalError:
                self.cursor.execute("ALTER TABLE violations ADD COLUMN processed BOOLEAN DEFAULT 0")
                
            # Check if evidence_path column exists, add if not
            try:
                self.cursor.execute("SELECT evidence_path FROM violations LIMIT 1")
            except sqlite3.OperationalError:
                self.cursor.execute("ALTER TABLE violations ADD COLUMN evidence_path TEXT")
        
        # Check if detection_events table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detection_events'")
        table_exists = self.cursor.fetchone()
        
        if not table_exists:
            # Create detection events table for more detailed tracking
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER,
                license_plate TEXT,
                event_type TEXT,
                zone TEXT,
                speed REAL,
                frame_number INTEGER,
                timestamp TIMESTAMP,
                FOREIGN KEY (vehicle_id) REFERENCES vehicles (vehicle_id)
            )
            ''')
        
        self.conn.commit()
    
    def add_vehicle(self, vehicle_id, license_plate=None, vehicle_type=None, confidence=None, zone=None):
        timestamp = datetime.now()
        
        # Check if vehicle already exists
        self.cursor.execute('SELECT * FROM vehicles WHERE vehicle_id = ?', (vehicle_id,))
        if self.cursor.fetchone() is None:
            # Insert new vehicle
            self.cursor.execute('''
            INSERT INTO vehicles (vehicle_id, license_plate, vehicle_type, confidence, first_zone, first_detected, last_detected)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (vehicle_id, license_plate, vehicle_type, confidence, zone, timestamp, timestamp))
        else:
            # Update existing vehicle
            self.update_vehicle(vehicle_id, license_plate, vehicle_type, confidence)
        
        self.conn.commit()
    
    def update_vehicle(self, vehicle_id, license_plate=None, vehicle_type=None, confidence=None):
        timestamp = datetime.now()
        
        # Build the update SQL dynamically based on which fields are provided
        update_fields = []
        params = []
        
        if license_plate:
            update_fields.append("license_plate = ?")
            params.append(license_plate)
        
        if vehicle_type:
            update_fields.append("vehicle_type = ?")
            params.append(vehicle_type)
        
        if confidence is not None:
            update_fields.append("confidence = ?")
            params.append(confidence)
        
        update_fields.append("last_detected = ?")
        params.append(timestamp)
        
        # Add the vehicle_id to the params
        params.append(vehicle_id)
        
        if update_fields:
            update_sql = f"UPDATE vehicles SET {', '.join(update_fields)} WHERE vehicle_id = ?"
            self.cursor.execute(update_sql, params)
            self.conn.commit()
    
    def add_detection_event(self, vehicle_id, event_type, zone, license_plate=None, speed=None, frame_number=None):
        timestamp = datetime.now()
        self.cursor.execute('''
        INSERT INTO detection_events (vehicle_id, license_plate, event_type, zone, speed, frame_number, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (vehicle_id, license_plate, event_type, zone, speed, frame_number, timestamp))
        self.conn.commit()
    
    def add_violation(self, vehicle_id, violation_type, license_plate=None, speed=None, evidence_path=None):
        timestamp = datetime.now()
        self.cursor.execute('''
        INSERT INTO violations (vehicle_id, license_plate, violation_type, speed, timestamp, evidence_path)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (vehicle_id, license_plate, violation_type, speed, timestamp, evidence_path))
        self.conn.commit()
    
    def get_vehicle_info(self, vehicle_id):
        self.cursor.execute('''
        SELECT license_plate, vehicle_type FROM vehicles WHERE vehicle_id = ?
        ''', (vehicle_id,))
        result = self.cursor.fetchone()
        if result:
            return {"license_plate": result[0], "vehicle_type": result[1]}
        return None
    
    def get_all_violations(self, processed=False):
        self.cursor.execute('''
        SELECT v.violation_id, v.vehicle_id, v.license_plate, v.violation_type, v.speed, v.timestamp, v.evidence_path,
               vh.vehicle_type
        FROM violations v
        LEFT JOIN vehicles vh ON v.vehicle_id = vh.vehicle_id
        WHERE v.processed = ?
        ORDER BY v.timestamp DESC
        ''', (1 if processed else 0,))
        return self.cursor.fetchall()
    
    def mark_violation_processed(self, violation_id):
        self.cursor.execute('''
        UPDATE violations SET processed = 1 WHERE violation_id = ?
        ''', (violation_id,))
        self.conn.commit()
    
    def close(self):
        self.conn.close()


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def determine_traffic_light_state(frame, detections, model_result):
    """Determine traffic light state by majority voting"""
    light_colors = []
    
    # Process all detections
    for i, box in enumerate(detections.xyxy):
        class_id = detections.class_id[i]
        
        # Check if it's a traffic light
        try:
            class_name = model_result.names[int(class_id)].lower()
            if 'light' in class_name:
                x1, y1, x2, y2 = map(int, box)
                
                # Determine color based on class name
                if 'green' in class_name:
                    light_colors.append("green")
                elif 'red' in class_name:
                    light_colors.append("red")
                elif 'yellow' in class_name:
                    light_colors.append("yellow")
                else:
                    # Use color detection for unknown light types
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        
                        # Check if red
                        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                        red_mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
                        red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)
                        
                        # Check if green
                        green_mask = cv2.inRange(hsv, (40, 100, 100), (80, 255, 255))
                        green_pixels = cv2.countNonZero(green_mask)
                        
                        # Check if yellow
                        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
                        yellow_pixels = cv2.countNonZero(yellow_mask)
                        
                        max_pixels = max(red_pixels, green_pixels, yellow_pixels)
                        if max_pixels > 0:
                            if max_pixels == red_pixels:
                                light_colors.append("red")
                            elif max_pixels == green_pixels:
                                light_colors.append("green")
                            elif max_pixels == yellow_pixels:
                                light_colors.append("yellow")
        except Exception as e:
            print(f"Error processing traffic light: {e}")
            continue
    
    # Count occurrences of each color
    color_counts = {"red": 0, "yellow": 0, "green": 0}
    for color in light_colors:
        if color in color_counts:
            color_counts[color] += 1
    
    # Find the majority color
    if not light_colors:
        return "unknown"
    
    max_count = max(color_counts.values())
    if max_count == 0:
        return "unknown"
    
    # Get all colors with the maximum count
    majority_colors = [color for color, count in color_counts.items() if count == max_count]
    
    # If there's a tie, prioritize red for safety
    if len(majority_colors) > 1 and "red" in majority_colors:
        return "red"
    
    return majority_colors[0]


class MockResult:
    """Mock result class to replace YOLOv8 results when model loading fails"""
    def __init__(self):
        self.names = {
            0: 'person', 
            1: 'bicycle', 
            2: 'car', 
            3: 'motorcycle', 
            5: 'bus', 
            7: 'truck', 
            9: 'traffic light',
            10: 'license-plate',
            11: 'green-light',
            12: 'red-light',
            13: 'yellow-light'
        }
        
        class MockBoxes:
            def __init__(self):
                self.xyxy = np.array([])
                self.cls = np.array([])
                self.conf = np.array([])
        
        self.boxes = MockBoxes()


def create_mock_license_detections(frame):
    """Create mock license plate and traffic light detections"""
    h, w = frame.shape[:2]
    
    # Create mock detections with reasonable positions
    mock_xyxy = np.array([
        [w * 0.4, h * 0.6, w * 0.5, h * 0.65],  # License plate 1
        [w * 0.6, h * 0.6, w * 0.7, h * 0.65],  # License plate 2
        [w * 0.1, h * 0.3, w * 0.15, h * 0.35],  # Traffic light 1
    ])
    
    mock_confidence = np.array([0.8, 0.7, 0.9])
    mock_class_id = np.array([10, 10, 11])  # 10: license-plate, 11: green-light
    
    # Create Detections object
    return sv.Detections(
        xyxy=mock_xyxy,
        confidence=mock_confidence,
        class_id=mock_class_id
    )


def create_mock_vehicle_detections(frame):
    """Create mock vehicle detections"""
    h, w = frame.shape[:2]
    
    # Create mock detections with reasonable positions
    mock_xyxy = np.array([
        [w * 0.35, h * 0.5, w * 0.55, h * 0.7],  # Car 1
        [w * 0.55, h * 0.5, w * 0.75, h * 0.7],  # Car 2
        [w * 0.2, h * 0.5, w * 0.4, h * 0.75],   # Truck
        [w * 0.6, h * 0.4, w * 0.8, h * 0.6]     # Bus
    ])
    
    mock_confidence = np.array([0.9, 0.85, 0.8, 0.75])
    mock_class_id = np.array([2, 2, 7, 5])  # 2: car, 7: truck, 5: bus
    
    # Create Detections object
    return sv.Detections(
        xyxy=mock_xyxy,
        confidence=mock_confidence,
        class_id=mock_class_id
    )


def safe_load_yolo(model_path):
    """
    Safely load a YOLO model with PyTorch 2.7.0 compatibility
    """
    try:
        # First try normal loading
        print(f"Attempting to load model from {model_path}...")
        model = YOLO(model_path)
        return model
    except Exception as e:
        if "weights_only" in str(e):
            print("PyTorch 2.7.0 detected. Attempting alternate loading method...")
            try:
                # Try to monkey patch torch.load for this specific loading
                import torch
                import functools
                
                # Store original function
                original_torch_load = torch.load
                
                # Create a patched version that sets weights_only=False
                @functools.wraps(torch.load)
                def patched_torch_load(f, *args, **kwargs):
                    kwargs["weights_only"] = False
                    return original_torch_load(f, *args, **kwargs)
                
                # Apply the monkey patch
                torch.load = patched_torch_load
                
                try:
                    # Try loading with the patched function
                    model = YOLO(model_path)
                    return model
                finally:
                    # Restore original function regardless of outcome
                    torch.load = original_torch_load
            except Exception as nested_e:
                print(f"Alternate loading method failed: {nested_e}")
                raise
        else:
            # For other types of errors, just raise the original exception
            raise


def main():
    try:
        # Load video info
        print("The Eye of God v1.1 başlatılıyor...")
        print("Bağımlılıklar kontrol ediliyor...")
        check_dependencies()

        # Initialize database
        print("Veritabanı başlatılıyor...")
        db_manager = DatabaseManager()
        db_manager.setup_database()

        # Get video source
        source = DEFAULT_VIDEO_SOURCE
        
        print(f"Video kaynağı: {source}")
        print("Video açılıyor ve analiz ediliyor...")
        
        # Video dosyasının varlığını kontrol et
        if not os.path.exists(source):
            print(f"HATA: Video dosyası bulunamadı: {source}")
            print(f"Çalışma dizini: {os.getcwd()}")
            print("Mevcut dosyalar:")
            for root, dirs, files in os.walk(".", topdown=False):
                for file in files:
                    if file.endswith('.mp4') or file.endswith('.MP4'):
                        print(f" - {os.path.join(root, file)}")
            return
        
        # Video bilgisini oku
        video_info = sv.VideoInfo.from_video_path(source)
        frame_width, frame_height = video_info.width, video_info.height
        print(f"Video boyutları: {frame_width}x{frame_height}, FPS: {video_info.fps}")
        
        # Debug: veritabanı ve tracked_vehicles yapısı için
        print("*"*50)
        print("Veritabanı ve takip yapıları başlatılıyor...")
        
        # Hız ve takip için değişkenler
        tracked_vehicles = {}  # Her araç ID'si için bilgiler
        vehicle_types = {}     # Araç ID'si -> araç türü eşleşmesi
        speed_measurements = {}  # Araç ID'si -> pozisyon ölçümleri
        license_plates = {}    # Araç ID'si -> plaka eşleşmesi
        
        print("Uygulamayı sonlandırmak için 'q' tuşuna basın.")
        print("*"*50)
        
        # Model yolları
        license_plate_model_path = LICENSE_PLATE_MODEL_PATH
        vehicle_model_path = VEHICLE_MODEL_PATH
        
        # Load models with error handling
        license_plate_model = None
        vehicle_model = None
        
        try:
            license_plate_model = safe_load_yolo(license_plate_model_path)
            print("License plate model loaded successfully.")
        except Exception as e:
            print(f"Error loading license plate model: {e}")
            print("Will use mock detections for license plates and traffic lights.")
        
        try:
            vehicle_model = safe_load_yolo(vehicle_model_path)
            print("Vehicle model loaded successfully.")
        except Exception as e:
            print(f"Error loading vehicle model: {e}")
            print("Will use mock detections for vehicles.")
        
        # Initialize ByteTrack for tracking
        byte_track = sv.ByteTrack(
            frame_rate=video_info.fps
        )
        
        # Manual calculation for thickness and text_scale
        resolution_width, resolution_height = video_info.resolution_wh
        thickness = max(2, int(min(resolution_width, resolution_height) / 400))
        text_scale = max(0.6, min(resolution_width, resolution_height) / 800)
        
        # Define colors for visualization
        colors = {
            'unknown': (200, 200, 200),  # Light gray for unknown
            'car': (0, 255, 0),          # Green for cars
            'truck': (0, 140, 255),      # Orange for trucks
            'bus': (0, 128, 255),        # Light orange for buses
            'motorcycle': (255, 255, 0), # Cyan for motorcycles
            'tracked': (255, 255, 0),    # Sarı for tracked vehicles
            'red': (0, 0, 255),          # Red for red lights
            'yellow': (0, 255, 255),     # Yellow for yellow lights
            'green': (0, 255, 0),        # Green for green lights
            'license': (255, 0, 255),    # Magenta for license plates
            'lower_zone': (0, 165, 255), # Orange for lower zone
            'upper_zone': (255, 0, 127)  # Purple for upper zone
        }
        
        # Vehicle type mapping - from YOLOv8x classes to our simplified categories
        vehicle_type_mapping = {
            'car': 'car',
            'motorcycle': 'motorcycle',
            'bus': 'bus',
            'truck': 'truck',
            '2': 'car',          # Map class IDs to vehicle types
            '3': 'motorcycle',
            '5': 'bus',
            '7': 'truck',
            'automobile': 'car',
            'bicycle': 'motorcycle'
        }
        
        # Annotators
        box_annotator = sv.BoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_padding=5
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2
        )
        
        # Frame generator
        frame_generator = sv.get_video_frames_generator(source_path=source)
        
        # Initialize the polygon zones for supervision 0.16.0 using minimal arguments
        # The first two arguments are required: polygon points and frame dimensions
        resolution_width, resolution_height = video_info.resolution_wh
        polygon_zone = sv.PolygonZone(
            SOURCE[:4],  # Use the four corner points
            (resolution_width, resolution_height)
        )
        
        lower_zone = sv.PolygonZone(
            LOWER_ZONE,
            (resolution_width, resolution_height)
        )
        
        upper_zone = sv.PolygonZone(
            UPPER_ZONE,
            (resolution_width, resolution_height)
        )
        
        # Initialize view transformer
        view_transformer = ViewTransformer(source=SOURCE[:4], target=TARGET)
        
        # Initialize traffic light color detector
        traffic_light_detector = TrafficLightColorDetector()
        
        # Tracking data
        tracked_vehicles = {}
        vehicles_in_lower_zone = set()  # Track vehicles currently in lower zone
        vehicles_in_upper_zone = set()  # Track vehicles currently in upper zone
        tracked_plates = {}
        license_plates = {}  # Store license plate for each vehicle ID
        vehicle_types = {}
        
        # Speed tracking data
        speed_measurements = defaultdict(list)  # Store multiple speed measurements per vehicle
        
        # Current traffic light state
        current_light_state = "unknown"
        
        # Frame counter
        frame_count = 0
        
        # Directory for saving evidence
        evidence_dir = "evidence"
        os.makedirs(evidence_dir, exist_ok=True)
        
        # Process the video
        with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
            for frame in frame_generator:
                frame_count += 1
                start_time = time.time()
                
                # Process with models or generate mock detections
                if license_plate_model:
                    try:
                        license_result = license_plate_model(frame)[0]
                        license_detections = sv.Detections.from_ultralytics(license_result)
                        license_detections = license_detections[license_detections.confidence > CONFIDENCE_THRESHOLD]
                        license_detections = license_detections[polygon_zone.trigger(license_detections)]
                        license_detections = license_detections.with_nms(threshold=0.7)
                    except Exception as e:
                        print(f"Error running license plate model: {e}")
                        license_result = None
                        license_detections = sv.Detections.empty()
                else:
                    # Create mock license plate detections
                    license_result = MockResult()
                    license_detections = create_mock_license_detections(frame)
                
                if vehicle_model:
                    try:
                        vehicle_result = vehicle_model(frame)[0]
                        vehicle_detections = sv.Detections.from_ultralytics(vehicle_result)
                        vehicle_detections = vehicle_detections[vehicle_detections.confidence > CONFIDENCE_THRESHOLD]
                        vehicle_detections = vehicle_detections[polygon_zone.trigger(vehicle_detections)]
                        vehicle_detections = vehicle_detections.with_nms(threshold=0.7)
                    except Exception as e:
                        print(f"Error running vehicle model: {e}")
                        vehicle_result = None
                        vehicle_detections = sv.Detections.empty()
                else:
                    # Create mock vehicle detections
                    vehicle_result = MockResult()
                    vehicle_detections = create_mock_vehicle_detections(frame)
                
                # Debug - print all detection info
                if frame_count <= 3:
                    print(f"Processing frame {frame_count}")
                    if isinstance(license_result, MockResult):
                        print("Using mock license detections")
                    else:
                        print(f"License model output: {[f'{license_result.names[int(c)]}' for c in license_result.boxes.cls]}")
                    
                    if isinstance(vehicle_result, MockResult):
                        print("Using mock vehicle detections")
                    else:
                        print(f"Vehicle model output: {[f'{vehicle_result.names[int(c)]}' for c in vehicle_result.boxes.cls]}")
                        print(f"Vehicle model classes: {vehicle_result.names}")
                
                # Standard detections processing for traffic lights and license plates
                license_detections = sv.Detections.from_ultralytics(license_result)
                
                # Process vehicle detections separately
                vehicle_type_detections = []
                vehicle_type_boxes = []
                try:
                    if len(vehicle_detections) > 0 and hasattr(vehicle_detections, 'xyxy') and hasattr(vehicle_detections, 'class_id'):
                        for i in range(len(vehicle_detections.xyxy)):
                            class_id = vehicle_detections.class_id[i]
                            if int(class_id) < len(vehicle_result.names):
                                class_name = vehicle_result.names[int(class_id)].lower()
                                # Check if it's a vehicle type we care about
                                if class_name in vehicle_type_mapping:
                                    vehicle_type = vehicle_type_mapping[class_name]
                                    vehicle_type_detections.append((vehicle_detections.xyxy[i], vehicle_detections.confidence[i], vehicle_type, i))
                                    vehicle_type_boxes.append(vehicle_detections.xyxy[i])
                except Exception as e:
                    print(f"Error processing vehicle detections: {e}")
                    # Continue with empty detections
                
                # Filter by confidence
                license_detections = license_detections[license_detections.confidence > CONFIDENCE_THRESHOLD]
                vehicle_detections = vehicle_detections[vehicle_detections.confidence > CONFIDENCE_THRESHOLD]
                
                # Filter by region of interest
                license_detections = license_detections[polygon_zone.trigger(license_detections)]
                vehicle_detections = vehicle_detections[polygon_zone.trigger(vehicle_detections)]
                
                # Apply NMS
                license_detections = license_detections.with_nms(threshold=0.7)
    # Get video source
    source = DEFAULT_VIDEO_SOURCE
    
    print(f"Video kaynağı: {source}")
    print("Video açılıyor ve analiz ediliyor...")
    video_info = sv.VideoInfo.from_video_path(source)
    frame_width, frame_height = video_info.width, video_info.height
    
    # Model paths - change these to match your model locations
    license_plate_model_path = LICENSE_PLATE_MODEL_PATH
    vehicle_model_path = VEHICLE_MODEL_PATH
    
    # Check if model files exist
    if not os.path.exists(license_plate_model_path):
        print(f"WARNING: License plate model not found at {license_plate_model_path}")
        print("You may need to download the model first.")
    
    if not os.path.exists(vehicle_model_path):
        print(f"WARNING: Vehicle model not found at {vehicle_model_path}")
        print("You may need to download the model first.")
    
    # Load models with error handling
    license_plate_model = None
    vehicle_model = None
    
    try:
        license_plate_model = safe_load_yolo(license_plate_model_path)
        print("License plate model loaded successfully.")
    except Exception as e:
        print(f"Error loading license plate model: {e}")
        print("Will use mock detections for license plates and traffic lights.")
    
    try:
        vehicle_model = safe_load_yolo(vehicle_model_path)
        print("Vehicle model loaded successfully.")
    except Exception as e:
        print(f"Error loading vehicle model: {e}")
        print("Will use mock detections for vehicles.")
    
    # Initialize ByteTrack for tracking
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps
    )
    
    # Manual calculation for thickness and text_scale
    resolution_width, resolution_height = video_info.resolution_wh
    thickness = max(2, int(min(resolution_width, resolution_height) / 400))
    text_scale = max(0.6, min(resolution_width, resolution_height) / 800)
    
    # Define colors for visualization
    colors = {
        'unknown': (200, 200, 200),  # Light gray for unknown
        'car': (0, 255, 0),          # Green for cars
        'truck': (0, 140, 255),      # Orange for trucks
        'bus': (0, 128, 255),        # Light orange for buses
        'motorcycle': (255, 255, 0), # Cyan for motorcycles
        'tracked': (255, 255, 0),    # Sarı for tracked vehicles
        'red': (0, 0, 255),          # Red for red lights
        'yellow': (0, 255, 255),     # Yellow for yellow lights
        'green': (0, 255, 0),        # Green for green lights
        'license': (255, 0, 255),    # Magenta for license plates
        'lower_zone': (0, 165, 255), # Orange for lower zone
        'upper_zone': (255, 0, 127)  # Purple for upper zone
    }
    
    # Vehicle type mapping - from YOLOv8x classes to our simplified categories
    vehicle_type_mapping = {
        'car': 'car',
        'motorcycle': 'motorcycle',
        'bus': 'bus',
        'truck': 'truck',
        '2': 'car',          # Map class IDs to vehicle types
        '3': 'motorcycle',
        '5': 'bus',
        '7': 'truck',
        'automobile': 'car',
        'bicycle': 'motorcycle'
    }
    
    # Annotators
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_padding=5
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2
    )
    
    # Frame generator
    frame_generator = sv.get_video_frames_generator(source_path=source)
    
    # Initialize the polygon zones for supervision 0.16.0 using minimal arguments
    # The first two arguments are required: polygon points and frame dimensions
    resolution_width, resolution_height = video_info.resolution_wh
    polygon_zone = sv.PolygonZone(
        SOURCE[:4],  # Use the four corner points
        (resolution_width, resolution_height)
    )
    
    lower_zone = sv.PolygonZone(
        LOWER_ZONE,
        (resolution_width, resolution_height)
    )
    
    upper_zone = sv.PolygonZone(
        UPPER_ZONE,
        (resolution_width, resolution_height)
    )
    
    # Initialize view transformer
    view_transformer = ViewTransformer(source=SOURCE[:4], target=TARGET)
    
    # Initialize traffic light color detector
    traffic_light_detector = TrafficLightColorDetector()
    
    # Tracking data
    tracked_vehicles = {}
    vehicles_in_lower_zone = set()  # Track vehicles currently in lower zone
    vehicles_in_upper_zone = set()  # Track vehicles currently in upper zone
    tracked_plates = {}
    license_plates = {}  # Store license plate for each vehicle ID
    vehicle_types = {}
    
    # Speed tracking data
    speed_measurements = defaultdict(list)  # Store multiple speed measurements per vehicle
    
    # Current traffic light state
    current_light_state = "unknown"
    
    # Frame counter
    frame_count = 0
    
    # Directory for saving evidence
    evidence_dir = "evidence"
    os.makedirs(evidence_dir, exist_ok=True)
    
    # Process the video
    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame in frame_generator:
            frame_count += 1
            start_time = time.time()
            
            # Process with models or generate mock detections
            if license_plate_model:
                try:
                    license_result = license_plate_model(frame)[0]
                    license_detections = sv.Detections.from_ultralytics(license_result)
                    license_detections = license_detections[license_detections.confidence > CONFIDENCE_THRESHOLD]
                    license_detections = license_detections[polygon_zone.trigger(license_detections)]
                    license_detections = license_detections.with_nms(threshold=0.7)
                except Exception as e:
                    print(f"Error running license plate model: {e}")
                    license_result = None
                    license_detections = sv.Detections.empty()
            else:
                # Create mock license plate detections
                license_result = MockResult()
                license_detections = create_mock_license_detections(frame)
            
            if vehicle_model:
                try:
                    vehicle_result = vehicle_model(frame)[0]
                    vehicle_detections = sv.Detections.from_ultralytics(vehicle_result)
                    vehicle_detections = vehicle_detections[vehicle_detections.confidence > CONFIDENCE_THRESHOLD]
                    vehicle_detections = vehicle_detections[polygon_zone.trigger(vehicle_detections)]
                    vehicle_detections = vehicle_detections.with_nms(threshold=0.7)
                except Exception as e:
                    print(f"Error running vehicle model: {e}")
                    vehicle_result = None
                    vehicle_detections = sv.Detections.empty()
            else:
                # Create mock vehicle detections
                vehicle_result = MockResult()
                vehicle_detections = create_mock_vehicle_detections(frame)
            
            # Debug - print all detection info
            if frame_count <= 3:
                print(f"Processing frame {frame_count}")
                if isinstance(license_result, MockResult):
                    print("Using mock license detections")
                else:
                    print(f"License model output: {[f'{license_result.names[int(c)]}' for c in license_result.boxes.cls]}")
                
                if isinstance(vehicle_result, MockResult):
                    print("Using mock vehicle detections")
                else:
                    print(f"Vehicle model output: {[f'{vehicle_result.names[int(c)]}' for c in vehicle_result.boxes.cls]}")
                    print(f"Vehicle model classes: {vehicle_result.names}")
            
            # Standard detections processing for traffic lights and license plates
            license_detections = sv.Detections.from_ultralytics(license_result)
            
            # Process vehicle detections separately
            vehicle_type_detections = []
            vehicle_type_boxes = []
            try:
                if len(vehicle_detections) > 0 and hasattr(vehicle_detections, 'xyxy') and hasattr(vehicle_detections, 'class_id'):
                    for i in range(len(vehicle_detections.xyxy)):
                        class_id = vehicle_detections.class_id[i]
                        if int(class_id) < len(vehicle_result.names):
                            class_name = vehicle_result.names[int(class_id)].lower()
                            # Check if it's a vehicle type we care about
                            if class_name in vehicle_type_mapping:
                                vehicle_type = vehicle_type_mapping[class_name]
                                vehicle_type_detections.append((vehicle_detections.xyxy[i], vehicle_detections.confidence[i], vehicle_type, i))
                                vehicle_type_boxes.append(vehicle_detections.xyxy[i])
            except Exception as e:
                print(f"Error processing vehicle detections: {e}")
                # Continue with empty detections
            
            # Filter by confidence
            license_detections = license_detections[license_detections.confidence > CONFIDENCE_THRESHOLD]
            vehicle_detections = vehicle_detections[vehicle_detections.confidence > CONFIDENCE_THRESHOLD]
            
            # Filter by region of interest
            license_detections = license_detections[polygon_zone.trigger(license_detections)]
            vehicle_detections = vehicle_detections[polygon_zone.trigger(vehicle_detections)]
            
            # Apply NMS
            license_detections = license_detections.with_nms(threshold=0.7)
            vehicle_detections = vehicle_detections.with_nms(threshold=0.7)
            
            # Create merged detections for tracking
            all_boxes = np.vstack([license_detections.xyxy, vehicle_detections.xyxy]) if len(license_detections) > 0 and len(vehicle_detections) > 0 else \
                      license_detections.xyxy if len(license_detections) > 0 else \
                      vehicle_detections.xyxy if len(vehicle_detections) > 0 else \
                      np.array([]).reshape(0, 4)  # Empty array with correct shape

            all_confidence = np.hstack([license_detections.confidence, vehicle_detections.confidence]) if len(license_detections) > 0 and len(vehicle_detections) > 0 else \
                           license_detections.confidence if len(license_detections) > 0 else \
                           vehicle_detections.confidence if len(vehicle_detections) > 0 else \
                           np.array([])
                           
            # Create class_id list, using the appropriate model's names
            license_class_ids = license_detections.class_id.tolist() if len(license_detections) > 0 else []
            vehicle_class_ids = vehicle_detections.class_id.tolist() if len(vehicle_detections) > 0 else []
            
            all_class_ids = np.hstack([license_class_ids, vehicle_class_ids]) if license_class_ids and vehicle_class_ids else \
                          np.array(license_class_ids) if license_class_ids else \
                          np.array(vehicle_class_ids) if vehicle_class_ids else \
                          np.array([])
            
            # Create a merged Detections object
            if len(all_boxes) > 0:
                merged_detections = sv.Detections(
                    xyxy=all_boxes,
                    confidence=all_confidence,
                    class_id=all_class_ids
                )
            else:
                merged_detections = sv.Detections.empty()
            
            # Update trackers on the merged detections
            if len(merged_detections) > 0:
                tracked_detections = byte_track.update_with_detections(detections=merged_detections)
            else:
                tracked_detections = sv.Detections.empty()
            
            # Determine traffic light state
            if isinstance(license_result, MockResult):
                # Use a mock traffic light state that alternates every 30 frames
                traffic_light_states = ["red", "yellow", "green"]
                current_light_state = traffic_light_states[(frame_count // 30) % 3]
            else:
                # Use real detection
                current_light_state = determine_traffic_light_state(frame, license_detections, license_result)
            
            # Copy frame for annotations
            annotated_frame = frame.copy()
            
            # Draw the time and processing FPS
            processing_fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            current_timestamp = frame_count / video_info.fps

            # Add info panel to top of screen
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame,
                f"The Eye of God v1.1 | FPS: {processing_fps:.1f} | Tespit: {len(tracked_vehicles)} arac | Trafik Isigi: {current_light_state}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Draw the regions
            # Main region
            pts = SOURCE[:4].reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(annotated_frame, [pts], True, (0, 255, 255), 2)
            
            # Lower zone - Vehicle ID and license plate zone
            pts_lower = LOWER_ZONE.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(annotated_frame, [pts_lower], True, colors['lower_zone'], 2)
            
            # Upper zone - Speed and vehicle type zone
            pts_upper = UPPER_ZONE.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(annotated_frame, [pts_upper], True, colors['upper_zone'], 2)
            
            # ================ STAGE 1: PROCESS ALL DETECTIONS ================
            # Extract license plates
            license_plate_detections = []
            try:
                if len(license_detections) > 0 and hasattr(license_detections, 'xyxy') and hasattr(license_detections, 'class_id'):
                    for i in range(len(license_detections.xyxy)):
                        class_id = license_detections.class_id[i]
                        if int(class_id) < len(license_result.names) and license_result.names[int(class_id)] == "license-plate":
                            license_plate_detections.append((license_detections.xyxy[i], license_detections.confidence[i], class_id, i))
            except Exception as e:
                print(f"Error processing license plate detections: {e}")
                # Continue with empty detections

            # Extract vehicles from vehicle model
            vehicle_type_detections = []
            vehicle_type_boxes = []
            try:
                if len(vehicle_detections) > 0 and hasattr(vehicle_detections, 'xyxy') and hasattr(vehicle_detections, 'class_id'):
                    for i in range(len(vehicle_detections.xyxy)):
                        class_id = vehicle_detections.class_id[i]
                        if int(class_id) < len(vehicle_result.names):
                            class_name = vehicle_result.names[int(class_id)].lower()
                            # Check if it's a vehicle type we care about
                            if class_name in vehicle_type_mapping:
                                vehicle_type = vehicle_type_mapping[class_name]
                                vehicle_type_detections.append((vehicle_detections.xyxy[i], vehicle_detections.confidence[i], vehicle_type, i))
                                vehicle_type_boxes.append(vehicle_detections.xyxy[i])
            except Exception as e:
                print(f"Error processing vehicle detections: {e}")
                # Continue with empty detections
            
            # ================ STAGE 2: PROCESS VEHICLES ================
            # Draw all tracked detections first with bounding boxes
            if len(tracked_detections) > 0:
                # Draw boxes for all detections
                annotated_frame = box_annotator.annotate(annotated_frame, tracked_detections)
            
            # Process tracked vehicles
            for i, tracker_id in enumerate(tracked_detections.tracker_id):
                # Get bounding box coordinates
                xyxy = tracked_detections.xyxy[i]
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Get center point
                center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                # Check zones
                in_lower_zone = lower_zone.trigger(tracked_detections)[i]
                in_upper_zone = upper_zone.trigger(tracked_detections)[i]
                
                # Get the right color
                box_color = colors['tracked']
                
                # Draw box with vehicle ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)
                
                # Araç türünü al
                vehicle_type = vehicle_types.get(tracker_id, "unknown")
                
                # ================ STAGE 3: PROCESS LOWER ZONE (ID & LICENSE PLATE) ================
                if in_lower_zone:
                    # Alt bölgede ise araç türü, ID ve plaka göster
                    plate_text = license_plates.get(tracker_id, "")
                    vehicle_label = f"{vehicle_type} #{tracker_id}"
                    if plate_text:
                        vehicle_label += f" - {plate_text}"
                    
                    # Label için arka plan ve yazı
                    label_width = len(vehicle_label) * 10
                    cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + label_width, y1), (255, 255, 0), -1)
                    cv2.putText(
                        annotated_frame,
                        vehicle_label,
                        (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, 
                        (0, 0, 0),
                        2
                    )
                    
                    # Veritabanında araç bilgisini güncelle
                    db_manager.update_vehicle(int(tracker_id), vehicle_type=vehicle_type)
                    
                    # Yakındaki plakaları bul ve işle
                    for plate_xyxy, plate_confidence, plate_class_id, plate_idx in license_plate_detections:
                        plate_x1, plate_y1, plate_x2, plate_y2 = map(int, plate_xyxy)
                        plate_center = ((plate_x1 + plate_x2) // 2, (plate_y1 + plate_y2) // 2)
                        
                        # Plaka için kutu çiz
                        cv2.rectangle(annotated_frame, (plate_x1, plate_y1), (plate_x2, plate_y2), colors['license'], 2)
                        
                        # Araç ve plaka arasındaki mesafeyi hesapla
                        dist = ((center_point[0] - plate_center[0]) ** 2 + 
                                (center_point[1] - plate_center[1]) ** 2) ** 0.5
                        
                        # Eğer yakınsa, plakayı araçla ilişkilendir
                        if dist < 200:  # Eşik değeri ayarlanabilir
                            # OCR ile plaka oku
                            plate_img, plate_text = extract_and_read_plate(frame, plate_xyxy)
                            
                            if plate_text and len(plate_text) > 3:
                                # Plakayı depola
                                license_plates[tracker_id] = plate_text
                                
                                # Veritabanını güncelle
                                db_manager.update_vehicle(int(tracker_id), license_plate=plate_text)
                                
                                # Plaka okuma olayını kaydet
                                db_manager.add_detection_event(
                                    int(tracker_id),
                                    "license_plate_read",
                                    "lower",
                                    license_plate=plate_text,
                                    frame_number=frame_count
                                )
                                
                                # Araçla plaka arasında bağlantı çiz
                                cv2.line(annotated_frame, center_point, plate_center, colors['license'], 2)
                
                # ================ STAGE 4: PROCESS UPPER ZONE (SPEED & VIOLATIONS) ================
                elif in_upper_zone:
                    # Vehicle is in upper zone - focus on speed and type
                    
                    # Record position for speed calculation
                    if tracker_id not in speed_measurements:
                        speed_measurements[tracker_id] = []
                    speed_measurements[tracker_id].append((frame_count, center_point))
                    
                    # Need at least 5 frames of data for speed calculation
                    if len(speed_measurements[tracker_id]) >= 5:
                        # Get oldest and newest measurements
                        oldest_frame, oldest_point = speed_measurements[tracker_id][0]
                        newest_frame, newest_point = speed_measurements[tracker_id][-1]
                        
                        # Calculate time elapsed
                        time_elapsed = (newest_frame - oldest_frame) / video_info.fps
                        
                        if time_elapsed > 0:
                            print(f"Hız hesaplanıyor: Araç #{tracker_id}")
                            # Transform points using homography matrix
                            oldest_transformed = view_transformer.transform_points(np.array([oldest_point]))[0]
                            newest_transformed = view_transformer.transform_points(np.array([newest_point]))[0]
                            
                            # Calculate distance in transformed space (meters)
                            distance_pixels = np.sqrt(
                                (newest_transformed[0] - oldest_transformed[0])**2 + 
                                (newest_transformed[1] - oldest_transformed[1])**2
                            )
                            
                            print(f"Mesafe (piksel): {distance_pixels:.2f}")
                            
                            # Önceden kalibre edilmiş değer (1 piksel = X metre)
                            pixels_to_meters = 0.05  # Daha yüksek kalibrasyon faktörü
                            
                            # Convert to real-world distance
                            distance_meters = distance_pixels * pixels_to_meters
                            
                            print(f"Mesafe (metre): {distance_meters:.2f}, Zaman: {time_elapsed:.2f} sn")
                            
                            # Calculate speed in km/h
                            speed = (distance_meters / time_elapsed) * 3.6
                            
                            print(f"Hız: {int(speed)} km/h")
                            
                            # Store speed for display
                            if tracker_id not in tracked_vehicles:
                                tracked_vehicles[tracker_id] = {}
                            tracked_vehicles[tracker_id]['speed'] = int(speed)
                            
                            # Üst zonda hız bilgisi göster
                            vehicle_label = f"{vehicle_type} #{tracker_id} - {int(speed)} km/h"
                            label_width = len(vehicle_label) * 10
                            cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + label_width, y1), (255, 255, 0), -1)
                            cv2.putText(
                                annotated_frame, 
                                vehicle_label, 
                                (x1 + 5, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7,
                                (0, 0, 0),
                                2
                            )
                        else:
                            # Yeterli veri yoksa sadece araç türü ve ID göster
                            vehicle_label = f"{vehicle_type} #{tracker_id}"
                            label_width = len(vehicle_label) * 10
                            cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + label_width, y1), (255, 255, 0), -1)
                            cv2.putText(
                                annotated_frame, 
                                vehicle_label, 
                                (x1 + 5, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7,
                                (0, 0, 0),
                                2
                            )
            
            # ================ STAGE 5: PROCESS TRAFFIC LIGHTS ================
            for i, box in enumerate(license_result.boxes.xyxy):
                try:
                    class_id = int(license_result.boxes.cls[i].item())
                    class_name = license_result.names[class_id]
                    confidence = float(license_result.boxes.conf[i].item())
                    
                    # Only process traffic lights
                    if 'light' in class_name.lower() and confidence > 0.3:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Determine color based on class name
                        if 'green' in class_name.lower():
                            light_color = "green"
                        elif 'red' in class_name.lower():
                            light_color = "red"
                        elif 'yellow' in class_name.lower():
                            light_color = "yellow"
                        else:
                            light_color = "unknown"
                        
                        # Draw the bounding box with thick lines
                        box_color = colors.get(light_color, colors['unknown'])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), 
                                    box_color, thickness+2)  # Very thick lines
                        
                        # Add a filled colored bar for better visibility
                        bar_height = max(30, int((y2 - y1) * 0.2))
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y1 + bar_height), 
                                    box_color, -1)  # Filled rectangle
                        
                        # Add label text with class name
                        cv2.putText(
                            annotated_frame,
                            f"{class_name} ({confidence:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            box_color,
                            2
                        )
                except Exception as e:
                    # Skip this detection if there's any error
                    if frame_count <= 5:
                        print(f"Error processing traffic light: {e}")
                    continue
            
            # ================ STAGE 7: ADD ANNOTATIONS AND FINALIZE FRAME ================
            # Add trace annotations for all vehicles
            if len(tracked_detections) > 0:
                annotated_frame = trace_annotator.annotate(annotated_frame, tracked_detections)
            
            # Draw current traffic light state
            light_color = colors.get(current_light_state, colors['unknown'])
            cv2.putText(
                annotated_frame,
                f"Trafik Isigi: {current_light_state.upper()}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                light_color,
                3
            )
            
            # Calculate FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            
            # Add top information panel
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame,
                f"The Eye of God v1.1 | FPS: {fps:.1f} | Arac: {len(tracked_vehicles)} | Isik: {current_light_state.upper()}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Add a legend
            legend_height = 230
            legend = np.zeros((legend_height, annotated_frame.shape[1], 3), dtype=np.uint8)
            cv2.rectangle(legend, (0, 0), (legend.shape[1], legend.shape[0]), (30, 30, 30), -1)
            
            # Draw color indicators for traffic lights
            cv2.putText(legend, "Trafik Isigi Renk Kodlari:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(legend, "Kirmizi Isik", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['red'], 2)
            cv2.putText(legend, "Sari Isik", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['yellow'], 2)
            cv2.putText(legend, "Yesil Isik", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['green'], 2)
            cv2.putText(legend, "Plaka", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['license'], 2)
            
            # Draw vehicle type info
            cv2.putText(legend, "Arac Tipleri:", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(legend, "Otomobil", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['car'], 2)
            cv2.putText(legend, "Motorsiklet", (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['motorcycle'], 2)
            cv2.putText(legend, "Otobus", (350, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['bus'], 2)
            cv2.putText(legend, "Kamyon", (500, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['truck'], 2)
            
            # Stats
            stats_x = annotated_frame.shape[1] - 300
            plate_count = len(license_plate_detections)
            
            cv2.putText(legend, f"Plaka Sayisi: {plate_count}", 
                      (stats_x, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['license'], 2)
            
            # Add current state info to legend
            cv2.putText(legend, f"Isik Durumu: {current_light_state.upper()}", 
                      (stats_x, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors.get(current_light_state, colors['unknown']), 2)
            
            cv2.putText(legend, f"Alt Bolge Arac: {len(vehicles_in_lower_zone)}", 
                      (stats_x, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['lower_zone'], 2)
            
            cv2.putText(legend, f"Ust Bolge Arac: {len(vehicles_in_upper_zone)}", 
                      (stats_x, 120), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['upper_zone'], 2)
            
            # Process box info
            for i, tracker_id in enumerate(tracked_detections.tracker_id):
                # Get bounding box coordinates
                xyxy = tracked_detections.xyxy[i]
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Get center point
                center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                # Check zones
                in_lower_zone = lower_zone.trigger(tracked_detections)[i]
                in_upper_zone = upper_zone.trigger(tracked_detections)[i]
                
                # Get the right color
                box_color = colors['tracked']
                
                # Draw box with vehicle ID
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, thickness)
                
                # Araç türünü al
                vehicle_type = vehicle_types.get(tracker_id, "unknown")
                
                # ================ STAGE 3: PROCESS LOWER ZONE (ID & LICENSE PLATE) ================
                if in_lower_zone:
                    # Alt bölgede ise araç türü, ID ve plaka göster
                    plate_text = license_plates.get(tracker_id, "")
                    vehicle_label = f"{vehicle_type} #{tracker_id}"
                    if plate_text:
                        vehicle_label += f" - {plate_text}"
                    
                    # Label için arka plan ve yazı
                    label_width = len(vehicle_label) * 10
                    cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + label_width, y1), (255, 255, 0), -1)
                    cv2.putText(
                        annotated_frame,
                        vehicle_label,
                        (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, 
                        (0, 0, 0),
                        2
                    )
                
                # ================ STAGE 4: PROCESS UPPER ZONE (SPEED & VIOLATIONS) ================
                elif in_upper_zone:
                    # Vehicle is in upper zone - focus on speed and type
                    
                    # Record position for speed calculation
                    if tracker_id not in speed_measurements:
                        speed_measurements[tracker_id] = []
                    speed_measurements[tracker_id].append((frame_count, center_point))
                    
                    # Need at least 5 frames of data for speed calculation
                    if len(speed_measurements[tracker_id]) >= 5:
                        # Get oldest and newest measurements
                        oldest_frame, oldest_point = speed_measurements[tracker_id][0]
                        newest_frame, newest_point = speed_measurements[tracker_id][-1]
                        
                        # Calculate time elapsed
                        time_elapsed = (newest_frame - oldest_frame) / video_info.fps
                        
                        if time_elapsed > 0:
                            # Transform points using homography matrix
                            oldest_transformed = view_transformer.transform_points(np.array([oldest_point]))[0]
                            newest_transformed = view_transformer.transform_points(np.array([newest_point]))[0]
                            
                            # Calculate distance in transformed space (meters)
                            distance_pixels = np.sqrt(
                                (newest_transformed[0] - oldest_transformed[0])**2 + 
                                (newest_transformed[1] - oldest_transformed[1])**2
                            )
                            
                            # Önceden kalibre edilmiş değer (1 piksel = X metre)
                            pixels_to_meters = 0.05  # Daha yüksek kalibrasyon faktörü
                            
                            # Convert to real-world distance
                            distance_meters = distance_pixels * pixels_to_meters
                            
                            # Calculate speed in km/h
                            speed = (distance_meters / time_elapsed) * 3.6
                            
                            # Store speed for display
                            if tracker_id not in tracked_vehicles:
                                tracked_vehicles[tracker_id] = {}
                            tracked_vehicles[tracker_id]['speed'] = int(speed)
                            
                            # Üst zonda hız bilgisi göster
                            vehicle_label = f"{vehicle_type} #{tracker_id} - {int(speed)} km/h"
                            label_width = len(vehicle_label) * 10
                            cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + label_width, y1), (255, 255, 0), -1)
                            cv2.putText(
                                annotated_frame, 
                                vehicle_label, 
                                (x1 + 5, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7,
                                (0, 0, 0),
                                2
                            )
                        else:
                            # Yeterli veri yoksa sadece araç türü ve ID göster
                            vehicle_label = f"{vehicle_type} #{tracker_id}"
                            label_width = len(vehicle_label) * 10
                            cv2.rectangle(annotated_frame, (x1, y1 - 30), (x1 + label_width, y1), (255, 255, 0), -1)
                            cv2.putText(
                                annotated_frame, 
                                vehicle_label, 
                                (x1 + 5, y1 - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7,
                                (0, 0, 0),
                                2
                            )
            
            # Display with legend
            display_frame = np.vstack((annotated_frame, legend))
            
            # Resize for display if too large
            display_height, display_width = display_frame.shape[:2]
            max_display_height = 900  # Maximum reasonable height
            
            if display_height > max_display_height:
                scale_factor = max_display_height / display_height
                display_width = int(display_width * scale_factor)
                display_height = max_display_height
                display_frame = cv2.resize(display_frame, (display_width, display_height))
            
            # Write frame to output video
            sink.write_frame(annotated_frame)
            
            # Display frame
            cv2.imshow("THE EYE OF GOD - Trafik Ihlal Tespit Sistemi", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Close the database connection
        db_manager.close()
        cv2.destroyAllWindows()
        
        # Hızları veritabanına kaydet
        for tracker_id, vehicle_data in tracked_vehicles.items():
            if 'speed' in vehicle_data:
                speed = vehicle_data['speed']
                try:
                    db_manager.add_detection_event(
                        int(tracker_id),
                        "speed_measurement",
                        "upper",
                        speed=speed,
                        frame_number=frame_count
                    )
                except Exception as e:
                    print(f"Hız kaydı sırasında hata: {e}")


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    # Convert boxes to [x1, y1, x2, y2] format
    box1 = [float(x) for x in box1]
    box2 = [float(x) for x in box2]
    
    # Calculate intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


if __name__ == "__main__":
    main()