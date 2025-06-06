import sqlite3
import os
import time
from datetime import datetime

class TrafficDatabase:
    """
    Araç ve ihlal kayıtlarını yöneten veritabanı sınıfı.
    """
    def __init__(self, db_path="database/violations.db"):
        """
        Args:
            db_path (str): Veritabanı dosya yolu
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Veritabanı bağlantısını oluştur
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # Tabloları oluştur
        self._create_tables()
    
    def _create_tables(self):
        """
        Gerekli tabloları oluşturur
        """
        # Araçlar tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY,
            tracker_id INTEGER NOT NULL,
            plate_text TEXT,
            city TEXT,
            vehicle_class TEXT,
            confidence REAL,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # İhlaller tablosu
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY,
            vehicle_id INTEGER,
            violation_type INTEGER NOT NULL,
            speed REAL,
            speed_limit INTEGER,
            light_color TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            evidence_image TEXT,
            evidence_text TEXT,
            FOREIGN KEY (vehicle_id) REFERENCES vehicles (id)
        )
        ''')
        
        self.conn.commit()
    
    def add_vehicle(self, tracker_id, vehicle_class=None, plate_text=None, city=None, confidence=None):
        """
        Yeni araç kaydı ekler veya mevcut aracı günceller
        
        Args:
            tracker_id (int): Araç izleme ID'si
            vehicle_class (str, optional): Araç türü
            plate_text (str, optional): Araç plakası
            city (str, optional): Plaka şehri
            confidence (float, optional): Plaka okuma güveni
            
        Returns:
            int: Veritabanı araç ID'si
        """
        # Önce bu tracker_id ile kayıt var mı kontrol et
        self.cursor.execute(
            "SELECT id, plate_text FROM vehicles WHERE tracker_id = ?",
            (tracker_id,)
        )
        result = self.cursor.fetchone()
        
        if result:
            # Araç zaten kayıtlı, bilgilerini güncelle
            vehicle_id, existing_plate = result
            
            # Plaka bilgisi olmayan bir araç ve yeni plaka bilgisi geldiyse güncelle
            if (existing_plate is None or existing_plate == "") and plate_text:
                self.cursor.execute(
                    "UPDATE vehicles SET plate_text = ?, city = ?, confidence = ?, last_seen = CURRENT_TIMESTAMP WHERE id = ?",
                    (plate_text, city, confidence, vehicle_id)
                )
            else:
                # Sadece son görülme zamanını güncelle
                self.cursor.execute(
                    "UPDATE vehicles SET last_seen = CURRENT_TIMESTAMP WHERE id = ?",
                    (vehicle_id,)
                )
            
            self.conn.commit()
            return vehicle_id
        else:
            # Yeni araç kaydı ekle
            self.cursor.execute(
                "INSERT INTO vehicles (tracker_id, vehicle_class, plate_text, city, confidence) VALUES (?, ?, ?, ?, ?)",
                (tracker_id, vehicle_class, plate_text, city, confidence)
            )
            self.conn.commit()
            return self.cursor.lastrowid
    
    def add_violation(self, vehicle_id, violation_type, speed=None, speed_limit=None, 
                     light_color=None, evidence_image=None, evidence_text=None):
        """
        Yeni ihlal kaydı ekler
        
        Args:
            vehicle_id (int): Veritabanındaki araç ID'si
            violation_type (int): İhlal türü kodu
            speed (float, optional): Araç hızı (km/saat)
            speed_limit (int, optional): Hız limiti (km/saat)
            light_color (str, optional): Trafik ışığı rengi
            evidence_image (str, optional): Kanıt görüntüsü dosya yolu
            evidence_text (str, optional): Kanıt metin dosyası yolu
            
        Returns:
            int: Veritabanı ihlal ID'si
        """
        self.cursor.execute(
            """
            INSERT INTO violations 
            (vehicle_id, violation_type, speed, speed_limit, light_color, evidence_image, evidence_text) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (vehicle_id, violation_type, speed, speed_limit, light_color, evidence_image, evidence_text)
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_vehicle_by_tracker_id(self, tracker_id):
        """
        Tracker ID'ye göre araç bilgilerini getirir
        
        Args:
            tracker_id (int): Araç izleme ID'si
            
        Returns:
            dict or None: Araç bilgileri veya bulunamazsa None
        """
        self.cursor.execute(
            """
            SELECT id, tracker_id, plate_text, city, vehicle_class 
            FROM vehicles WHERE tracker_id = ?
            """,
            (tracker_id,)
        )
        result = self.cursor.fetchone()
        
        if result:
            return {
                'id': result[0],
                'tracker_id': result[1],
                'plate_text': result[2],
                'city': result[3],
                'vehicle_class': result[4]
            }
        return None
    
    def get_vehicle_by_id(self, vehicle_id):
        """
        Veritabanı ID'sine göre araç bilgilerini getirir
        
        Args:
            vehicle_id (int): Veritabanı araç ID'si
            
        Returns:
            dict or None: Araç bilgileri veya bulunamazsa None
        """
        self.cursor.execute(
            """
            SELECT id, tracker_id, plate_text, city, vehicle_class 
            FROM vehicles WHERE id = ?
            """,
            (vehicle_id,)
        )
        result = self.cursor.fetchone()
        
        if result:
            return {
                'id': result[0],
                'tracker_id': result[1],
                'plate_text': result[2],
                'city': result[3],
                'vehicle_class': result[4]
            }
        return None
    
    def get_violations_by_vehicle_id(self, vehicle_id):
        """
        Araç ID'sine göre ihlalleri getirir
        
        Args:
            vehicle_id (int): Veritabanı araç ID'si
            
        Returns:
            list: İhlal kayıtları listesi
        """
        self.cursor.execute(
            """
            SELECT id, violation_type, speed, speed_limit, light_color, timestamp, evidence_image, evidence_text
            FROM violations WHERE vehicle_id = ?
            ORDER BY timestamp DESC
            """,
            (vehicle_id,)
        )
        violations = []
        for row in self.cursor.fetchall():
            violations.append({
                'id': row[0],
                'violation_type': row[1],
                'speed': row[2],
                'speed_limit': row[3],
                'light_color': row[4],
                'timestamp': row[5],
                'evidence_image': row[6],
                'evidence_text': row[7]
            })
        return violations
    
    def get_all_violations(self, limit=100):
        """
        Tüm ihlalleri getirir
        
        Args:
            limit (int, optional): Maksimum kayıt sayısı
            
        Returns:
            list: İhlal kayıtları listesi
        """
        self.cursor.execute(
            """
            SELECT v.id, v.violation_type, v.speed, v.speed_limit, v.light_color, v.timestamp, 
                   v.evidence_image, v.evidence_text, vh.plate_text, vh.city, vh.vehicle_class
            FROM violations v
            JOIN vehicles vh ON v.vehicle_id = vh.id
            ORDER BY v.timestamp DESC
            LIMIT ?
            """,
            (limit,)
        )
        violations = []
        for row in self.cursor.fetchall():
            violations.append({
                'id': row[0],
                'violation_type': row[1],
                'speed': row[2],
                'speed_limit': row[3],
                'light_color': row[4],
                'timestamp': row[5],
                'evidence_image': row[6],
                'evidence_text': row[7],
                'plate_text': row[8],
                'city': row[9],
                'vehicle_class': row[10]
            })
        return violations
    
    def close(self):
        """
        Veritabanı bağlantısını kapatır
        """
        if self.conn:
            self.conn.close() 