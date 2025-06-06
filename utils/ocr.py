import cv2
import numpy as np
import pytesseract
import re

# Tesseract OCR yolunu ayarla
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Türkiye plaka bilgileri
PLATE_CITIES = {
    '01': 'Adana', '02': 'Adıyaman', '03': 'Afyonkarahisar', '04': 'Ağrı', '05': 'Amasya',
    '06': 'Ankara', '07': 'Antalya', '08': 'Artvin', '09': 'Aydın', '10': 'Balıkesir',
    '11': 'Bilecik', '12': 'Bingöl', '13': 'Bitlis', '14': 'Bolu', '15': 'Burdur',
    '16': 'Bursa', '17': 'Çanakkale', '18': 'Çankırı', '19': 'Çorum', '20': 'Denizli',
    '21': 'Diyarbakır', '22': 'Edirne', '23': 'Elazığ', '24': 'Erzincan', '25': 'Erzurum',
    '26': 'Eskişehir', '27': 'Gaziantep', '28': 'Giresun', '29': 'Gümüşhane', '30': 'Hakkari',
    '31': 'Hatay', '32': 'Isparta', '33': 'Mersin', '34': 'İstanbul', '35': 'İzmir',
    '36': 'Kars', '37': 'Kastamonu', '38': 'Kayseri', '39': 'Kırklareli', '40': 'Kırşehir',
    '41': 'Kocaeli', '42': 'Konya', '43': 'Kütahya', '44': 'Malatya', '45': 'Manisa',
    '46': 'Kahramanmaraş', '47': 'Mardin', '48': 'Muğla', '49': 'Muş', '50': 'Nevşehir',
    '51': 'Niğde', '52': 'Ordu', '53': 'Rize', '54': 'Sakarya', '55': 'Samsun',
    '56': 'Siirt', '57': 'Sinop', '58': 'Sivas', '59': 'Tekirdağ', '60': 'Tokat',
    '61': 'Trabzon', '62': 'Tunceli', '63': 'Şanlıurfa', '64': 'Uşak', '65': 'Van',
    '66': 'Yozgat', '67': 'Zonguldak', '68': 'Aksaray', '69': 'Bayburt', '70': 'Karaman',
    '71': 'Kırıkkale', '72': 'Batman', '73': 'Şırnak', '74': 'Bartın', '75': 'Ardahan',
    '76': 'Iğdır', '77': 'Yalova', '78': 'Karabük', '79': 'Kilis', '80': 'Osmaniye',
    '81': 'Düzce'
}

def preprocess_plate_image(plate_image):
    """
    Plaka görüntüsünü OCR için ön işleme adımlarından geçirir
    """
    # Gri tonlama
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    
    # Gürültü azaltma
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptif eşikleme
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 19, 9)
    
    # Morfolojik işlemler (gürültü temizleme)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def read_license_plate(plate_image):
    """
    Plaka görüntüsünden plaka metnini okur ve Türk plaka formatına uygunluğunu kontrol eder
    
    Returns:
        dict: {
            'plate_text': str,  # Okunan plaka metni
            'city': str,        # Plaka ait olduğu şehir
            'confidence': float # Okuma güveni (0-100)
        }
    """
    # Plaka görüntüsünü ön işleme
    processed_img = preprocess_plate_image(plate_image)
    
    # Tesseract OCR ile plaka okuma
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ocr_result = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # En yüksek güvenilirliğe sahip sonucu al
    confidences = [int(conf) for conf in ocr_result['conf'] if conf != '-1']
    if not confidences:
        return {'plate_text': None, 'city': None, 'confidence': 0}
    
    max_conf_idx = ocr_result['conf'].index(str(max(confidences)))
    plate_text = ocr_result['text'][max_conf_idx]
    confidence = float(ocr_result['conf'][max_conf_idx])
    
    # Plaka düzeltme ve doğrulama
    plate_text = clean_plate_text(plate_text)
    
    # Şehir bilgisini çıkar
    city = None
    if plate_text and len(plate_text) >= 2:
        city_code = plate_text[:2]
        city = PLATE_CITIES.get(city_code)
    
    return {
        'plate_text': plate_text,
        'city': city,
        'confidence': confidence
    }

def clean_plate_text(text):
    """
    OCR sonucunu temizler ve Türk plaka formatına uygun hale getirir
    Türk plakası formatı: 34ABC123 (2 rakam, 1-3 harf, 2-4 rakam)
    """
    if not text or len(text) < 4:  # Çok kısa sonuçları reddet
        return None
    
    # Boşlukları kaldır ve büyük harfe çevir
    text = text.replace(" ", "").upper()
    
    # Yanlış karakterleri düzelt
    text = text.replace('İ', 'I').replace('Ö', 'O').replace('Ü', 'U')
    text = text.replace('Ğ', 'G').replace('Ş', 'S').replace('Ç', 'C')
    
    # Türk plaka formatı kontrolü: 00XXX00 veya 00X0000
    turkish_plate_pattern = r'^(\d{1,2})([A-Z]{1,3})(\d{2,4})$'
    match = re.match(turkish_plate_pattern, text)
    
    if match:
        return text
    
    # Gelişmiş temizlik ve format düzeltme (OCR hatalarını düzeltme)
    # Sadece rakam ve harfleri tut
    cleaned = re.sub(r'[^A-Z0-9]', '', text)
    
    # İlk iki karakter rakam olmalı
    if len(cleaned) >= 2 and cleaned[:2].isdigit():
        # Sonraki 1-3 karakter harf olmalı
        letters_part = ""
        for i in range(2, min(5, len(cleaned))):
            if cleaned[i].isalpha():
                letters_part += cleaned[i]
            else:
                break
        
        if letters_part:
            # Kalan kısım rakam olmalı
            numbers_part = ""
            for i in range(2 + len(letters_part), len(cleaned)):
                if cleaned[i].isdigit():
                    numbers_part += cleaned[i]
                else:
                    break
            
            if numbers_part and 2 <= len(numbers_part) <= 4:
                return cleaned[:2] + letters_part + numbers_part
    
    return None 