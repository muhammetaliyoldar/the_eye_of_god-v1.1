# The Eye of God - Trafik İhlali Tespit Sistemi

Yapay zeka destekli, gerçek zamanlı trafik ihlali tespit ve arşivleme sistemi.

## Özellikler

- **Trafik Işığı İhlali Tespiti**: Kırmızı veya sarı ışıkta geçen araçları tespit eder
- **Hız İhlali Tespiti**: Matris dönüşümü ile araç hızlarını ölçer ve limit aşımlarını kaydeder
- **Plaka Tanıma**: Tesseract OCR ile Türk plakalarını yüksek doğrulukta okur
- **Detaylı Arşivleme**: Her ihlal için kanıt görüntüsü ve detaylı bilgi kaydı oluşturur
- **Veritabanı Takibi**: Tüm araç ve ihlal bilgilerinin SQLite veritabanında saklanması
- **Gerçek Zamanlı Görselleştirme**: Akan videoda tespit, izleme ve ölçüm bilgilerinin gösterimi

## Sistem Gereksinimleri

- Python 3.8-3.10 (3.10 önerilen)
- CUDA destekli NVIDIA GPU (önerilen)
- Tesseract OCR (Windows: C:\Program Files\Tesseract-OCR)
- Yeterli RAM (en az 8GB, 16GB önerilen)

## Kurulum

### 1. Ortam Hazırlığı

Sanal ortam oluşturun (opsiyonel ama önerilir):
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 2. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### 3. YOLOv8 Modellerini Kontrol Edin

models/ klasöründe aşağıdaki dosyaların bulunduğundan emin olun:
- yolov8x.pt (araç tespiti için)
- yolov8y.pt (trafik ışığı ve plaka tespiti için)

Eksik modelleri [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) sitesinden indirebilirsiniz.

### 4. Tesseract OCR Kurulumu

- **Windows**: [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki) adresinden indirip kurun
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

### 5. Olası Sorunlar ve Çözümleri

#### YOLOv8 Model Yükleme Sorunları

Program birden fazla yükleme yöntemi deneyerek modelleri otomatik olarak yüklemeye çalışır:
1. `ultralytics.YOLO` ile doğrudan yükleme
2. `ultralytics.models.YOLO` alternatif yolu
3. `torch.hub.load` yöntemi
4. Başarısız olursa mock modelleri kullanma

Modeller hala yüklenemiyorsa:

```bash
# Ultralytics paketini yeniden yükleyin
pip uninstall ultralytics
pip install ultralytics==8.0.225

# Torch ve torchvision'ı yeniden yükleyin (CUDA 11.8 için)
pip uninstall torch torchvision
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

## Sistemi Çalıştırma

```bash
python main.py
```

## Proje Yapısı

```
the_eye_of_god-v1.1/
├── models/                # YOLOv8 modelleri
├── video-data/            # Test ve ana video dosyaları
├── outputs/               # İhlal kayıtları ve log dosyaları
├── database/              # SQLite veritabanı
├── utils/                 # Yardımcı fonksiyonlar
├── main.py                # Ana program
├── README.md              # Bu dosya
├── PRD.md                 # Ürün gereksinim belgesi
├── LICENSE                # MIT Lisansı
└── requirements.txt       # Gerekli paketler
```

## Kalibrasyonlar

Matris koordinatları:
- `[856, 247], [1179, 245], [1917, 820], [288, 841]`

Trafik ışığı ihlal çizgisi:
- `[856, 247]` ile `[1179, 245]` arasındaki sanal çizgi

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın. 