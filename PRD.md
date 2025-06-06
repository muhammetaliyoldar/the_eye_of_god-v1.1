# Ürün Gereksinim Belgesi (PRD) - The Eye of God

## 1. Giriş

"The Eye of God" (Tanrının Gözü), yapay zeka destekli bir trafik ihlali tespit sistemidir. Sistem, trafik kameralarından alınan görüntüleri analiz ederek trafik ışığı ihlallerini ve hız sınırı aşımlarını tespit eder, kaydeder ve arşivler.

## 2. Ürün Amacı

Trafik güvenliğini artırmak ve trafik kurallarına uyumu teşvik etmek için:
- Trafik ışığı ihlallerini (kırmızı veya sarı ışıkta geçiş) tespit etmek
- Araç hızlarını ölçerek hız sınırı aşımlarını belirlemek
- Tespit edilen ihlalleri kanıtlarıyla birlikte arşivlemek
- Trafik denetimlerini daha etkin ve adil hale getirmek

## 3. Hedef Kullanıcılar

- Trafik Denetim Birimleri
- Belediye Trafik Yönetim Merkezleri
- Karayolları Genel Müdürlüğü
- Akıllı Şehir Yönetim Sistemleri

## 4. Sistem Bileşenleri

### 4.1 Girdi Kaynakları

- **Video Kaynağı**: 4K çözünürlükte, 30 FPS hızında trafik kamera görüntüleri
- **AI Modelleri**:
  - YOLOv8x: Araç tespiti ve sınıflandırması
  - YOLOv8y: Trafik ışığı ve plaka bölgesi tespiti
- **OCR Motoru**: Tesseract OCR (Türkçe desteği ile)

### 4.2 Temel Fonksiyonlar

#### Tespit ve Tanıma
- Araç tespiti ve sınıflandırması (otomobil, kamyon, motosiklet, vb.)
- Trafik ışığı tespiti ve renk analizi
- Plaka bölgesi tespiti ve OCR ile plaka okuması
- Araç takibi ve benzersiz ID atama

#### İhlal Analizi
- Trafik ışığı ihlali (kırmızı/sarı ışıkta geçiş) tespiti
- Matris dönüşümü ile araç hızı hesaplama
- Araç türüne göre hız limiti kontrolü
- İhlal durumunda kanıt oluşturma

#### Veri Yönetimi
- İhlal kayıtlarının yapılandırılmış dosyalama sistemi
- SQLite veritabanında araç ve ihlal bilgilerinin saklanması
- Türkiye plaka sistemine uygun il eşleştirmesi

#### Görselleştirme
- Gerçek zamanlı tespit ve izleme görselleştirmesi
- Araç bilgilerinin ve hız değerlerinin ekranda gösterimi
- Trafik ışığı durumunun renk paletiyle gösterimi

## 5. Teknik Gereksinimler

### 5.1 Donanım
- CUDA destekli NVIDIA GPU (Minimum 6GB VRAM)
- 16GB+ RAM
- 500GB+ depolama alanı (video arşivi için)

### 5.2 Yazılım
- Python 3.10
- YOLOv8 modelleri
- Tesseract OCR (Türkçe dil paketi ile)
- CUDA ve cuDNN (GPU kullanımı için)
- OpenCV, NumPy, PyTorch

### 5.3 Performans Gereksinimleri
- 4K video akışında gerçek zamanlı işleme (minimum 25 FPS)
- %95+ araç tespit doğruluğu
- %90+ plaka okuma doğruluğu
- %98+ trafik ışığı renk analizi doğruluğu

## 6. İhlal Tespit Kriterleri

### 6.1 Trafik Işığı İhlali
- Işık durumu yeşil olmadığında (kırmızı, sarı veya kırmızı+sarı) sanal çizgiyi geçen araçlar
- Sanal çizgi: [856, 247] ile [1179, 245] koordinatları arası
- Işık rengi HSV analizi ile doğrulanacak

### 6.2 Hız İhlali
- Şehir içi hız limitleri:
  - Otomobil: 50 km/h
  - Motosiklet: 50 km/h
  - Kamyonet: 50 km/h
  - Kamyon: 40 km/h
  - Otobüs: 50 km/h
- Matris koordinatları kullanılarak perspektif dönüşümü yapılacak
- En az 3 saniye takip edilen araçlar için hız hesaplanacak

## 7. Çıktı Formatları

### 7.1 İhlal Kaydı Dosya Yapısı
```
ihlal_kayitlari/
└── [PLAKA]/
    ├── ihlal_[ID]_[TARIH]_[SAAT].jpg  # İhlal anı görüntüsü
    └── ihlal_[ID]_[TARIH]_[SAAT].txt  # İhlal detayları
```

### 7.2 Veritabanı Şeması
- **vehicles**: id, plate, city, vehicle_type, first_seen, last_seen
- **violations**: id, vehicle_id, violation_type, speed, speed_limit, light_color, timestamp, evidence_path

## 8. Uygulama Süreci

1. Test videosu ile sistem kalibrasyonu
2. Gerçek trafik kameralarına entegrasyon
3. 3 aylık pilot uygulama
4. Tam ölçekli uygulamaya geçiş

## 9. Genişletme Planları

- Çoklu kamera desteği
- Bulut tabanlı merkezi sistem
- Web arayüzü ve mobil uygulama
- Yapay zeka modellerinin sürekli iyileştirilmesi
- Ek ihlal türlerinin tespiti (emniyet şeridi ihlali, park ihlali vb.) 