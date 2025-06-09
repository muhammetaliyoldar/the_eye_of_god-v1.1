import cv2
import numpy as np
import sys
import os

# Global variables
points = []
image = None
window_name = "Koordinat Secici"
point_names = ["Sol Ust", "Sag Ust", "Sag Alt", "Sol Alt", "Sol Orta", "Sag Orta"]
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]  # BGR renkleri

def click_event(event, x, y, flags, param):
    """Mouse tıklama olayını yönetir"""
    global points, image
    
    # Sol tıklama olduğunda
    if event == cv2.EVENT_LBUTTONDOWN:
        # Eğer 6 nokta seçilmediyse
        if len(points) < 6:
            # Noktayı kaydet
            points.append((x, y))
            print(f"{point_names[len(points)-1]} secildi: ({x}, {y})")
            
            # Noktayı göster
            cv2.circle(image, (x, y), 5, colors[len(points)-1], -1)
            cv2.putText(image, f"{point_names[len(points)-1]}: ({x},{y})", 
                       (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[len(points)-1], 2)
            cv2.imshow(window_name, image)
            
            # Eğer 6 nokta seçildiyse sonuçları göster
            if len(points) == 6:
                show_results()

def show_results():
    """Seçilen noktaları gösterir ve sonuçları yazdırır"""
    global points, image
    
    # Dörtgeni çiz (ilk 4 nokta için)
    for i in range(4):
        cv2.line(image, points[i], points[(i+1)%4], (0, 255, 255), 2)
    
    # Orta noktaları göstermek için çizgiler
    cv2.line(image, points[4], points[5], (128, 128, 128), 2)  # Sol orta ile sağ orta arası
    
    # Sonuçları ekranda göster
    cv2.imshow(window_name, image)
    
    # Koordinatları konsola yazdır
    print("\nSecilen koordinatlar:")
    print(f"SOURCE = np.array([{points[0]}, {points[1]}, {points[2]}, {points[3]}, {points[4]}, {points[5]}])")
    
    # Koordinatları metin dosyasına kaydet
    with open("koordinatlar.txt", "w") as f:
        f.write(f"SOURCE = np.array([{points[0]}, {points[1]}, {points[2]}, {points[3]}, {points[4]}, {points[5]}])\n")
    
    print("\nKoordinatlar 'koordinatlar.txt' dosyasına kaydedildi.")
    print("Programdan çıkmak için herhangi bir tuşa basın.")

def main():
    global image, window_name
    
    # Komut satırı argümanı olarak resim dosyasını al
    if len(sys.argv) < 2:
        print("Kullanım: python coordinate_picker.py <resim_dosya_yolu>")
        return
    
    image_path = sys.argv[1]
    
    # Resim dosyasının var olup olmadığını kontrol et
    if not os.path.isfile(image_path):
        print(f"Hata: '{image_path}' dosyası bulunamadı.")
        return
    
    # Resmi oku
    image = cv2.imread(image_path)
    if image is None:
        print(f"Hata: '{image_path}' resmi okunamadı.")
        return
    
    # Pencere oluştur ve fare olaylarını dinle
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    
    # Talimatları göster
    instructions = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(instructions, "Talimatlar:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(instructions, "1. Sol Ust koseyi secin", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[0], 2)
    cv2.putText(instructions, "2. Sag Ust koseyi secin", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[1], 2)
    cv2.putText(instructions, "3. Sag Alt koseyi secin", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[2], 2)
    cv2.putText(instructions, "4. Sol Alt koseyi secin", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[3], 2)
    cv2.putText(instructions, "5. Sol Orta noktayi secin", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[4], 2)
    cv2.putText(instructions, "6. Sag Orta noktayi secin", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[5], 2)
    cv2.imshow("Talimatlar", instructions)
    
    # Resmi göster
    cv2.imshow(window_name, image)
    
    # Kullanıcı bir tuşa basana kadar bekle
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 