import cv2
import numpy as np
import matplotlib.pyplot as plt

class HomographyTransformer:
    """
    Perspektif dönüşüm işlemlerini gerçekleştiren sınıf.
    """
    def __init__(self, source_points, target_width, target_height):
        """
        Args:
            source_points (np.array): Video üzerindeki 4 kalibrasyon noktası [üst sol, üst sağ, alt sağ, alt sol]
            target_width (int): Çıktı perspektif genişliği
            target_height (int): Çıktı perspektif yüksekliği
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
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(self.target_points, self.source_points)
        
        self.target_width = target_width
        self.target_height = target_height
    
    def transform_image(self, image):
        """
        Görüntüyü perspektif dönüşüm ile kuş bakışı görünümüne çevirir
        
        Args:
            image (np.array): Dönüştürülecek görüntü
            
        Returns:
            np.array: Dönüştürülmüş görüntü
        """
        return cv2.warpPerspective(
            image, 
            self.transform_matrix, 
            (self.target_width, self.target_height)
        )
    
    def inverse_transform_image(self, bird_view_image, original_shape):
        """
        Kuş bakışı görüntüyü orijinal perspektife geri çevirir
        
        Args:
            bird_view_image (np.array): Kuş bakışı görüntü
            original_shape (tuple): Orijinal görüntünün boyutu (height, width)
            
        Returns:
            np.array: Orijinal perspektife dönüştürülmüş görüntü
        """
        return cv2.warpPerspective(
            bird_view_image,
            self.inverse_transform_matrix,
            (original_shape[1], original_shape[0])
        )
    
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
    
    def inverse_transform_point(self, point):
        """
        Bir noktayı ters perspektif dönüşüm ile çevirir
        
        Args:
            point (np.array): Dönüştürülecek nokta [x, y]
            
        Returns:
            np.array: Dönüştürülmüş nokta [x, y]
        """
        point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(point, self.inverse_transform_matrix)
        return transformed.reshape(-1, 2)[0]
    
    def inverse_transform_points(self, points):
        """
        Çoklu noktaları ters perspektif dönüşüm ile çevirir
        
        Args:
            points (np.array): Dönüştürülecek noktalar, şekli (n, 2)
            
        Returns:
            np.array: Dönüştürülmüş noktalar, şekli (n, 2)
        """
        if len(points) == 0:
            return points
        
        points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(points, self.inverse_transform_matrix)
        return transformed.reshape(-1, 2)
    
    def draw_region_on_image(self, image, color=(0, 255, 0), thickness=2):
        """
        Görüntü üzerinde kalibrasyon bölgesini çizer
        
        Args:
            image (np.array): Çizim yapılacak görüntü
            color (tuple, optional): Çizgi rengi (B, G, R)
            thickness (int, optional): Çizgi kalınlığı
            
        Returns:
            np.array: Çizim yapılmış görüntü
        """
        img = image.copy()
        
        # Kapalı poligon çizimi
        pts = self.source_points.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [pts], True, color, thickness)
        
        return img
    
    def visualize_transformation(self, image):
        """
        Orijinal görüntü ve kuş bakışı dönüşümünü yan yana gösterir
        
        Args:
            image (np.array): Orijinal görüntü
            
        Returns:
            np.array: Birleştirilmiş görselleştirme
        """
        # Orijinal görüntü üzerine bölgeyi çiz
        region_marked = self.draw_region_on_image(image)
        
        # Kuş bakışı dönüşüm
        bird_view = self.transform_image(image)
        
        # İki görüntüyü yan yana birleştir
        h1, w1 = region_marked.shape[:2]
        h2, w2 = bird_view.shape[:2]
        
        # Yükseklikleri eşitle
        scale = h1 / h2
        bird_view_resized = cv2.resize(bird_view, (int(w2 * scale), h1))
        
        # Yan yana birleştir
        combined = np.hstack((region_marked, bird_view_resized))
        
        return combined 