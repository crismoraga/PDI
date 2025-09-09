"""
Módulo para procesamiento de imágenes usando OpenCV y técnicas de PDI
"""

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

class ImageProcessor:
    """
    Clase para procesamiento de imágenes y preparación para ML
    """
    
    def __init__(self):
        """Inicializar el procesador de imágenes"""
        self.target_size = (224, 224)  # Tamaño estándar para modelos de clasificación
        
    def preprocess_for_classification(self, image):
        """
        Preprocesar imagen para clasificación con modelo de ML
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            
        Returns:
            numpy.ndarray: Imagen preprocesada
        """
        if image is None:
            return None
            
        # Convertir de BGR a RGB si es necesario
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Redimensionar a tamaño objetivo
        resized = cv2.resize(image_rgb, self.target_size)
        
        # Normalizar valores de pixel (0-255 -> 0-1)
        normalized = resized.astype(np.float32) / 255.0
        
        # Expandir dimensiones para batch
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
        
    def enhance_image(self, image):
        """
        Mejorar la calidad de la imagen usando técnicas de PDI
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            
        Returns:
            numpy.ndarray: Imagen mejorada
        """
        if image is None:
            return None
            
        enhanced = image.copy()
        
        # 1. Reducción de ruido
        enhanced = cv2.medianBlur(enhanced, 5)
        
        # 2. Mejora de contraste usando CLAHE
        if len(enhanced.shape) == 3:
            # Para imágenes en color, aplicar CLAHE en el canal L del espacio LAB
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Para imágenes en escala de grises
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(enhanced)
            
        # 3. Enfoque (sharpening)
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
        
    def detect_objects(self, image):
        """
        Detectar objetos en la imagen usando técnicas de visión por computadora
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            
        Returns:
            list: Lista de bounding boxes de objetos detectados
        """
        if image is None:
            return []
            
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar filtro Gaussiano
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detección de bordes usando Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar contornos por área
        min_area = 1000
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Crear bounding boxes
        bboxes = []
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, x+w, y+h))
            
        return bboxes
        
    def crop_largest_object(self, image):
        """
        Recortar el objeto más grande de la imagen
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            
        Returns:
            numpy.ndarray: Imagen recortada del objeto más grande
        """
        bboxes = self.detect_objects(image)
        
        if not bboxes:
            return image
            
        # Encontrar el bbox más grande
        largest_bbox = max(bboxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2 = largest_bbox
        
        # Agregar padding
        padding = 20
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Recortar
        cropped = image[y1:y2, x1:x2]
        
        return cropped
        
    def apply_filters(self, image, filter_type="enhance"):
        """
        Aplicar diferentes filtros a la imagen
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            filter_type (str): Tipo de filtro a aplicar
            
        Returns:
            numpy.ndarray: Imagen con filtro aplicado
        """
        if image is None:
            return None
            
        if filter_type == "enhance":
            return self.enhance_image(image)
        elif filter_type == "blur":
            return cv2.GaussianBlur(image, (15, 15), 0)
        elif filter_type == "sharp":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(image, -1, kernel)
        elif filter_type == "edge":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            return image
            
    def create_histogram(self, image):
        """
        Crear histograma de la imagen
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            
        Returns:
            dict: Histogramas por canal
        """
        if image is None:
            return None
            
        histograms = {}
        
        if len(image.shape) == 3:
            # Imagen en color
            colors = ['b', 'g', 'r']
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                histograms[color] = hist
        else:
            # Imagen en escala de grises
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            histograms['gray'] = hist
            
        return histograms
        
    def segment_image(self, image, method="kmeans"):
        """
        Segmentar imagen usando diferentes métodos
        
        Args:
            image (numpy.ndarray): Imagen de entrada
            method (str): Método de segmentación
            
        Returns:
            numpy.ndarray: Imagen segmentada
        """
        if image is None:
            return None
            
        if method == "kmeans":
            # K-means clustering
            data = image.reshape((-1, 3))
            data = np.float32(data)
            
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            k = 3
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            centers = np.uint8(centers)
            segmented = centers[labels.flatten()]
            segmented = segmented.reshape(image.shape)
            
            return segmented
            
        elif method == "watershed":
            # Watershed algorithm
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Operaciones morfológicas
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Área de fondo segura
            sure_bg = cv2.dilate(opening, kernel, iterations=3)
            
            # Área de primer plano segura
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            
            # Región desconocida
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Etiquetas de marcadores
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            markers = cv2.watershed(image, markers)
            image[markers == -1] = [255, 0, 0]
            
            return image
            
        return image

def test_image_processing():
    """Función de prueba para procesamiento de imágenes"""
    print("🧪 Probando procesamiento de imágenes...")
    
    processor = ImageProcessor()
    
    # Crear imagen de prueba
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Probar preprocesamiento
    processed = processor.preprocess_for_classification(test_image)
    print(f"✅ Imagen preprocesada: {processed.shape}")
    
    # Probar mejora de imagen
    enhanced = processor.enhance_image(test_image)
    print(f"✅ Imagen mejorada: {enhanced.shape}")
    
    # Probar detección de objetos
    bboxes = processor.detect_objects(test_image)
    print(f"✅ Objetos detectados: {len(bboxes)}")

if __name__ == "__main__":
    test_image_processing()
