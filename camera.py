"""
Módulo para manejo de cámara usando OpenCV
"""

import cv2
import numpy as np
import threading
import time

class CameraCapture:
    """
    Clase para manejar la captura de video desde la cámara
    """
    
    def __init__(self, camera_index=0):
        """
        Inicializar la captura de cámara
        
        Args:
            camera_index (int): Índice de la cámara (0 para la cámara principal)
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        
    def start(self):
        """
        Iniciar la captura de cámara
        
        Returns:
            bool: True si se inició correctamente, False en caso contrario
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"❌ No se puede abrir la cámara {self.camera_index}")
                return False
                
            # Configurar propiedades de la cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            
            # Iniciar hilo de captura
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            print("✅ Cámara iniciada correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error al iniciar la cámara: {str(e)}")
            return False
            
    def stop(self):
        """Detener la captura de cámara"""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        print("⏹️ Cámara detenida")
        
    def _capture_loop(self):
        """Loop de captura en hilo separado"""
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            
            if ret:
                with self.lock:
                    self.current_frame = frame.copy()
            else:
                print("⚠️ No se pudo leer frame de la cámara")
                time.sleep(0.1)
                
    def get_frame(self):
        """
        Obtener el frame actual de la cámara
        
        Returns:
            numpy.ndarray: Frame actual o None si no hay frame disponible
        """
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
            
    def is_camera_available(self):
        """
        Verificar si la cámara está disponible
        
        Returns:
            bool: True si la cámara está disponible
        """
        return self.is_running and self.cap is not None and self.cap.isOpened()
        
    def get_camera_info(self):
        """
        Obtener información de la cámara
        
        Returns:
            dict: Diccionario con información de la cámara
        """
        if not self.is_camera_available():
            return None
            
        info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        return info
        
    def capture_photo(self, filename=None):
        """
        Capturar una foto y guardarla
        
        Args:
            filename (str): Nombre del archivo. Si es None, genera uno automático
            
        Returns:
            str: Ruta del archivo guardado o None si falló
        """
        frame = self.get_frame()
        if frame is None:
            return None
            
        if filename is None:
            timestamp = int(time.time())
            filename = f"capture_{timestamp}.jpg"
            
        try:
            cv2.imwrite(filename, frame)
            print(f"📸 Foto guardada: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error al guardar foto: {str(e)}")
            return None

def test_camera():
    """Función de prueba para la cámara"""
    print("🧪 Probando la cámara...")
    
    camera = CameraCapture()
    
    if camera.start():
        print("✅ Cámara iniciada correctamente")
        
        # Mostrar info de la cámara
        info = camera.get_camera_info()
        if info:
            print(f"📹 Resolución: {info['width']}x{info['height']}")
            print(f"🎬 FPS: {info['fps']}")
        
        # Capturar algunos frames
        for i in range(5):
            frame = camera.get_frame()
            if frame is not None:
                print(f"✅ Frame {i+1} capturado: {frame.shape}")
            else:
                print(f"❌ No se pudo capturar frame {i+1}")
            time.sleep(1)
            
        camera.stop()
    else:
        print("❌ No se pudo iniciar la cámara")

if __name__ == "__main__":
    test_camera()
