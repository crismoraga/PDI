#!/usr/bin/env python3
"""
Pokedex Animal - Aplicación de Reconocimiento de Animales
Asignatura: Procesamiento Digital de Imágenes

Este programa utiliza OpenCV para capturar video de la cámara,
TensorFlow/Keras para clasificar animales usando machine learning,
y APIs web para obtener información detallada sobre los animales detectados.

Autor: Estudiante PDI
Fecha: Septiembre 2025
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import sys
import os

# Importar módulos personalizados
from utils.camera import CameraCapture
from utils.image_processing import ImageProcessor
from utils.api import AnimalInfoAPI
from model.animal_classifier import AnimalClassifier

class AnimalPokedexApp:
    """
    Aplicación principal de la Pokedex Animal
    """
    
    def __init__(self):
        """Inicializar la aplicación"""
        self.root = tk.Tk()
        self.root.title("🐾 Pokedex Animal - PDI Project")
        self.root.geometry("1200x800")
        
        # Variables de control
        self.camera_active = False
        self.current_prediction = ""
        self.confidence_score = 0.0
        
        # Inicializar componentes
        self.setup_components()
        self.setup_ui()
        
    def setup_components(self):
        """Configurar los componentes principales"""
        print("🔧 Inicializando componentes...")
        
        # Inicializar cámara
        self.camera = CameraCapture()
        
        # Inicializar procesador de imágenes
        self.image_processor = ImageProcessor()
        
        # Inicializar clasificador de animales
        self.classifier = AnimalClassifier()
        
        # Inicializar API de información
        self.animal_api = AnimalInfoAPI()
        
        print("✅ Componentes inicializados correctamente")
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="🐾 Pokedex Animal", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Frame de video
        video_frame = ttk.LabelFrame(main_frame, text="📹 Cámara en Vivo", padding="10")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()
        
        # Frame de información
        info_frame = ttk.LabelFrame(main_frame, text="📋 Información del Animal", padding="10")
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Predicción actual
        self.prediction_label = ttk.Label(info_frame, text="Animal: No detectado", 
                                         font=('Arial', 14, 'bold'))
        self.prediction_label.pack(anchor='w', pady=(0, 10))
        
        # Confianza
        self.confidence_label = ttk.Label(info_frame, text="Confianza: 0%")
        self.confidence_label.pack(anchor='w', pady=(0, 10))
        
        # Información detallada
        self.info_text = tk.Text(info_frame, wrap=tk.WORD, height=20, width=40)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar para el texto
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        # Frame de controles
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0))
        
        # Botones de control
        self.start_button = ttk.Button(controls_frame, text="🎥 Iniciar Cámara", 
                                      command=self.toggle_camera)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.capture_button = ttk.Button(controls_frame, text="📸 Capturar y Analizar", 
                                        command=self.capture_and_analyze)
        self.capture_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.quit_button = ttk.Button(controls_frame, text="❌ Salir", 
                                     command=self.quit_app)
        self.quit_button.pack(side=tk.LEFT)
        
    def toggle_camera(self):
        """Activar/desactivar la cámara"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """Iniciar la captura de cámara"""
        if self.camera.start():
            self.camera_active = True
            self.start_button.config(text="⏹️ Detener Cámara")
            self.update_video_feed()
            print("📹 Cámara iniciada")
        else:
            print("❌ Error al iniciar la cámara")
            
    def stop_camera(self):
        """Detener la captura de cámara"""
        self.camera_active = False
        self.camera.stop()
        self.start_button.config(text="🎥 Iniciar Cámara")
        print("⏹️ Cámara detenida")
        
    def update_video_feed(self):
        """Actualizar el feed de video"""
        if self.camera_active:
            frame = self.camera.get_frame()
            if frame is not None:
                # Redimensionar frame para la UI
                frame = cv2.resize(frame, (640, 480))
                
                # Convertir de BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convertir a formato PIL
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Actualizar label
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
            # Programar siguiente actualización
            self.root.after(30, self.update_video_feed)
            
    def capture_and_analyze(self):
        """Capturar frame actual y analizarlo"""
        if not self.camera_active:
            print("⚠️ La cámara no está activa")
            return
            
        frame = self.camera.get_frame()
        if frame is None:
            print("❌ No se pudo capturar el frame")
            return
            
        print("🔍 Analizando imagen...")
        
        # Procesar imagen
        processed_frame = self.image_processor.preprocess_for_classification(frame)
        
        # Realizar predicción
        prediction, confidence = self.classifier.predict(processed_frame)
        
        # Actualizar UI con predicción
        self.current_prediction = prediction
        self.confidence_score = confidence
        
        self.prediction_label.config(text=f"Animal: {prediction}")
        self.confidence_label.config(text=f"Confianza: {confidence:.1%}")
        
        # Buscar información del animal en internet
        if confidence > 0.3:  # Solo si la confianza es mayor al 30%
            self.search_animal_info(prediction)
        else:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "Confianza muy baja. Intenta con mejor iluminación o acércate más al animal.")
            
    def search_animal_info(self, animal_name):
        """Buscar información del animal en internet"""
        def search_thread():
            try:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, "🔍 Buscando información...")
                
                info = self.animal_api.get_animal_info(animal_name)
                
                self.info_text.delete(1.0, tk.END)
                if info:
                    formatted_info = self.format_animal_info(info)
                    self.info_text.insert(tk.END, formatted_info)
                else:
                    self.info_text.insert(tk.END, f"No se encontró información para: {animal_name}")
                    
            except Exception as e:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, f"Error al buscar información: {str(e)}")
                
        # Ejecutar búsqueda en hilo separado
        threading.Thread(target=search_thread, daemon=True).start()
        
    def format_animal_info(self, info):
        """Formatear la información del animal para mostrar"""
        formatted = f"🐾 {info.get('name', 'Desconocido')}\n\n"
        
        if info.get('summary'):
            formatted += f"📖 Descripción:\n{info['summary']}\n\n"
            
        if info.get('habitat'):
            formatted += f"🏠 Hábitat:\n{info['habitat']}\n\n"
            
        if info.get('diet'):
            formatted += f"🍽️ Dieta:\n{info['diet']}\n\n"
            
        if info.get('characteristics'):
            formatted += f"⭐ Características:\n{info['characteristics']}\n\n"
            
        if info.get('conservation_status'):
            formatted += f"🌍 Estado de Conservación:\n{info['conservation_status']}\n\n"
            
        return formatted
        
    def quit_app(self):
        """Cerrar la aplicación"""
        self.stop_camera()
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Ejecutar la aplicación"""
        print("🚀 Iniciando Pokedex Animal...")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.mainloop()

def main():
    """Función principal"""
    print("=" * 60)
    print("🐾 POKEDEX ANIMAL - Procesamiento Digital de Imágenes")
    print("=" * 60)
    
    try:
        app = AnimalPokedexApp()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️ Aplicación interrumpida por el usuario")
    except Exception as e:
        print(f"❌ Error crítico: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 ¡Gracias por usar Pokedex Animal!")

if __name__ == "__main__":
    main()
