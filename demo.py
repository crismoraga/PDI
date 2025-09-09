#!/usr/bin/env python3
"""
Demo de Pokedex Animal - Versión sin cámara para pruebas
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import sys
import os

# Importar módulos personalizados
from utils.image_processing import ImageProcessor
from utils.api import AnimalInfoAPI
from model.animal_classifier import AnimalClassifier

class AnimalPokedexDemo:
    """
    Versión demo de la Pokedex Animal (sin cámara)
    """
    
    def __init__(self):
        """Inicializar la aplicación demo"""
        self.root = tk.Tk()
        self.root.title("🐾 Pokedex Animal - Demo (Sin Cámara)")
        self.root.geometry("1000x700")
        
        # Variables de control
        self.current_image = None
        self.current_prediction = ""
        self.confidence_score = 0.0
        
        # Inicializar componentes
        self.setup_components()
        self.setup_ui()
        
    def setup_components(self):
        """Configurar los componentes principales"""
        print("🔧 Inicializando componentes...")
        
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
        title_label = ttk.Label(main_frame, text="🐾 Pokedex Animal - Demo", 
                               font=('Arial', 20, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Frame de imagen
        image_frame = ttk.LabelFrame(main_frame, text="📸 Imagen del Animal", padding="10")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="Selecciona una imagen para analizar")
        self.image_label.pack()
        
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
        self.info_text = tk.Text(info_frame, wrap=tk.WORD, height=15, width=40)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar para el texto
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        # Frame de controles
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0))
        
        # Botones de control
        self.load_button = ttk.Button(controls_frame, text="📁 Cargar Imagen", 
                                     command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.analyze_button = ttk.Button(controls_frame, text="🔍 Analizar Animal", 
                                        command=self.analyze_image)
        self.analyze_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.demo_button = ttk.Button(controls_frame, text="🎯 Análisis Demo", 
                                     command=self.demo_analysis)
        self.demo_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.quit_button = ttk.Button(controls_frame, text="❌ Salir", 
                                     command=self.quit_app)
        self.quit_button.pack(side=tk.LEFT)
        
        # Información inicial
        self.show_welcome_message()
        
    def show_welcome_message(self):
        """Mostrar mensaje de bienvenida"""
        welcome = """🐾 ¡Bienvenido a Pokedex Animal!

Esta es la versión demo del proyecto de Procesamiento Digital de Imágenes.

FUNCIONALIDADES:
• 📁 Cargar Imagen: Selecciona una foto de un animal
• 🔍 Analizar Animal: Identifica la especie usando IA
• 🎯 Análisis Demo: Prueba con imágenes predefinidas
• 🌐 Información: Busca datos del animal en internet

TECNOLOGÍAS USADAS:
• OpenCV: Procesamiento de imágenes
• TensorFlow: Machine Learning (MobileNetV2)
• Wikipedia API: Información de especies
• Tkinter: Interfaz gráfica

INSTRUCCIONES:
1. Carga una imagen de un animal
2. Presiona "Analizar Animal"
3. Ve la predicción y información detallada

¡Prueba la funcionalidad demo para ver el sistema en acción!"""
        
        self.info_text.insert(tk.END, welcome)
        
    def load_image(self):
        """Cargar imagen desde archivo"""
        file_types = [
            ('Imágenes', '*.jpg *.jpeg *.png *.bmp *.gif *.tiff'),
            ('JPEG', '*.jpg *.jpeg'),
            ('PNG', '*.png'),
            ('Todos los archivos', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Seleccionar imagen de animal",
            filetypes=file_types
        )
        
        if filename:
            try:
                # Cargar imagen
                self.current_image = cv2.imread(filename)
                
                if self.current_image is None:
                    messagebox.showerror("Error", "No se pudo cargar la imagen")
                    return
                
                # Mostrar imagen en la interfaz
                self.display_image(self.current_image)
                
                print(f"✅ Imagen cargada: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar imagen: {str(e)}")
                
    def display_image(self, image):
        """Mostrar imagen en la interfaz"""
        # Redimensionar para mostrar
        display_image = image.copy()
        
        # Calcular nuevo tamaño manteniendo proporción
        height, width = display_image.shape[:2]
        max_size = 400
        
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
            
        display_image = cv2.resize(display_image, (new_width, new_height))
        
        # Convertir de BGR a RGB
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        # Convertir a formato PIL
        pil_image = Image.fromarray(display_image_rgb)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Actualizar label
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
        
    def analyze_image(self):
        """Analizar la imagen cargada"""
        if self.current_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
            
        print("🔍 Analizando imagen...")
        
        try:
            # Procesar imagen
            processed_image = self.image_processor.preprocess_for_classification(self.current_image)
            
            # Realizar predicción
            prediction, confidence = self.classifier.predict(processed_image)
            
            # Actualizar UI con predicción
            self.current_prediction = prediction
            self.confidence_score = confidence
            
            self.prediction_label.config(text=f"Animal: {prediction}")
            self.confidence_label.config(text=f"Confianza: {confidence:.1%}")
            
            # Buscar información del animal
            if confidence > 0.1:  # Umbral más bajo para demo
                self.search_animal_info(prediction)
            else:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, "Confianza muy baja. La imagen podría no contener un animal reconocible o la calidad es insuficiente.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al analizar imagen: {str(e)}")
            print(f"❌ Error en análisis: {str(e)}")
            
    def demo_analysis(self):
        """Realizar análisis de demostración"""
        print("🎯 Ejecutando análisis demo...")
        
        # Crear imagen sintética de ejemplo
        demo_image = self.create_demo_image()
        self.current_image = demo_image
        
        # Mostrar imagen
        self.display_image(demo_image)
        
        # Simular análisis
        demo_animals = ["Perro", "Gato", "León", "Tigre", "Elefante"]
        demo_confidence = [0.85, 0.92, 0.78, 0.88, 0.75]
        
        import random
        idx = random.randint(0, len(demo_animals)-1)
        
        prediction = demo_animals[idx]
        confidence = demo_confidence[idx]
        
        self.current_prediction = prediction
        self.confidence_score = confidence
        
        self.prediction_label.config(text=f"Animal: {prediction}")
        self.confidence_label.config(text=f"Confianza: {confidence:.1%}")
        
        # Mostrar información demo
        self.show_demo_info(prediction)
        
    def create_demo_image(self):
        """Crear imagen demo para pruebas"""
        # Crear imagen sintética
        image = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        
        # Agregar algunos patrones para simular un animal
        cv2.circle(image, (200, 200), 100, (139, 69, 19), -1)  # Cuerpo marrón
        cv2.circle(image, (170, 170), 30, (139, 69, 19), -1)   # Cabeza
        cv2.circle(image, (160, 160), 5, (0, 0, 0), -1)        # Ojo
        cv2.circle(image, (180, 160), 5, (0, 0, 0), -1)        # Ojo
        
        return image
        
    def search_animal_info(self, animal_name):
        """Buscar información del animal"""
        def search_thread():
            try:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, "🔍 Buscando información...")
                self.root.update()
                
                # Para demo, mostrar información básica
                self.show_demo_info(animal_name)
                    
            except Exception as e:
                self.info_text.delete(1.0, tk.END)
                self.info_text.insert(tk.END, f"Error al buscar información: {str(e)}")
                
        # Ejecutar búsqueda en hilo separado
        threading.Thread(target=search_thread, daemon=True).start()
        
    def show_demo_info(self, animal_name):
        """Mostrar información demo del animal"""
        # Información demo predefinida
        demo_info = {
            "Perro": {
                "description": "El perro (Canis lupus familiaris) es un mamífero carnívoro doméstico. Es una subespecie del lobo gris y está relacionado con zorros, chacales y otros cánidos.",
                "habitat": "Los perros domésticos viven junto a los humanos en todo el mundo. Se adaptan a diversos ambientes urbanos y rurales.",
                "diet": "Los perros son omnívoros que se alimentan de carne, vegetales y granos. Su dieta debe incluir proteínas, carbohidratos y grasas.",
                "characteristics": "Poseen un excelente sentido del olfato y oído. Son animales sociales, leales y pueden ser entrenados para diversas tareas.",
                "conservation": "Especie doméstica estable. Sin riesgo de extinción."
            },
            "Gato": {
                "description": "El gato doméstico (Felis catus) es un mamífero carnívoro pequeño. Es la única especie domesticada de la familia Felidae.",
                "habitat": "Los gatos domésticos viven en hogares humanos en todo el mundo. También existen poblaciones ferales en diversos hábitats.",
                "diet": "Son carnívoros obligados que requieren una dieta rica en proteínas animales, especialmente taurina.",
                "characteristics": "Poseen excelente visión nocturna, agilidad y flexibilidad. Son cazadores naturales con garras retráctiles.",
                "conservation": "Especie doméstica estable. Sin riesgo de extinción."
            },
            "León": {
                "description": "El león (Panthera leo) es un felino grande conocido como 'el rey de la selva'. Vive en grupos llamados manadas.",
                "habitat": "Principalmente en sabanas africanas, con una pequeña población en India (Bosque de Gir).",
                "diet": "Carnívoro que caza grandes mamíferos como cebras, ñus, búfalos y antílopes.",
                "characteristics": "Los machos tienen melena distintiva. Peso: 150-250 kg. Excelentes cazadores grupales.",
                "conservation": "Vulnerable - Población en declive debido a pérdida de hábitat y caza."
            },
            "Tigre": {
                "description": "El tigre (Panthera tigris) es el felino más grande del mundo. Conocido por sus rayas distintivas.",
                "habitat": "Bosques, manglares y pastizales de Asia (India, China, Rusia, Sudeste Asiático).",
                "diet": "Carnívoro que caza ciervos, jabalíes, búfalos de agua y otros grandes mamíferos.",
                "characteristics": "Rayas únicas como huellas dactilares. Peso: 140-300 kg. Cazadores solitarios.",
                "conservation": "En peligro - Quedan aproximadamente 3,900 tigres en estado salvaje."
            },
            "Elefante": {
                "description": "Los elefantes son los mamíferos terrestres más grandes. Existen dos especies: africano y asiático.",
                "habitat": "Sabanas africanas, bosques tropicales de África y Asia.",
                "diet": "Herbívoros que consumen hasta 300 kg de vegetación diaria: hierbas, frutas, cortezas.",
                "characteristics": "Trompa versátil, memoria excepcional, estructura social matriarcal. Peso: 4,000-7,000 kg.",
                "conservation": "Vulnerable - Amenazados por caza furtiva y pérdida de hábitat."
            }
        }
        
        info = demo_info.get(animal_name, {
            "description": f"Información sobre {animal_name} no disponible en modo demo.",
            "habitat": "Información no disponible",
            "diet": "Información no disponible", 
            "characteristics": "Información no disponible",
            "conservation": "Información no disponible"
        })
        
        formatted_info = f"""🐾 {animal_name}

📖 Descripción:
{info['description']}

🏠 Hábitat:
{info['habitat']}

🍽️ Dieta:
{info['diet']}

⭐ Características:
{info['characteristics']}

🌍 Estado de Conservación:
{info['conservation']}

---
ℹ️ Esta es información demo. En la versión completa se obtiene de APIs externas en tiempo real."""
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, formatted_info)
        
    def quit_app(self):
        """Cerrar la aplicación"""
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Ejecutar la aplicación"""
        print("🚀 Iniciando Pokedex Animal Demo...")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.mainloop()

def main():
    """Función principal"""
    print("=" * 60)
    print("🐾 POKEDEX ANIMAL - DEMO SIN CÁMARA")
    print("=" * 60)
    
    try:
        app = AnimalPokedexDemo()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️ Aplicación interrumpida por el usuario")
    except Exception as e:
        print(f"❌ Error crítico: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("👋 ¡Gracias por usar Pokedex Animal Demo!")

if __name__ == "__main__":
    main()
