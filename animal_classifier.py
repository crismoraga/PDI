"""
Módulo para clasificación de animales usando Machine Learning
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import os
import pickle
import json
class AnimalClassifier:
    """
    Clasificador de animales usando Transfer Learning con MobileNetV2
    """
    
    def __init__(self, model_path=None):
        """
        Inicializar el clasificador
        
        Args:
            model_path (str): Ruta del modelo entrenado
        """
        self.model = None
        self.class_names = []
        self.model_path = model_path or "model/animal_classifier.h5"
        self.labels_path = "model/class_labels.json"
        
        # Cargar o crear modelo
        self.load_or_create_model()
        
    def load_or_create_model(self):
        """Cargar modelo existente o crear uno nuevo"""
        if os.path.exists(self.model_path) and os.path.exists(self.labels_path):
            print("📁 Cargando modelo existente...")
            self.load_model()
        else:
            print("🔧 Creando nuevo modelo...")
            self.create_pretrained_model()
            
    def create_pretrained_model(self):
        """Crear modelo usando transfer learning con MobileNetV2"""
        print("🏗️ Creando modelo basado en MobileNetV2...")
        
        # Cargar modelo preentrenado de MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        
        self.model = base_model
        
        # Cargar etiquetas de ImageNet (que incluye muchos animales)
        self.class_names = self._load_imagenet_classes()
        
        # Filtrar solo clases de animales
        self.animal_classes = self._filter_animal_classes()
        
        print(f"✅ Modelo creado con {len(self.animal_classes)} clases de animales")
        
    def _load_imagenet_classes(self):
        """Cargar clases de ImageNet"""
        # Las clases de ImageNet están incluidas en TensorFlow/Keras
        # Usaremos decode_predictions para obtener los nombres
        return list(range(1000))  # ImageNet tiene 1000 clases
        
    def _filter_animal_classes(self):
        """Filtrar solo clases de animales de ImageNet"""
        # Lista de clases de animales comunes en ImageNet
        animal_keywords = [
            'dog', 'cat', 'bird', 'fish', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'tiger', 'lion', 'wolf', 'fox',
            'rabbit', 'squirrel', 'monkey', 'ape', 'pig', 'duck', 'goose',
            'chicken', 'rooster', 'turkey', 'eagle', 'hawk', 'owl', 'parrot',
            'penguin', 'flamingo', 'swan', 'pelican', 'shark', 'whale',
            'dolphin', 'turtle', 'frog', 'snake', 'lizard', 'spider',
            'butterfly', 'bee', 'ant', 'beetle', 'fly', 'dragonfly'
        ]
        
        # Crear dummy test para obtener nombres de clases
        dummy_prediction = np.zeros((1, 1000))
        dummy_prediction[0, 0] = 1.0
        
        # Esta es una aproximación - en un proyecto real usarías la lista completa
        animal_indices = list(range(151, 269))  # Rango aproximado de animales en ImageNet
        animal_indices.extend(range(281, 285))  # Algunos otros animales
        
        return animal_indices
        
    def predict(self, image_array):
        """
        Realizar predicción sobre una imagen
        
        Args:
            image_array (numpy.ndarray): Imagen preprocesada
            
        Returns:
            tuple: (nombre_animal, confianza)
        """
        if self.model is None:
            return "Modelo no cargado", 0.0
            
        try:
            # Asegurar que la imagen tenga el formato correcto
            if image_array.shape != (1, 224, 224, 3):
                # Redimensionar si es necesario
                from tensorflow.keras.preprocessing.image import img_to_array
                if len(image_array.shape) == 4:
                    img = image_array[0]
                else:
                    img = image_array
                    
                img = tf.image.resize(img, [224, 224])
                img = tf.expand_dims(img, 0)
                image_array = img
                
            # Preprocesar para MobileNetV2
            processed_image = preprocess_input(image_array * 255.0)
            
            # Realizar predicción
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Decodificar predicciones de ImageNet
            decoded = decode_predictions(predictions, top=5)[0]
            
            # Filtrar solo animales y traducir
            best_animal = None
            best_confidence = 0.0
            
            for _, class_name, confidence in decoded:
                # Traducir nombre del animal al español
                spanish_name = self._translate_to_spanish(class_name)
                if spanish_name and confidence > best_confidence:
                    best_animal = spanish_name
                    best_confidence = confidence
                    
            if best_animal is None:
                # Si no se encontró un animal específico, usar la predicción con mayor confianza
                _, class_name, confidence = decoded[0]
                best_animal = self._translate_to_spanish(class_name) or class_name
                best_confidence = confidence
                
            return best_animal, float(best_confidence)
            
        except Exception as e:
            print(f"❌ Error en predicción: {str(e)}")
            return "Error en predicción", 0.0
            
    def _translate_to_spanish(self, english_name):
        """
        Traducir nombres de animales del inglés al español
        
        Args:
            english_name (str): Nombre en inglés
            
        Returns:
            str: Nombre en español o None si no es un animal
        """
        # Diccionario de traducción para animales comunes
        translations = {
            # Perros
            'golden_retriever': 'Golden Retriever',
            'beagle': 'Beagle',
            'german_shepherd': 'Pastor Alemán',
            'labrador_retriever': 'Labrador',
            'chihuahua': 'Chihuahua',
            'bulldog': 'Bulldog',
            'poodle': 'Caniche',
            'border_collie': 'Border Collie',
            'siberian_husky': 'Husky Siberiano',
            'rottweiler': 'Rottweiler',
            
            # Gatos
            'persian_cat': 'Gato Persa',
            'siamese_cat': 'Gato Siamés',
            'maine_coon': 'Maine Coon',
            'british_shorthair': 'Británico de Pelo Corto',
            'tabby': 'Gato Atigrado',
            'egyptian_cat': 'Gato Egipcio',
            
            # Animales salvajes
            'tiger': 'Tigre',
            'lion': 'León',
            'leopard': 'Leopardo',
            'cheetah': 'Guepardo',
            'jaguar': 'Jaguar',
            'elephant': 'Elefante',
            'giraffe': 'Jirafa',
            'zebra': 'Cebra',
            'rhinoceros': 'Rinoceronte',
            'hippopotamus': 'Hipopótamo',
            'bear': 'Oso',
            'polar_bear': 'Oso Polar',
            'brown_bear': 'Oso Pardo',
            'panda': 'Oso Panda',
            'wolf': 'Lobo',
            'fox': 'Zorro',
            'deer': 'Ciervo',
            'moose': 'Alce',
            'elk': 'Wapití',
            'bison': 'Bisonte',
            'buffalo': 'Búfalo',
            
            # Aves
            'eagle': 'Águila',
            'hawk': 'Halcón',
            'falcon': 'Halcón',
            'owl': 'Búho',
            'parrot': 'Loro',
            'peacock': 'Pavo Real',
            'penguin': 'Pingüino',
            'flamingo': 'Flamenco',
            'swan': 'Cisne',
            'goose': 'Ganso',
            'duck': 'Pato',
            'chicken': 'Gallina',
            'rooster': 'Gallo',
            'turkey': 'Pavo',
            'crane': 'Grulla',
            'pelican': 'Pelícano',
            'seagull': 'Gaviota',
            'cardinal': 'Cardenal',
            'blue_jay': 'Arrendajo Azul',
            'robin': 'Petirrojo',
            'hummingbird': 'Colibrí',
            
            # Animales marinos
            'whale': 'Ballena',
            'dolphin': 'Delfín',
            'shark': 'Tiburón',
            'seal': 'Foca',
            'sea_lion': 'León Marino',
            'walrus': 'Morsa',
            'octopus': 'Pulpo',
            'squid': 'Calamar',
            'jellyfish': 'Medusa',
            'starfish': 'Estrella de Mar',
            'sea_turtle': 'Tortuga Marina',
            
            # Reptiles y anfibios
            'snake': 'Serpiente',
            'lizard': 'Lagarto',
            'crocodile': 'Cocodrilo',
            'alligator': 'Caimán',
            'turtle': 'Tortuga',
            'frog': 'Rana',
            'toad': 'Sapo',
            'salamander': 'Salamandra',
            'iguana': 'Iguana',
            'chameleon': 'Camaleón',
            
            # Otros mamíferos
            'horse': 'Caballo',
            'cow': 'Vaca',
            'pig': 'Cerdo',
            'sheep': 'Oveja',
            'goat': 'Cabra',
            'donkey': 'Burro',
            'mule': 'Mula',
            'llama': 'Llama',
            'alpaca': 'Alpaca',
            'camel': 'Camello',
            'rabbit': 'Conejo',
            'hare': 'Liebre',
            'squirrel': 'Ardilla',
            'chipmunk': 'Ardilla Listada',
            'beaver': 'Castor',
            'otter': 'Nutria',
            'raccoon': 'Mapache',
            'skunk': 'Mofeta',
            'opossum': 'Zarigüeya',
            'bat': 'Murciélago',
            'monkey': 'Mono',
            'ape': 'Simio',
            'gorilla': 'Gorila',
            'chimpanzee': 'Chimpancé',
            'orangutan': 'Orangután',
            'baboon': 'Babuino',
            'lemur': 'Lémur',
            
            # Insectos
            'butterfly': 'Mariposa',
            'moth': 'Polilla',
            'bee': 'Abeja',
            'wasp': 'Avispa',
            'ant': 'Hormiga',
            'beetle': 'Escarabajo',
            'ladybug': 'Mariquita',
            'dragonfly': 'Libélula',
            'fly': 'Mosca',
            'mosquito': 'Mosquito',
            'spider': 'Araña',
            'scorpion': 'Escorpión',
            'centipede': 'Ciempiés',
            'millipede': 'Milpiés'
        }
        
        # Buscar traducción exacta
        if english_name.lower() in translations:
            return translations[english_name.lower()]
            
        # Buscar por palabras clave
        english_lower = english_name.lower()
        for eng_key, spanish_name in translations.items():
            if eng_key in english_lower or english_lower in eng_key:
                return spanish_name
                
        # Si contiene palabras relacionadas con animales, devolver el nombre original
        animal_indicators = ['dog', 'cat', 'bird', 'fish', 'animal', 'mammal', 'reptile']
        if any(indicator in english_lower for indicator in animal_indicators):
            return english_name.replace('_', ' ').title()
            
        return None
        
    def save_model(self):
        """Guardar el modelo entrenado"""
        if self.model is not None:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Guardar modelo
            self.model.save(self.model_path)
            
            # Guardar etiquetas
            with open(self.labels_path, 'w', encoding='utf-8') as f:
                json.dump(self.class_names, f, ensure_ascii=False, indent=2)
                
            print(f"✅ Modelo guardado en {self.model_path}")
            
    def load_model(self):
        """Cargar modelo existente"""
        try:
            self.model = keras.models.load_model(self.model_path)
            
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                self.class_names = json.load(f)
                
            print(f"✅ Modelo cargado desde {self.model_path}")
            
        except Exception as e:
            print(f"❌ Error al cargar modelo: {str(e)}")
            self.create_pretrained_model()
            
    def get_model_info(self):
        """Obtener información del modelo"""
        if self.model is None:
            return None
            
        return {
            'model_type': 'MobileNetV2 + Transfer Learning',
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.class_names),
            'parameters': self.model.count_params()
        }

def test_classifier():
    """Función de prueba para el clasificador"""
    print("🧪 Probando clasificador de animales...")
    
    classifier = AnimalClassifier()
    
    # Crear imagen de prueba
    test_image = np.random.rand(1, 224, 224, 3)
    
    # Realizar predicción
    animal_name, confidence = classifier.predict(test_image)
    
    print(f"✅ Predicción: {animal_name} (Confianza: {confidence:.2%})")
    
    # Mostrar información del modelo
    info = classifier.get_model_info()
    if info:
        print(f"📊 Información del modelo:")
        for key, value in info.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    test_classifier()
