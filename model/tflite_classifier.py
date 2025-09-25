"""
Clasificador TensorFlow Lite optimizado para dispositivos edge (Raspberry Pi).
Proporciona inferencia rápida y eficiente para reconocimiento de animales en tiempo real.
"""

from __future__ import annotations

import json
import os
import time
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow.lite import Interpreter as TFInterpreter
        from tensorflow.lite.python.interpreter import load_delegate

        class tflite:
            Interpreter = TFInterpreter
            load_delegate = staticmethod(load_delegate)
    except ImportError:
        raise ImportError("Neither tflite_runtime nor tensorflow.lite available")


class TFLiteAnimalClassifier:
    """
    Clasificador de animales optimizado usando TensorFlow Lite.
    
    Características:
    - Inferencia rápida optimizada para dispositivos edge
    - Soporte para aceleración Edge TPU (Google Coral)
    - Traducción automática de nombres de especies
    - Filtrado de clases de animales
    - Análisis de confianza avanzado
    """
    
    def __init__(self, 
                 model_path: str = "model/animal_classifier.tflite", 
                 labels_path: str = "model/labels.txt", 
                 use_edgetpu: bool = False,
                 confidence_threshold: float = 0.1):
        """
        Inicializar el clasificador TensorFlow Lite.
        
        Args:
            model_path: Ruta al modelo .tflite
            labels_path: Ruta a archivo de etiquetas
            use_edgetpu: Si usar aceleración Edge TPU
            confidence_threshold: Umbral mínimo de confianza
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.use_edgetpu = use_edgetpu
        self.confidence_threshold = confidence_threshold
        
        # Cargar etiquetas y modelo
        self.labels = self._load_labels()
        self.animal_classes = self._filter_animal_classes()
        self.interpreter = self._load_interpreter()
        
        # Obtener detalles de entrada y salida
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Información del modelo
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        
        print(f"TensorFlow Lite classifier inicializado")
        print(f"- Modelo: {model_path}")
        print(f"- Input shape: {self.input_shape}")
        print(f"- Classes: {len(self.labels)}")
        print(f"- Animal classes: {len(self.animal_classes)}")
        print(f"- Edge TPU: {'Si' if use_edgetpu else 'No'}")

    def _load_labels(self) -> List[str]:
        """Cargar etiquetas de clasificación."""
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        
        # Fallback: generar etiquetas genéricas
        print(f"Warning: Labels file {self.labels_path} not found, using generic labels")
        return [f"class_{i}" for i in range(1000)]  # ImageNet típicamente tiene 1000 clases

    def _filter_animal_classes(self) -> List[str]:
        """Filtrar solo clases de animales de las etiquetas disponibles."""
        animal_keywords = [
            'dog', 'cat', 'bird', 'fish', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'tiger', 'lion', 'wolf', 'fox',
            'rabbit', 'squirrel', 'monkey', 'ape', 'pig', 'duck', 'goose',
            'chicken', 'rooster', 'turkey', 'eagle', 'hawk', 'owl', 'parrot',
            'penguin', 'flamingo', 'swan', 'pelican', 'shark', 'whale',
            'dolphin', 'turtle', 'frog', 'snake', 'lizard', 'spider',
            'butterfly', 'bee', 'ant', 'beetle', 'moth', 'dragonfly'
        ]
        
        animal_classes = []
        for i, label in enumerate(self.labels):
            label_lower = label.lower()
            if any(keyword in label_lower for keyword in animal_keywords):
                animal_classes.append(label)
        
        return animal_classes if animal_classes else self.labels[:100]  # Fallback

    def _load_interpreter(self) -> tflite.Interpreter:
        """Cargar intérprete TensorFlow Lite con soporte opcional para Edge TPU."""
        model_to_use = self.model_path
        delegates = []
        
        # Intentar usar Edge TPU si está habilitado
        if self.use_edgetpu:
            edgetpu_model = self.model_path.replace(".tflite", "_edgetpu.tflite")
            
            if os.path.exists(edgetpu_model):
                model_to_use = edgetpu_model
                try:
                    delegates = [tflite.load_delegate("libedgetpu.so.1")]
                    print("Edge TPU delegate cargado exitosamente")
                except Exception as e:
                    print(f"Warning: No se pudo cargar Edge TPU delegate: {e}")
            else:
                print(f"Warning: Modelo Edge TPU {edgetpu_model} no encontrado")
        
        # Verificar si el modelo existe
        if not os.path.exists(model_to_use):
            raise FileNotFoundError(f"Modelo TensorFlow Lite no encontrado: {model_to_use}")
        
        # Crear intérprete
        if delegates:
            interpreter = tflite.Interpreter(model_path=model_to_use, 
                                           experimental_delegates=delegates)
        else:
            interpreter = tflite.Interpreter(model_path=model_to_use)
        
        interpreter.allocate_tensors()
        return interpreter

    def predict(self, image_array: np.ndarray) -> Tuple[str, float]:
        """
        Realizar predicción de animal en imagen.
        
        Args:
            image_array: Array de imagen preprocessada (batch, height, width, channels)
        
        Returns:
            Tuple de (nombre_animal, confianza)
        """
        try:
            start_time = time.time()
            
            # Asegurar que el input tenga el formato correcto
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            # Redimensionar si es necesario
            expected_shape = tuple(self.input_shape[1:])  # Excluir dimensión de batch
            if image_array.shape[1:] != expected_shape:
                # Usar interpolación para redimensionar
                from PIL import Image
                h, w = expected_shape[:2]
                
                # Convertir a PIL y redimensionar
                if image_array.shape[0] == 1:
                    pil_image = Image.fromarray((image_array[0] * 255).astype(np.uint8))
                    pil_image = pil_image.resize((w, h), Image.LANCZOS)
                    image_array = np.expand_dims(np.array(pil_image) / 255.0, axis=0)
            
            # Convertir tipo de datos si es necesario
            if image_array.dtype != self.input_dtype:
                image_array = image_array.astype(self.input_dtype)
            
            # Realizar inferencia
            self.interpreter.set_tensor(self.input_details[0]['index'], image_array)
            self.interpreter.invoke()
            
            # Obtener predicciones
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Encontrar las mejores predicciones
            top_indices = np.argsort(predictions)[::-1][:5]  # Top 5 predicciones
            
            # Buscar la primera predicción que sea un animal con suficiente confianza
            for idx in top_indices:
                confidence = float(predictions[idx])
                
                if confidence >= self.confidence_threshold:
                    class_name = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
                    
                    # Verificar si es un animal
                    if self._is_animal_class(class_name):
                        animal_name = self._translate_to_spanish(class_name)
                        
                        inference_time = time.time() - start_time
                        print(f"Predicción: {animal_name} ({confidence:.2%}) en {inference_time:.3f}s")
                        
                        return animal_name, confidence
            
            # Si no se encontró animal con suficiente confianza
            return "Animal no identificado", 0.0
            
        except Exception as e:
            print(f"Error en predicción TensorFlow Lite: {e}")
            return "Error en predicción", 0.0

    def _is_animal_class(self, class_name: str) -> bool:
        """Verificar si una clase corresponde a un animal."""
        animal_keywords = [
            'dog', 'cat', 'bird', 'fish', 'horse', 'sheep', 'cow', 'elephant',
            'bear', 'zebra', 'giraffe', 'tiger', 'lion', 'wolf', 'fox',
            'rabbit', 'squirrel', 'monkey', 'ape', 'pig', 'duck', 'goose',
            'chicken', 'rooster', 'turkey', 'eagle', 'hawk', 'owl', 'parrot',
            'penguin', 'flamingo', 'swan', 'pelican', 'shark', 'whale',
            'dolphin', 'turtle', 'frog', 'snake', 'lizard', 'spider'
        ]
        
        class_lower = class_name.lower()
        return any(keyword in class_lower for keyword in animal_keywords)

    def _translate_to_spanish(self, english_name: str) -> str:
        """Traducir nombre de animal de inglés a español."""
        translations = {
            # Perros
            'golden_retriever': 'Golden Retriever',
            'labrador_retriever': 'Labrador Retriever',
            'german_shepherd': 'Pastor Alemán',
            'beagle': 'Beagle',
            'bulldog': 'Bulldog',
            'poodle': 'Caniche',
            'rottweiler': 'Rottweiler',
            'siberian_husky': 'Husky Siberiano',
            'chihuahua': 'Chihuahua',
            'dachshund': 'Dachshund',
            
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
            'monkey': 'Mono',
            'gorilla': 'Gorila',
            'chimpanzee': 'Chimpancé',
            'orangutan': 'Orangután',
            'kangaroo': 'Canguro',
            'koala': 'Koala',
            'sloth': 'Perezoso',
            'bat': 'Murciélago',
            'rabbit': 'Conejo',
            'squirrel': 'Ardilla',
            'beaver': 'Castor',
            'otter': 'Nutria',
            'raccoon': 'Mapache',
            'skunk': 'Zorrillo',
            'opossum': 'Zarigüeya',
            
            # Insectos y arácnidos
            'spider': 'Araña',
            'butterfly': 'Mariposa',
            'bee': 'Abeja',
            'ant': 'Hormiga',
            'beetle': 'Escarabajo',
            'moth': 'Polilla',
            'dragonfly': 'Libélula',
            'cricket': 'Grillo',
            'grasshopper': 'Saltamontes',
            'ladybug': 'Mariquita',
            'mosquito': 'Mosquito',
            'fly': 'Mosca',
            'wasp': 'Avispa',
            'caterpillar': 'Oruga',
            'cockroach': 'Cucaracha',
            'termite': 'Termita',
            'centipede': 'Ciempiés',
            'millipede': 'Milpiés'
        }
        
        # Buscar traducción exacta
        english_lower = english_name.lower()
        if english_lower in translations:
            return translations[english_lower]
        
        # Buscar por palabras clave
        for eng_key, spanish_name in translations.items():
            if eng_key in english_lower or english_lower in eng_key:
                return spanish_name
        
        # Si contiene palabras relacionadas con animales, devolver nombre formateado
        animal_indicators = ['dog', 'cat', 'bird', 'fish', 'animal', 'mammal', 'reptile']
        if any(indicator in english_lower for indicator in animal_indicators):
            return english_name.replace('_', ' ').title()
        
        # Último recurso: formatear nombre original
        return english_name.replace('_', ' ').title()

    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información detallada del modelo."""
        return {
            'model_type': 'TensorFlow Lite',
            'model_path': self.model_path,
            'input_shape': self.input_shape.tolist(),
            'input_dtype': str(self.input_dtype),
            'num_classes': len(self.labels),
            'animal_classes': len(self.animal_classes),
            'edge_tpu_enabled': self.use_edgetpu,
            'confidence_threshold': self.confidence_threshold
        }


def test_tflite_classifier():
    """Función de prueba para el clasificador TensorFlow Lite."""
    print("Probando clasificador TensorFlow Lite...")
    
    try:
        classifier = TFLiteAnimalClassifier()
        
        # Crear imagen de prueba
        input_shape = classifier.input_shape[1:]  # Excluir batch dimension
        test_image = np.random.rand(*input_shape).astype(classifier.input_dtype)
        
        # Realizar predicción
        animal_name, confidence = classifier.predict(test_image)
        
        print(f"Predicción de prueba: {animal_name} (Confianza: {confidence:.2%})")
        
        # Mostrar información del modelo
        info = classifier.get_model_info()
        print("Información del modelo:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        return True
        
    except Exception as e:
        print(f"Error en prueba de TensorFlow Lite: {e}")
        return False


if __name__ == "__main__":
    test_tflite_classifier()

    def predict(self, image_batched_rgb01: np.ndarray) -> Tuple[str, float]:
        # Ajustar al tamaño esperado
        input_shape = self.input_details[0]["shape"]
        _, h, w, c = input_shape
        img = image_batched_rgb01
        if img.shape[1] != h or img.shape[2] != w:
            import cv2

            img0 = img[0]
            img0 = cv2.resize((img0 * 255).astype(np.uint8), (w, h))
            img0 = img0.astype(np.float32) / 255.0
            img = np.expand_dims(img0, axis=0)

        # Tipado: float32 típico
        dtype = self.input_details[0]["dtype"]
        if dtype == np.uint8:
            tensor = (img * 255).astype(np.uint8)
        else:
            tensor = img.astype(dtype)

        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # Softmax si fuese necesario
        if output_data.ndim == 1 and np.abs(np.sum(output_data) - 1) > 1e-3:
            exps = np.exp(output_data - np.max(output_data))
            output_data = exps / np.sum(exps)

        idx = int(np.argmax(output_data))
        score = float(output_data[idx])
        label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
        return label, score
