"""
Módulo para obtener información de animales desde APIs web
"""

import requests
import json
import time
from urllib.parse import quote
import wikipedia
from bs4 import BeautifulSoup

class AnimalInfoAPI:
    """
    Clase para obtener información detallada de animales desde diferentes fuentes
    """
    
    def __init__(self):
        """Inicializar el cliente de API"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AnimalPokedex/1.0 (Educational Project)'
        })
        
        # Configurar Wikipedia en español
        wikipedia.set_lang("es")
        
    def get_animal_info(self, animal_name):
        """
        Obtener información completa de un animal
        
        Args:
            animal_name (str): Nombre del animal
            
        Returns:
            dict: Información del animal
        """
        print(f" Buscando información para: {animal_name}")
        
        info = {
            'name': animal_name,
            'summary': None,
            'habitat': None,
            'diet': None,
            'characteristics': None,
            'conservation_status': None,
            'scientific_name': None,
            'images': []
        }
        
        # Intentar obtener información de Wikipedia
        wiki_info = self._get_wikipedia_info(animal_name)
        if wiki_info:
            info.update(wiki_info)
            
        # Intentar obtener información adicional de otras fuentes
        additional_info = self._get_additional_info(animal_name)
        if additional_info:
            # Combinar información
            for key, value in additional_info.items():
                if value and not info.get(key):
                    info[key] = value
                    
        return info
        
    def _get_wikipedia_info(self, animal_name):
        """
        Obtener información de Wikipedia
        
        Args:
            animal_name (str): Nombre del animal
            
        Returns:
            dict: Información de Wikipedia
        """
        try:
            print(" Consultando Wikipedia...")
            
            # Buscar páginas relacionadas
            search_results = wikipedia.search(animal_name, results=3)
            
            if not search_results:
                return None
                
            # Intentar obtener la página más relevante
            page = None
            for result in search_results:
                try:
                    page = wikipedia.page(result)
                    break
                except wikipedia.exceptions.DisambiguationError as e:
                    # Si hay ambigüedad, tomar la primera opción
                    try:
                        page = wikipedia.page(e.options[0])
                        break
                    except:
                        continue
                except:
                    continue
                    
            if not page:
                return None
                
            # Extraer información
            summary = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
            
            # Buscar información específica en el contenido
            content = page.content.lower()
            
            habitat = self._extract_habitat_info(content)
            diet = self._extract_diet_info(content)
            characteristics = self._extract_characteristics(content)
            conservation = self._extract_conservation_status(content)
            
            return {
                'summary': summary,
                'habitat': habitat,
                'diet': diet,
                'characteristics': characteristics,
                'conservation_status': conservation,
                'url': page.url
            }
            
        except Exception as e:
            print(f" Error al consultar Wikipedia: {str(e)}")
            return None
            
    def _extract_habitat_info(self, content):
        """Extraer información de hábitat del contenido"""
        habitat_keywords = [
            'hábitat', 'habitat', 'vive en', 'se encuentra en', 'habita en',
            'distribución', 'geografía', 'región', 'bosque', 'selva', 'desierto',
            'océano', 'río', 'montaña', 'praderas'
        ]
        
        sentences = content.split('.')
        habitat_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in habitat_keywords):
                habitat_sentences.append(sentence.strip())
                if len(habitat_sentences) >= 2:
                    break
                    
        return '. '.join(habitat_sentences) if habitat_sentences else None
        
    def _extract_diet_info(self, content):
        """Extraer información de dieta del contenido"""
        diet_keywords = [
            'alimentación', 'dieta', 'come', 'se alimenta', 'carnívoro', 'herbívoro',
            'omnívoro', 'depredador', 'caza', 'presa', 'frutos', 'plantas', 'carne'
        ]
        
        sentences = content.split('.')
        diet_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in diet_keywords):
                diet_sentences.append(sentence.strip())
                if len(diet_sentences) >= 2:
                    break
                    
        return '. '.join(diet_sentences) if diet_sentences else None
        
    def _extract_characteristics(self, content):
        """Extraer características del contenido"""
        char_keywords = [
            'características', 'descripción', 'tamaño', 'peso', 'color', 'pelaje',
            'plumas', 'comportamiento', 'longitud', 'altura', 'forma'
        ]
        
        sentences = content.split('.')
        char_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in char_keywords):
                char_sentences.append(sentence.strip())
                if len(char_sentences) >= 3:
                    break
                    
        return '. '.join(char_sentences) if char_sentences else None
        
    def _extract_conservation_status(self, content):
        """Extraer estado de conservación del contenido"""
        conservation_keywords = [
            'conservación', 'extinción', 'amenazado', 'peligro', 'vulnerable',
            'estable', 'protegido', 'especie protegida', 'iucn'
        ]
        
        sentences = content.split('.')
        conservation_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence for keyword in conservation_keywords):
                conservation_sentences.append(sentence.strip())
                if len(conservation_sentences) >= 1:
                    break
                    
        return '. '.join(conservation_sentences) if conservation_sentences else None
        
    def _get_additional_info(self, animal_name):
        """
        Obtener información adicional de otras fuentes
        
        Args:
            animal_name (str): Nombre del animal
            
        Returns:
            dict: Información adicional
        """
        try:
            # Aquí se pueden agregar más fuentes de información
            # Por ejemplo, APIs de biodiversidad, enciclopedias online, etc.
            
            # Placeholder para futuras implementaciones
            return {}
            
        except Exception as e:
            print(f"⚠️ Error al obtener información adicional: {str(e)}")
            return {}
            
    def search_animal_images(self, animal_name, max_images=3):
        """
        Buscar imágenes del animal (placeholder para futura implementación)
        
        Args:
            animal_name (str): Nombre del animal
            max_images (int): Número máximo de imágenes
            
        Returns:
            list: Lista de URLs de imágenes
        """
        # Esta función se puede implementar usando APIs como Unsplash, Pixabay, etc.
        # Por ahora retorna una lista vacía
        return []
        
    def get_animal_facts(self, animal_name):
        """
        Obtener datos curiosos del animal
        
        Args:
            animal_name (str): Nombre del animal
            
        Returns:
            list: Lista de datos curiosos
        """
        try:
            # Buscar datos curiosos en Wikipedia
            page = wikipedia.page(animal_name)
            content = page.content
            
            # Buscar secciones con datos interesantes
            facts = []
            
            # Buscar oraciones que contengan números o datos específicos
            sentences = content.split('.')
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in 
                      ['puede', 'llega a', 'hasta', 'aproximadamente', 'record', 'único']):
                    facts.append(sentence.strip())
                    if len(facts) >= 5:
                        break
                        
            return facts
            
        except Exception as e:
            print(f"⚠️ Error al obtener datos curiosos: {str(e)}")
            return []

def test_animal_api():
    """Función de prueba para la API de animales"""
    print(" Probando API de información de animales...")
    
    api = AnimalInfoAPI()
    
    # Probar con algunos animales
    test_animals = ["perro", "gato", "león"]
    
    for animal in test_animals:
        print(f"\n Probando con: {animal}")
        info = api.get_animal_info(animal)
        
        if info:
            print(f" Información obtenida para {animal}")
            print(f"   Resumen: {info.get('summary', 'N/A')[:100]}...")
            print(f"   Hábitat: {info.get('habitat', 'N/A')[:100]}...")
        else:
            print(f" No se pudo obtener información para {animal}")
            
        time.sleep(1)  # Pausa para no sobrecargar la API

if __name__ == "__main__":
    test_animal_api()
