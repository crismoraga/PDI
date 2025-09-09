# 🐾 Pokedex Animal - Proyecto PDI

## Descripción

**Pokedex Animal** es una aplicación de reconocimiento de animales en tiempo real que utiliza visión por computadora, machine learning e inteligencia artificial para identificar animales a través de la cámara y proporcionar información detallada sobre ellos, similar a una Pokédex pero para animales reales.

## 🎯 Objetivos del Proyecto

- **Principal**: Crear una aplicación que identifique animales en tiempo real usando la cámara
- **Secundario**: Aplicar técnicas de Procesamiento Digital de Imágenes (PDI)
- **Técnico**: Integrar OpenCV, TensorFlow y APIs web en una aplicación funcional
- **Educativo**: Demostrar el uso práctico de ML/AI en aplicaciones reales

## 🛠️ Tecnologías Utilizadas

### Core Technologies
- **Python 3.8+**: Lenguaje principal
- **OpenCV**: Captura y procesamiento de video/imágenes
- **TensorFlow/Keras**: Machine Learning y clasificación de imágenes
- **Tkinter**: Interfaz gráfica de usuario

### Machine Learning
- **MobileNetV2**: Modelo preentrenado para transfer learning
- **Transfer Learning**: Aprovecha conocimiento de ImageNet
- **Clasificación de imágenes**: Reconocimiento de especies animales

### APIs y Datos
- **Wikipedia API**: Información detallada de animales
- **BeautifulSoup**: Web scraping para datos adicionales
- **Requests**: Cliente HTTP para APIs

### Procesamiento de Imágenes
- **Filtros de mejora**: CLAHE, reducción de ruido
- **Detección de objetos**: Contornos y bounding boxes
- **Segmentación**: K-means y Watershed
- **Normalización**: Preprocesamiento para ML

## 📁 Estructura del Proyecto

```
PDI/
├── main.py                 # Aplicación principal
├── requirements.txt        # Dependencias Python
├── README.md              # Documentación (este archivo)
├── data/                  # Datos de entrenamiento (futuro)
├── model/                 # Modelos de Machine Learning
│   ├── __init__.py
│   └── animal_classifier.py
└── utils/                 # Módulos auxiliares
    ├── __init__.py
    ├── camera.py          # Manejo de cámara
    ├── image_processing.py # Procesamiento de imágenes
    └── api.py            # APIs externas
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- Webcam/cámara conectada
- Conexión a Internet (para búsqueda de información)

### Paso 1: Clonar/Descargar el Proyecto

```bash
# Si usas Git
git clone <url-del-repositorio>
cd PDI

# O simplemente descargar y extraer en la carpeta PDI
```

### Paso 2: Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar Instalación

```bash
# Probar cámara
python utils/camera.py

# Probar procesamiento de imágenes
python utils/image_processing.py

# Probar clasificador
python model/animal_classifier.py

# Probar API
python utils/api.py
```

## 🎮 Uso de la Aplicación

### Ejecutar la Aplicación

```bash
python main.py
```

### Interfaz de Usuario

1. **Iniciar Cámara**: Presiona "🎥 Iniciar Cámara" para activar la webcam
2. **Vista en Vivo**: El video aparece en tiempo real en el panel izquierdo
3. **Capturar y Analizar**: Presiona "📸 Capturar y Analizar" para identificar el animal
4. **Ver Información**: La información aparece en el panel derecho
5. **Detener**: Usa "⏹️ Detener Cámara" para pausar o "❌ Salir" para cerrar

### Flujo de Trabajo

1. **Captura**: La cámara captura frames en tiempo real
2. **Preprocesamiento**: OpenCV mejora la imagen (filtros, normalización)
3. **Clasificación**: TensorFlow/MobileNetV2 identifica el animal
4. **Búsqueda**: Se consulta Wikipedia para obtener información
5. **Visualización**: Se muestra el resultado en la interfaz

## 🧠 Arquitectura del Sistema

### Módulo Principal (`main.py`)
- **AnimalPokedexApp**: Clase principal de la aplicación
- **Interfaz GUI**: Tkinter para la experiencia de usuario
- **Coordinación**: Integra todos los módulos

### Módulo de Cámara (`utils/camera.py`)
- **CameraCapture**: Manejo de webcam con OpenCV
- **Threading**: Captura asíncrona de frames
- **Configuración**: Resolución, FPS, propiedades de cámara

### Módulo de Procesamiento (`utils/image_processing.py`)
- **ImageProcessor**: Algoritmos de PDI
- **Filtros**: CLAHE, Gaussian blur, sharpening
- **Detección**: Contornos, bounding boxes
- **Segmentación**: K-means clustering, Watershed

### Módulo de Clasificación (`model/animal_classifier.py`)
- **AnimalClassifier**: Red neuronal con transfer learning
- **MobileNetV2**: Modelo base preentrenado
- **Traducción**: Mapeo inglés-español de especies
- **Predicción**: Clasificación con scores de confianza

### Módulo de APIs (`utils/api.py`)
- **AnimalInfoAPI**: Cliente para fuentes de información
- **Wikipedia**: Consultas automáticas
- **Extracción**: Hábitat, dieta, características
- **Formateo**: Presentación legible de datos

## 🔬 Técnicas de PDI Implementadas

### 1. Mejora de Imágenes
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
  - Mejora el contraste local
  - Evita la amplificación excesiva de ruido
  
- **Filtro Mediano**
  - Reducción de ruido salt-and-pepper
  - Preserva bordes importantes

- **Sharpening**
  - Realza detalles y bordes
  - Kernel de convolución personalizado

### 2. Detección y Segmentación
- **Detección de Bordes (Canny)**
  ```python
  edges = cv2.Canny(blurred, 50, 150)
  ```

- **Segmentación por Contornos**
  ```python
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  ```

- **K-means Clustering**
  ```python
  _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  ```

### 3. Preprocesamiento para ML
- **Normalización**: Valores 0-255 → 0-1
- **Redimensionado**: Ajuste a 224x224 (entrada MobileNetV2)
- **Formato de lote**: Expansión de dimensiones para predicción

## 🤖 Machine Learning

### Modelo Base: MobileNetV2
- **Arquitectura**: Efficient neural network para móviles
- **Preentrenamiento**: ImageNet (1000 clases, incluyendo animales)
- **Transfer Learning**: Aprovecha conocimiento previo

### Pipeline de Clasificación
1. **Entrada**: Imagen 224x224x3
2. **Preprocesamiento**: MobileNetV2 preprocessing
3. **Predicción**: Forward pass por la red
4. **Decodificación**: Top-5 predicciones de ImageNet
5. **Filtrado**: Solo clases de animales
6. **Traducción**: Nombres en español

### Clases de Animales Soportadas
- **Mamíferos**: Perros, gatos, caballos, vacas, osos, etc.
- **Aves**: Águilas, loros, pingüinos, flamencos, etc.
- **Reptiles**: Serpientes, lagartos, tortugas, etc.
- **Animales Marinos**: Ballenas, delfines, tiburones, etc.
- **Insectos**: Mariposas, abejas, escarabajos, etc.

## 🌐 Integración con APIs

### Wikipedia API
```python
import wikipedia
wikipedia.set_lang("es")
page = wikipedia.page(animal_name)
summary = page.summary
```

### Extracción de Información
- **Hábitat**: Búsqueda por palabras clave relacionadas
- **Dieta**: Identificación de patrones alimentarios
- **Características**: Descripción física y comportamiento
- **Conservación**: Estado de amenaza/protección

## 📊 Métricas y Rendimiento

### Precisión del Modelo
- **Modelo base**: MobileNetV2 (Top-1: ~71%, Top-5: ~90% en ImageNet)
- **Animales específicos**: Variable según especie y calidad de imagen
- **Filtrado**: Solo predicciones con >30% confianza

### Rendimiento del Sistema
- **FPS de cámara**: ~30 FPS
- **Tiempo de predicción**: ~100-300ms
- **Tiempo de búsqueda API**: ~1-3 segundos
- **Memoria RAM**: ~500MB-1GB (según modelo)

## 🐛 Solución de Problemas

### Problemas Comunes

1. **Cámara no detectada**
   ```python
   # Verificar índice de cámara
   python utils/camera.py
   ```

2. **Error de dependencias**
   ```bash
   pip install --upgrade tensorflow opencv-python
   ```

3. **Predicciones incorrectas**
   - Mejorar iluminación
   - Acercarse al animal
   - Evitar fondos complejos

4. **Error de conexión API**
   - Verificar conexión a Internet
   - Reintentar después de unos segundos

### Logs y Debugging
- Los errores se muestran en consola
- Usar modo verbose para más detalles
- Verificar cada módulo individualmente

## 🔮 Futuras Mejoras

### Modelo de ML
- [ ] Entrenar modelo específico para animales
- [ ] Aumentar dataset con más especies
- [ ] Implementar detección de múltiples animales
- [ ] Agregar reconocimiento de sonidos

### Funcionalidades
- [ ] Modo offline (base de datos local)
- [ ] Historial de detecciones
- [ ] Exportar información a PDF
- [ ] Modo de comparación de especies

### Técnicas Avanzadas
- [ ] YOLO para detección en tiempo real
- [ ] Segmentación semántica
- [ ] Análisis de comportamiento
- [ ] Realidad aumentada

### Interfaz
- [ ] Versión web (Flask/Django)
- [ ] App móvil (React Native)
- [ ] Temas personalizables
- [ ] Múltiples idiomas

## 📚 Referencias Académicas

1. **MobileNetV2**: Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018.

2. **CLAHE**: Zuiderveld, K. "Contrast Limited Adaptive Histogram Equalization." Graphics Gems IV, 1994.

3. **Transfer Learning**: Pan, S.J., Yang, Q. "A Survey on Transfer Learning." IEEE TKDE, 2010.

4. **OpenCV**: Bradski, G. "The OpenCV Library." Dr. Dobb's Journal, 2000.

## 👥 Créditos

- **Desarrollo**: Estudiante de Procesamiento Digital de Imágenes
- **Asignatura**: PDI - Universidad
- **Profesor**: [Nombre del profesor]
- **Fecha**: Septiembre 2025

## 📄 Licencia

Este proyecto es desarrollado con fines educativos para la asignatura de Procesamiento Digital de Imágenes.

## 🤝 Contribuciones

Para mejoras o reportar bugs:
1. Fork del proyecto
2. Crear branch para feature
3. Commit de cambios
4. Push al branch
5. Crear Pull Request

---

**¡Gracias por usar Pokedex Animal! 🐾**

*Proyecto desarrollado con ❤️ para la asignatura de PDI*
