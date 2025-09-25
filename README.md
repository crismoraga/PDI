# Pokédex Animal - Proyecto PDI

## Descripción

**Pokédx Animal** es una aplicación avanzada de reconocimiento de animales en tiempo real que utiliza visión por computadora, machine learning e inteligencia artificial para identificar animales a través de la cámara y proporcionar información detallada sobre ellos, replicando el funcionamiento de una Pokédx pero para animales reales.

## Objetivos del Proyecto

- **Principal**: Crear una aplicación que identifique animales en tiempo real usando la cámara
- **Secundario**: Aplicar técnicas avanzadas de Procesamiento Digital de Imágenes (PDI)
- **Técnico**: Integrar OpenCV, TensorFlow Lite y APIs web en una aplicación optimizada
- **Educativo**: Demostrar el uso práctico de ML/AI en aplicaciones reales para Raspberry Pi

## Tecnologías Utilizadas

### Core Technologies

- **Python 3.8+**: Lenguaje principal
- **OpenCV 4.8+**: Captura y procesamiento de video/imágenes
- **TensorFlow Lite**: Machine Learning optimizado para edge devices
- **Tkinter**: Interfaz gráfica moderna y responsiva
- **SQLite**: Base de datos Pokédx para persistencia de entradas

### Machine Learning

- **MobileNetV2**: Modelo preentrenado optimizado para dispositivos móviles
- **TensorFlow Lite Runtime**: Inferencia rápida en Raspberry Pi
- **Transfer Learning**: Aprovecha conocimiento de ImageNet
- **Edge TPU Support**: Aceleración opcional con Coral Edge TPU

### APIs y Datos

- **Wikipedia API**: Información detallada y actualizada de animales
- **BeautifulSoup**: Web scraping para datos adicionales
- **Requests**: Cliente HTTP robusto para APIs

### Procesamiento de Imágenes

- **Filtros de mejora**: CLAHE, reducción de ruido
- **Detección de objetos**: Contornos y bounding boxes
- **Segmentación avanzada**: K-means y Watershed
- **Análisis visual**: Color dominante, tamaño relativo, características

## Características Principales

### Funcionalidades Pokédx

- **Captura en tiempo real**: Reconocimiento automático de animales
- **Base de datos persistente**: Almacena cada encuentro con timestamp
- **Sistema de "capturados"**: Marca animales vistos vs capturados
- **Vista de detalle**: Información completa de cada entrada
- **Exportación**: JSON y Markdown para análisis externos
- **Búsqueda y filtros**: Encuentra entradas por nombre o estado

### Características Técnicas

- **Optimizado para Raspberry Pi**: Rendimiento optimizado para hardware limitado
- **Detección de características visuales**: Color, tamaño, bounding box
- **Interfaz moderna**: UI responsiva con tema oscuro
- **Manejo de errores robusto**: Fallbacks automáticos
- **Logging completo**: Trazabilidad de operaciones
- **Configuración flexible**: Parámetros ajustables por JSON

## Estructura del Proyecto

```text
PDI/
├── main.py                    # Aplicación principal
├── demo.py                   # Versión demo sin cámara
├── requirements.txt          # Dependencias Python
├── config.json              # Configuración de la aplicación
├── RASPBERRY_PI_SETUP.md    # Guía detallada para Raspberry Pi
├── INSTALL_RPI.md           # Instrucciones de instalación Pi
├── TECHNICAL_DOCS.md        # Documentación técnica
├── data/                    # Datos y modelos
│   ├── snapshots/          # Imágenes capturadas
│   ├── exports/            # Exportaciones JSON/MD
│   └── pokedx.db          # Base de datos SQLite
├── model/                   # Modelos de Machine Learning
│   ├── __init__.py
│   ├── animal_classifier.py # Clasificador Keras
│   └── tflite_classifier.py # Clasificador TensorFlow Lite
├── utils/                   # Módulos auxiliares
│   ├── __init__.py
│   ├── camera.py           # Manejo de cámara
│   ├── image_processing.py # Procesamiento de imágenes
│   └── api.py             # APIs externas
├── pokedx/                 # Sistema Pokédx
│   ├── __init__.py
│   └── db.py              # Repositorio y modelos de datos
├── scripts/                # Scripts de utilidad
│   └── download_tflite_model.py # Descarga de modelos
└── tests/                  # Suite de pruebas
    ├── test_all.py
    └── final_check.py
```

## Instalación Rápida (Desktop)

### Prerrequisitos

- Python 3.8 o superior
- Webcam/cámara conectada
- Conexión a Internet

### Instalación

```bash
# Clonar el proyecto
git clone <url-del-repositorio>
cd PDI

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
python main.py
```

## Instalación en Raspberry Pi

Para una instalación completa y optimizada en Raspberry Pi, consulta la guía detallada en:

- **[RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)** - Guía paso a paso completa
- **[INSTALL_RPI.md](INSTALL_RPI.md)** - Instrucciones técnicas específicas

### Resumen Rápido Pi

```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias del sistema
sudo apt install -y python3-dev python3-pip python3-venv
sudo apt install -y libatlas-base-dev libjpeg-dev libpng-dev

# Configurar proyecto
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Ejecutar
python main.py
```

## Uso de la Aplicación

### Interfaz Principal

1. **Iniciar cámara**: Botón para activar el feed de video
2. **Capturar y analizar**: Procesa la imagen actual
3. **Guardar captura**: Marca el animal como "capturado"
4. **Vista de detalle**: Muestra información completa
5. **Exportar**: Genera archivos JSON o Markdown

### Flujo Pokédx

1. Inicia la cámara
2. Apunta a un animal
3. Presiona "Capturar y analizar"
4. La IA identifica la especie
5. Se busca información automáticamente
6. Se guarda la entrada en la Pokédx
7. Opcionalmente marca como "capturado"

### Características Visuales

La aplicación analiza automáticamente:
- **Color dominante**: Identifica el color principal
- **Tamaño relativo**: Porcentaje del frame ocupado
- **Bounding box**: Coordenadas de detección
- **Confianza**: Nivel de certeza de la identificación

## Optimizaciones para Raspberry Pi

### Configuración de Rendimiento

- Resolución optimizada: 640x480 @ 15fps
- TensorFlow Lite Runtime para menor uso de memoria
- Procesamiento en hilos separados
- Cache inteligente de resultados
- Configuración automática de CPU governor

### Soporte Edge TPU

Compatible con Google Coral Edge TPU para aceleración de IA:

```bash
# Instalar soporte Edge TPU
sudo apt install libedgetpu1-std
pip install pycoral

# Usar modelo Edge TPU
export USE_EDGE_TPU=true
python main.py
```

## API y Extensibilidad

### Configuración JSON

```json
{
    "camera": {
        "resolution": [640, 480],
        "fps": 15,
        "auto_exposure": true
    },
    "ml": {
        "confidence_threshold": 0.7,
        "model_path": "data/models/mobilenet_v2.tflite",
        "use_edge_tpu": false
    },
    "ui": {
        "theme": "dark",
        "language": "es",
        "window_size": [1280, 820]
    }
}
```

### Base de Datos

Esquema SQLite completo con campos:
- Información básica (nombre, confianza, timestamp)
- Metadata (nickname, notas, estado capturado)
- Características visuales (color, tamaño, bbox)
- Información externa (Wikipedia, hábitat, dieta)

## Testing y Validación

### Suite de Pruebas

```bash
# Ejecutar todas las pruebas
python test_all.py

# Verificación completa del sistema
python final_check.py

# Demo sin cámara (para desarrollo)
python demo.py
```

### Resultado Esperado

```
✅ OpenCV: PASÓ
✅ TensorFlow Lite: PASÓ  
✅ Cámara: PASÓ
✅ Modelo ML: PASÓ
✅ Base de datos: PASÓ
✅ APIs externas: PASÓ

RESULTADO: 6/6 pruebas exitosas (100%)
```

## Contribución y Desarrollo

### Estructura de Desarrollo

1. **Fork** el repositorio
2. Crea una **branch** para tu feature
3. Implementa cambios con **tests**
4. Asegura que **final_check.py** pase
5. Envía **pull request**

### Estándares de Código

- **Type hints** en todas las funciones
- **Docstrings** en clases y métodos públicos
- **Error handling** robusto
- **Logging** apropiado
- **Sin emojis** en código o documentación

## Licencia y Créditos

Este proyecto es desarrollado con fines educativos para demostrar técnicas avanzadas de procesamiento digital de imágenes y machine learning aplicado a la clasificación de animales.

### Tecnologías Utilizadas

- OpenCV para procesamiento de imágenes
- TensorFlow/TensorFlow Lite para machine learning
- Wikipedia API para información de especies
- SQLite para persistencia de datos

## Soporte y Documentación

- **Documentación técnica**: [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)
- **Guía Raspberry Pi**: [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)
- **Instalación Pi**: [INSTALL_RPI.md](INSTALL_RPI.md)
- **Issues**: Reporta problemas en el repositorio

Para soporte específico de Raspberry Pi o problemas de rendimiento, consulta la documentación técnica detallada incluida en el proyecto.
