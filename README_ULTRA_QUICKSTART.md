# Pokedex Animal Ultra - Windows Edition

**Version Ultra-Profesional con Maxima Capacidad para Windows**

Sistema avanzado de reconocimiento de animales en tiempo real con IA de ultima generacion, interfaz grafica futurista y pipeline de entrenamiento profesional.

## INICIO RAPIDO

### Requisitos Minimos
- Windows 10/11 (64-bit)
- Python 3.12.x
- 8 GB RAM
- Camara web
- (Opcional) GPU NVIDIA para maxima performance

### Instalacion Express

```powershell
# Clonar repositorio
git clone https://github.com/tu-usuario/PDI.git
cd PDI

# Instalar dependencias
pip install -r requirements_windows_ultra_py312.txt

# Verificar instalacion
python verify_system_ultra.py

# Ejecutar aplicacion
python pokedex_ultra_windows.py
```

## ESTADO DE INSTALACION

Ver `INSTALLATION_STATUS.md` para detalles completos del estado actual.

**Sistema actual: FUNCIONAL EN CPU**
- Todos los paquetes core instalados
- Camara verificada
- GPU opcional para maxima performance

## ARQUITECTURA

### Motor de IA Ensemble
- **YOLO v8x**: Deteccion de objetos en tiempo real
- **EfficientNetB7**: Clasificacion de alta precision
- **MobileNetV2**: Velocidad y eficiencia

### Procesamiento Multi-Thread
- **Video Thread**: Captura 60 FPS
- **Prediction Thread**: Inferencia 10-15 FPS
- **Metrics Thread**: Monitoreo 1 Hz

### Interfaz Grafica Futurista
- CustomTkinter con animaciones
- Panel de video en tiempo real
- Metricas de sistema
- Informacion detallada de especies

### Base de Datos Avanzada
- SQLite con modo WAL
- Sistema de logros
- Estadisticas completas
- Full-text search

## CARACTERISTICAS

### Reconocimiento en Tiempo Real
- Deteccion multi-modelo
- Confianza ponderada
- Extraccion de features visuales
- Bounding boxes precisos

### Entrenamiento Profesional
- Transfer learning avanzado
- Data augmentation con Albumentations
- TensorBoard monitoring
- Exportacion TFLite

### Monitoreo de Sistema
- FPS video y predicciones
- Uso CPU/GPU/RAM
- Tiempo de procesamiento
- Metricas en tiempo real

### Sistema de Logros
- 10+ logros desbloqueables
- Sistema de niveles
- Experiencia acumulativa
- Estadisticas completas

## DOCUMENTACION

### Guias Completas
- `WINDOWS_ULTRA_INSTALLATION.md`: Instalacion paso a paso
- `TRAINING_GUIDE.md`: Guia de entrenamiento profesional
- `INSTALLATION_STATUS.md`: Estado actual del sistema
- `TECHNICAL_DOCS.md`: Documentacion tecnica completa

### Archivos de Configuracion
- `requirements_windows_ultra_py312.txt`: Dependencias Python 3.12
- `config.json`: Configuracion basica
- `config_advanced.json`: Configuracion avanzada
- `config_optimized.json`: Optimizaciones

## USO

### Modo Basico
```powershell
python pokedex_ultra_windows.py
```

### Entrenar Modelos
```powershell
python train_professional_models.py --dataset data/training --epochs 50
```

### Verificar Sistema
```powershell
python verify_system_ultra.py
```

## RENDIMIENTO

### Con GPU (CUDA 11.8)
- Video: 60 FPS
- Predicciones: 10-15 FPS
- Latencia: 66-100ms

### Sin GPU (CPU)
- Video: 30 FPS
- Predicciones: 2-5 FPS
- Latencia: 200-500ms

## ESTRUCTURA DEL PROYECTO

```
PDI/
├── pokedex_ultra_windows.py     # Aplicacion principal
├── train_professional_models.py # Sistema de entrenamiento
├── verify_system_ultra.py       # Verificacion rigurosa
├── model/
│   ├── animal_classifier.py     # Clasificador base
│   ├── tflite_classifier.py     # Clasificador TFLite
│   └── ultra/                   # Modelos entrenados
├── utils/
│   ├── camera.py               # Manejo de camara
│   ├── image_processing.py     # Procesamiento de imagenes
│   └── api.py                  # Integracion API
├── pokedex/
│   └── db.py                   # Base de datos
├── data/
│   ├── pokedex_ultra.db        # Base de datos SQLite
│   ├── snapshots_ultra/        # Capturas automaticas
│   ├── training/               # Dataset entrenamiento
│   └── logs_ultra/             # Logs y TensorBoard
└── docs/
    ├── WINDOWS_ULTRA_INSTALLATION.md
    ├── TRAINING_GUIDE.md
    └── INSTALLATION_STATUS.md
```

## TROUBLESHOOTING

### GPU no detectada
```powershell
# Verificar driver NVIDIA
nvidia-smi

# Instalar CUDA 11.8
# https://developer.nvidia.com/cuda-11-8-0-download-archive

# Reinstalar PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Camara no funciona
```powershell
# Verificar permisos de camara en Windows
# Configuracion > Privacidad > Camara

# Probar indices diferentes en codigo
# camera_index = 0, 1, 2...
```

### Importacion falla
```powershell
# Reinstalar paquetes problematicos
pip install --upgrade --force-reinstall opencv-contrib-python
pip install --upgrade --force-reinstall tensorflow
```

## CONTRIBUIR

Este es un proyecto educativo de vision por computadora. Contribuciones bienvenidas.

## LICENCIA

MIT License - Ver archivo LICENSE para detalles

## CONTACTO

Proyecto desarrollado para PDI - Procesamiento Digital de Imagenes

## CHANGELOG

### v2.0.0 - Ultra Windows Edition (30/09/2025)
- Sistema ensemble de 3 modelos IA
- Interfaz CustomTkinter futurista
- Multi-threading optimizado
- Entrenamiento profesional
- Verificacion rigurosa
- Compatibilidad Python 3.12
- Base de datos avanzada
- Sistema de logros
