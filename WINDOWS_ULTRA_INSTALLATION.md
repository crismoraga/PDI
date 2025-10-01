# Pokedex Animal Ultra - Guia de Instalacion Windows

Guia completa para instalar y configurar Pokedex Animal Ultra en Windows con aceleracion GPU, modelos de IA avanzados y maxima capacidad de procesamiento.

## Tabla de Contenidos

1. [Requisitos del Sistema](#requisitos-del-sistema)
2. [Instalacion de Python y Herramientas](#instalacion-de-python-y-herramientas)
3. [Configuracion de CUDA y GPU](#configuracion-de-cuda-y-gpu)
4. [Instalacion de Dependencias](#instalacion-de-dependencias)
5. [Descarga y Configuracion de Modelos](#descarga-y-configuracion-de-modelos)
6. [Configuracion de la Base de Datos](#configuracion-de-la-base-de-datos)
7. [Ejecucion y Verificacion](#ejecucion-y-verificacion)
8. [Optimizaciones de Rendimiento](#optimizaciones-de-rendimiento)
9. [Resolucion de Problemas](#resolucion-de-problemas)

---

## Requisitos del Sistema

### Hardware Minimo

- Procesador: Intel Core i5 8th Gen o AMD Ryzen 5 2600
- RAM: 16 GB DDR4
- GPU: NVIDIA GTX 1060 6GB o superior
- Almacenamiento: 50 GB SSD
- Camara web: 720p minimo

### Hardware Recomendado

- Procesador: Intel Core i7 10th Gen o AMD Ryzen 7 3700X
- RAM: 32 GB DDR4 3200MHz
- GPU: NVIDIA RTX 3060 12GB o superior
- Almacenamiento: 100 GB NVMe SSD
- Camara web: 1080p 60fps

### Software Requerido

- Windows 10/11 (64-bit)
- Python 3.10 o 3.11
- NVIDIA CUDA Toolkit 11.8
- cuDNN 8.6
- Visual Studio 2019/2022 Build Tools

---

## Instalacion de Python y Herramientas

### Paso 1: Instalar Python

1. Descargar Python 3.11 desde https://www.python.org/downloads/
2. Ejecutar instalador con las siguientes opciones:
   - Marcar "Add Python to PATH"
   - Marcar "Install for all users"
   - Seleccionar "Customize installation"
   - Marcar todas las opciones en "Optional Features"
   - En "Advanced Options" marcar:
     - Install for all users
     - Add Python to environment variables
     - Precompile standard library
     - Download debugging symbols

3. Verificar instalacion:

```powershell
python --version
# Debe mostrar: Python 3.11.x

pip --version
# Debe mostrar version de pip
```

### Paso 2: Instalar Visual Studio Build Tools

Requerido para compilar paquetes de Python con extensiones en C/C++.

1. Descargar desde: https://visualstudio.microsoft.com/downloads/
2. Seleccionar "Build Tools for Visual Studio 2022"
3. En el instalador, seleccionar:
   - "Desktop development with C++"
   - En la seccion de detalles individuales:
     - Windows 10/11 SDK
     - MSVC v143 - VS 2022 C++ x64/x86 build tools
     - C++ CMake tools for Windows

4. Instalar y reiniciar el sistema

### Paso 3: Instalar Git

```powershell
winget install Git.Git
# O descargar desde https://git-scm.com/download/win
```

---

## Configuracion de CUDA y GPU

### Verificar Compatibilidad de GPU

```powershell
nvidia-smi
```

Verificar que la GPU aparece listada y la version del driver.

### Instalar CUDA Toolkit

1. Descargar CUDA Toolkit 11.8 desde:
   https://developer.nvidia.com/cuda-11-8-0-download-archive

2. Seleccionar:
   - Operating System: Windows
   - Architecture: x86_64
   - Version: 10 o 11
   - Installer Type: exe (local)

3. Ejecutar instalador:
   - Seleccionar "Express Installation"
   - Esperar completar instalacion (puede tomar 15-30 minutos)

4. Verificar instalacion:

```powershell
nvcc --version
# Debe mostrar: Cuda compilation tools, release 11.8
```

### Instalar cuDNN

1. Crear cuenta en NVIDIA Developer (gratuita)
2. Descargar cuDNN 8.6 para CUDA 11.x desde:
   https://developer.nvidia.com/cudnn

3. Extraer archivo ZIP descargado

4. Copiar archivos a directorio de CUDA:

```powershell
# Copiar archivos de cuDNN a CUDA
# Ubicacion por defecto de CUDA: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

# Copiar bin\cudnn*.dll a CUDA\v11.8\bin
# Copiar include\cudnn*.h a CUDA\v11.8\include
# Copiar lib\x64\cudnn*.lib a CUDA\v11.8\lib\x64
```

5. Agregar al PATH del sistema:

```powershell
# Agregar a las variables de entorno:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
```

### Verificar Instalacion CUDA/cuDNN

Crear archivo `test_cuda.py`:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Ejecutar:

```powershell
python test_cuda.py
```

---

## Instalacion de Dependencias

### Crear Entorno Virtual

```powershell
# Navegar al directorio del proyecto
cd C:\Users\TuUsuario\Documents\GitHub\PDI

# Crear entorno virtual
python -m venv venv_ultra

# Activar entorno virtual
.\venv_ultra\Scripts\Activate.ps1

# Si aparece error de politicas de ejecucion:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Actualizar pip y setuptools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### Instalar PyTorch con CUDA

```powershell
# Para CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

Verificar:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
# Debe mostrar: True
```

### Instalar TensorFlow con GPU

```powershell
pip install tensorflow[and-cuda]==2.15.0
```

Verificar:

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Debe mostrar lista de GPUs disponibles
```

### Instalar Dependencias Restantes

```powershell
pip install -r requirements_windows_ultra.txt
```

Este proceso puede tomar 15-30 minutos dependiendo de la velocidad de internet.

### Verificar Instalaciones Criticas

```powershell
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import customtkinter; print('CustomTkinter: OK')"
python -c "from ultralytics import YOLO; print('YOLO: OK')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## Descarga y Configuracion de Modelos

### Estructura de Directorios

```powershell
# Crear estructura de directorios para modelos
New-Item -ItemType Directory -Path "model/ultra" -Force
New-Item -ItemType Directory -Path "data/snapshots_ultra" -Force
New-Item -ItemType Directory -Path "data/exports_ultra" -Force
New-Item -ItemType Directory -Path "data/logs_ultra" -Force
New-Item -ItemType Directory -Path "data/cache" -Force
```

### Descargar Modelo YOLOv8

El sistema descargara automaticamente el modelo al primer uso, pero puede pre-descargarse:

```powershell
python -c "from ultralytics import YOLO; model = YOLO('yolov8x.pt')"
```

El modelo se descargara a `C:\Users\TuUsuario\.cache\ultralytics\`

Copiar a directorio del proyecto:

```powershell
Copy-Item "$env:USERPROFILE\.cache\ultralytics\yolov8x.pt" -Destination "model/ultra/yolov8x.pt"
```

### Crear Labels para Modelos

Crear archivo `model/ultra/efficientnet_labels.json`:

```json
{
  "0": "Perro",
  "1": "Gato",
  "2": "Pajaro",
  "3": "Caballo",
  "4": "Oveja"
}
```

Este archivo sera expandido durante el entrenamiento.

---

## Configuracion de la Base de Datos

La base de datos SQLite se crea automaticamente al ejecutar la aplicacion por primera vez.

Para verificar la estructura:

```powershell
python -c "from pokedex_ultra_windows import UltraPokedexDatabase; db = UltraPokedexDatabase(); print('Database initialized')"
```

Verificar archivo creado:

```powershell
Test-Path "data/pokedex_ultra.db"
# Debe mostrar: True
```

---

## Ejecucion y Verificacion

### Verificacion Pre-Ejecucion

Ejecutar script de verificacion:

```powershell
python -c @"
import sys
import importlib

required_packages = [
    'cv2', 'tensorflow', 'torch', 'customtkinter',
    'numpy', 'pandas', 'PIL', 'ultralytics'
]

print('Verificando paquetes requeridos...')
for package in required_packages:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, '__version__', 'N/A')
        print(f'[OK] {package}: {version}')
    except ImportError as e:
        print(f'[ERROR] {package}: No instalado')
        sys.exit(1)

print('\nTodos los paquetes estan instalados correctamente')
"@
```

### Ejecutar Aplicacion

```powershell
# Asegurar que el entorno virtual esta activado
.\venv_ultra\Scripts\Activate.ps1

# Ejecutar aplicacion
python pokedex_ultra_windows.py
```

### Verificacion de Funcionalidad

Al ejecutar, verificar:

1. La ventana principal se abre en resolucion 1920x1080
2. El video de la camara se muestra en tiempo real
3. El panel lateral muestra metricas del sistema
4. Las metricas de FPS se actualizan en tiempo real
5. La GPU aparece en el monitor de recursos

---

## Optimizaciones de Rendimiento

### Optimizaciones de Windows

#### Deshabilitar Windows Defender para Directorio del Proyecto

```powershell
# Ejecutar como Administrador
Add-MpPreference -ExclusionPath "C:\Users\TuUsuario\Documents\GitHub\PDI"
```

#### Configurar Plan de Energia de Alto Rendimiento

```powershell
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
```

#### Desactivar Game Bar y DVR

```powershell
# En PowerShell como Administrador:
Set-ItemProperty -Path "HKCU:\System\GameConfigStore" -Name "GameDVR_Enabled" -Value 0
Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\GameDVR" -Name "AppCaptureEnabled" -Value 0
```

### Optimizaciones de GPU

Crear archivo `optimize_gpu.py`:

```python
import torch
import tensorflow as tf

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
print("GPU optimizations applied")
```

### Configuracion de Variables de Entorno

Crear archivo `setup_env.ps1`:

```powershell
# Optimizaciones de TensorFlow
$env:TF_GPU_THREAD_MODE = "gpu_private"
$env:TF_GPU_THREAD_COUNT = "2"
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"

# Optimizaciones de PyTorch
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

# Optimizaciones de OpenCV
$env:OPENCV_VIDEOIO_PRIORITY_MSMF = "0"
$env:OPENCV_VIDEOIO_PRIORITY_DSHOW = "1"

# Optimizaciones generales
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"

Write-Host "Environment variables configured for optimal performance"
```

Ejecutar antes de la aplicacion:

```powershell
.\setup_env.ps1
python pokedex_ultra_windows.py
```

### Monitore
o de Recursos

Crear script de monitoreo:

```python
import psutil
import GPUtil
import time

def monitor_resources():
    while True:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"CPU: {cpu}% | RAM: {ram}% | GPU: {gpu.load*100:.1f}% | VRAM: {gpu.memoryUtil*100:.1f}%")
        else:
            print(f"CPU: {cpu}% | RAM: {ram}%")
            
        time.sleep(2)

if __name__ == "__main__":
    monitor_resources()
```

Ejecutar en terminal separada:

```powershell
python monitor_resources.py
```

---

## Resolucion de Problemas

### Error: DLL load failed while importing

**Problema**: No se pueden cargar bibliotecas CUDA/cuDNN

**Solucion**:

```powershell
# Verificar PATH del sistema incluye:
$env:PATH -split ';' | Select-String -Pattern "CUDA"

# Debe mostrar:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp

# Si no aparecen, agregar manualmente
```

### Error: CUDA out of memory

**Problema**: GPU sin memoria suficiente

**Soluciones**:

1. Reducir batch size en configuracion
2. Usar modelo mas pequeño (yolov8m en lugar de yolov8x)
3. Configurar memory growth:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Error: Camera not found

**Problema**: No se detecta camara web

**Soluciones**:

1. Verificar permisos de camara en Windows:
   - Configuracion > Privacidad > Camara
   - Activar "Permitir que las aplicaciones accedan a la camara"

2. Verificar indice de camara:

```python
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
```

3. Actualizar drivers de camara desde Device Manager

### Error: CustomTkinter window not responding

**Problema**: Interfaz congelada

**Soluciones**:

1. Verificar que los threads no bloquean el UI thread
2. Reducir FPS target si el sistema no puede mantener el ritmo
3. Verificar que no hay bucles infinitos en callbacks

### Performance bajo

**Problema**: FPS muy bajo o predicciones lentas

**Diagnostico**:

```powershell
# Verificar uso de GPU
nvidia-smi -l 1

# Debe mostrar utilizacion de GPU durante ejecucion
```

**Soluciones**:

1. Reducir resolucion de video
2. Aumentar intervalo de prediccion
3. Usar modelos mas ligeros
4. Verificar que TensorFlow/PyTorch usan GPU:

```python
import tensorflow as tf
import torch

print(f"TensorFlow GPU: {tf.test.is_gpu_available()}")
print(f"PyTorch GPU: {torch.cuda.is_available()}")
```

---

## Creacion de Acceso Directo

Crear archivo `Pokedex_Ultra.bat`:

```batch
@echo off
cd /d "C:\Users\TuUsuario\Documents\GitHub\PDI"
call venv_ultra\Scripts\activate.bat
python pokedex_ultra_windows.py
pause
```

Crear acceso directo del .bat en el escritorio con icono personalizado.

---

## Actualizaciones y Mantenimiento

### Actualizar Dependencias

```powershell
.\venv_ultra\Scripts\Activate.ps1
pip install --upgrade -r requirements_windows_ultra.txt
```

### Limpieza de Cache

```powershell
# Limpiar cache de predicciones
Remove-Item -Path "data/cache/*" -Force -Recurse

# Limpiar logs antiguos
Get-ChildItem "data/logs_ultra/*.log" | Where-Object {$_.LastWriteTime -lt (Get-Date).AddDays(-30)} | Remove-Item
```

### Backup de Base de Datos

```powershell
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item "data/pokedex_ultra.db" -Destination "data/backups/pokedex_ultra_$timestamp.db"
```

---

## Rendimiento Esperado

### Sistema Minimo (GTX 1060):

- FPS Video: 30-40
- FPS Prediccion: 5-8
- Tiempo de prediccion: 150-200ms
- Uso de VRAM: 3-4 GB

### Sistema Recomendado (RTX 3060):

- FPS Video: 60+
- FPS Prediccion: 10-15
- Tiempo de prediccion: 80-120ms
- Uso de VRAM: 4-6 GB

### Sistema Alto Rendimiento (RTX 4070+):

- FPS Video: 60+ (limitado por camara)
- FPS Prediccion: 15-20
- Tiempo de prediccion: 50-80ms
- Uso de VRAM: 5-8 GB

---

## Soporte y Documentacion Adicional

Para problemas adicionales, consultar:

- Logs de aplicacion en `data/logs_ultra/ultra_pokedex.log`
- Documentacion de TensorFlow: https://www.tensorflow.org/install/gpu
- Documentacion de PyTorch: https://pytorch.org/get-started/locally/
- Documentacion de CUDA: https://docs.nvidia.com/cuda/

---

La instalacion esta completa. El sistema ahora esta configurado para maxima capacidad de procesamiento con aceleracion GPU y modelos de IA de ultima generacion.
