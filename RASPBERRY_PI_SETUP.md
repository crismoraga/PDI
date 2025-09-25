# Guía de Instalación Completa - Pokédex Animal en Raspberry Pi

Esta guía te llevará paso a paso para instalar, configurar y ejecutar la Pokédx Animal en una Raspberry Pi con Raspberry Pi OS (Raspbian).

## Tabla de Contenidos

1. [Requisitos del Hardware](#requisitos-del-hardware)
2. [Preparación del Sistema](#preparación-del-sistema)
3. [Configuración de la Cámara](#configuración-de-la-cámara)
4. [Instalación de Dependencias del Sistema](#instalación-de-dependencias-del-sistema)
5. [Configuración del Entorno Python](#configuración-del-entorno-python)
6. [Instalación de OpenCV](#instalación-de-opencv)
7. [Instalación de TensorFlow Lite](#instalación-de-tensorflow-lite)
8. [Configuración del Proyecto](#configuración-del-proyecto)
9. [Optimizaciones de Rendimiento](#optimizaciones-de-rendimiento)
10. [Configuración Opcional Edge TPU](#configuración-opcional-edge-tpu)
11. [Ejecución y Pruebas](#ejecución-y-pruebas)
12. [Resolución de Problemas](#resolución-de-problemas)

## Requisitos del Hardware

### Componentes Necesarios
- **Raspberry Pi 4** (4GB RAM mínimo, 8GB recomendado)
- **Tarjeta microSD** 32GB Clase 10 o superior
- **Cámara Raspberry Pi** (V2.1 o superior) o **Webcam USB compatible**
- **Fuente de alimentación** oficial de 3A
- **Disipadores de calor** (recomendado)
- **Ventilador** (opcional pero recomendado)

### Versiones Compatibles
- Raspberry Pi 4 (recomendado)
- Raspberry Pi 3B+ (funcional, rendimiento reducido)
- Raspberry Pi Zero 2W (mínimo, rendimiento limitado)

## Preparación del Sistema

### 1. Instalación del Sistema Operativo

```bash
# Descargar Raspberry Pi Imager
# Flashear Raspberry Pi OS (64-bit) en la tarjeta SD
# Configurar WiFi y SSH durante el flasheo (recomendado)
```

### 2. Primera Configuración

```bash
# Actualizar el sistema
sudo apt update && sudo apt upgrade -y

# Configurar localización (opcional)
sudo raspi-config
# Elegir: Localisation Options > Change Locale > es_ES.UTF-8
# Elegir: Localisation Options > Change Timezone

# Reiniciar para aplicar cambios
sudo reboot
```

### 3. Configuración de Memoria

```bash
# Aumentar la memoria GPU para mejor rendimiento de cámara
sudo raspi-config
# Elegir: Advanced Options > Memory Split
# Configurar: 128 MB para GPU

# Habilitar memoria swap adicional
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Cambiar: CONF_SWAPSIZE=1024 (o 2048 si tienes 8GB RAM)
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## Configuración de la Cámara

### Opción A: Cámara Raspberry Pi (Recomendado)

```bash
# Habilitar la cámara
sudo raspi-config
# Elegir: Interface Options > Camera > Enable

# Reiniciar
sudo reboot

# Verificar que la cámara funcione
libcamera-hello --list-cameras
libcamera-hello -t 5000

# Instalar herramientas adicionales
sudo apt install -y libcamera-apps libcamera-dev
```

### Opción B: Webcam USB

```bash
# Instalar herramientas USB
sudo apt install -y fswebcam v4l-utils

# Verificar dispositivos conectados
lsusb
v4l2-ctl --list-devices

# Probar la webcam
fswebcam -r 640x480 test.jpg
```

## Instalación de Dependencias del Sistema

### Dependencias Básicas

```bash
# Herramientas de desarrollo
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    unzip

# Bibliotecas de imagen y video
sudo apt install -y \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module

# Bibliotecas matemáticas y optimización
sudo apt install -y \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    gfortran
```

### Dependencias Python

```bash
# Python y herramientas
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel

# Bibliotecas Python del sistema (aceleran la instalación)
sudo apt install -y \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-pil \
    python3-requests
```

## Configuración del Entorno Python

### 1. Crear Entorno Virtual

```bash
# Navegar al directorio home
cd ~

# Clonar o descargar el proyecto
git clone <url-del-repositorio> PDI
# O copiar los archivos del proyecto a ~/PDI

cd PDI

# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Actualizar herramientas
pip install --upgrade pip setuptools wheel
```

### 2. Configurar Variables de Entorno

```bash
# Crear archivo de configuración
nano ~/.bashrc

# Agregar al final del archivo:
export PYTHONPATH="${PYTHONPATH}:$HOME/PDI"
export PDI_HOME="$HOME/PDI"

# Variables para optimización
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Aplicar cambios
source ~/.bashrc
```

## Instalación de OpenCV

### Opción A: Instalación Rápida (Recomendada)

```bash
# Activar entorno virtual
cd ~/PDI
source venv/bin/activate

# Instalar OpenCV precompilado
pip install opencv-python==4.8.1.78

# Verificar instalación
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

### Opción B: Compilación desde Fuente (Avanzado)

```bash
# Solo si necesitas optimizaciones específicas o la precompilada falla
# ADVERTENCIA: Este proceso toma 2-4 horas

# Dependencias adicionales para compilación
sudo apt install -y \
    cmake-curses-gui \
    libeigen3-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    yasm \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libv4l-dev \
    libxine2-dev

# Descargar OpenCV
cd /tmp
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.1.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.8.1.zip

unzip opencv.zip
unzip opencv_contrib.zip

# Compilar OpenCV (proceso largo)
cd opencv-4.8.1
mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-4.8.1/modules \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D CMAKE_SHARED_LINKER_FLAGS=-latomic \
    -D BUILD_EXAMPLES=OFF ..

make -j4
sudo make install
sudo ldconfig
```

## Instalación de TensorFlow Lite

### 1. Instalar TensorFlow Lite Runtime

```bash
# Activar entorno virtual
cd ~/PDI
source venv/bin/activate

# Instalar TensorFlow Lite Runtime (más ligero que TensorFlow completo)
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.14.0-cp39-cp39-linux_aarch64.whl

# Si el enlace anterior no funciona, usar pip:
pip install tflite-runtime

# Verificar instalación
python -c "import tflite_runtime.interpreter as tflite; print('TFLite Runtime instalado correctamente')"
```

### 2. Instalar Dependencias ML Adicionales

```bash
# Bibliotecas de ciencia de datos
pip install \
    numpy==1.21.6 \
    scipy==1.7.3 \
    matplotlib==3.5.3 \
    pillow==9.5.0 \
    scikit-learn==1.1.3 \
    pandas==1.5.3

# Bibliotecas web y utilidades
pip install \
    requests==2.28.2 \
    beautifulsoup4==4.12.2 \
    wikipedia==1.4.0
```

## Configuración del Proyecto

### 1. Instalar Dependencias del Proyecto

```bash
cd ~/PDI
source venv/bin/activate

# Instalar desde requirements.txt
pip install -r requirements.txt

# Si hay errores, instalar manualmente:
pip install \
    opencv-python==4.8.1.78 \
    tensorflow==2.13.0 \
    numpy==1.21.6 \
    pillow==9.5.0 \
    requests==2.28.2 \
    beautifulsoup4==4.12.2 \
    wikipedia==1.4.0 \
    scikit-learn==1.1.3 \
    matplotlib==3.5.3 \
    pandas==1.5.3
```

### 2. Configurar Estructura de Directorios

```bash
# Crear directorios necesarios
mkdir -p ~/PDI/data/snapshots
mkdir -p ~/PDI/data/exports
mkdir -p ~/PDI/data/models
mkdir -p ~/PDI/logs

# Configurar permisos
chmod 755 ~/PDI/data
chmod 755 ~/PDI/data/snapshots
chmod 755 ~/PDI/data/exports
```

### 3. Descargar Modelos (Opcional)

```bash
cd ~/PDI
source venv/bin/activate

# Ejecutar script de descarga de modelos
python scripts/download_tflite_model.py

# Verificar que los modelos se descargaron
ls -la data/models/
```

### 4. Configurar Variables de Entorno del Proyecto

```bash
# Crear archivo de configuración
nano ~/PDI/.env

# Contenido del archivo:
CAMERA_RESOLUTION_WIDTH=640
CAMERA_RESOLUTION_HEIGHT=480
CAMERA_FPS=15
MODEL_PATH=data/models/mobilenet_v2_animal.tflite
LABELS_PATH=data/models/labels.txt
DATABASE_PATH=data/pokedex.db
LOG_LEVEL=INFO
USE_EDGE_TPU=false
MAX_DETECTION_CONFIDENCE=0.7
```

## Optimizaciones de Rendimiento

### 1. Configuración de CPU

```bash
# Configurar CPU governor para rendimiento
sudo nano /etc/rc.local

# Agregar antes de "exit 0":
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Configurar límites de temperatura
sudo nano /boot/firmware/config.txt

# Agregar:
arm_freq=1800
over_voltage=6
temp_limit=80
```

### 2. Optimizaciones de Memoria

```bash
# Configurar límites de memoria para Python
nano ~/PDI/config.json

# Contenido:
{
    "camera": {
        "resolution": [640, 480],
        "fps": 15,
        "buffer_size": 2
    },
    "processing": {
        "max_threads": 2,
        "batch_size": 1,
        "image_resize": [224, 224]
    },
    "performance": {
        "enable_gpu_acceleration": true,
        "memory_limit_mb": 512,
        "cache_size": 100
    }
}
```

### 3. Configurar Inicio Automático

```bash
# Crear servicio systemd
sudo nano /etc/systemd/system/pokedex-animal.service

# Contenido:
[Unit]
Description=Pokedex Animal Recognition
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/PDI
Environment=DISPLAY=:0.0
ExecStart=/home/pi/PDI/venv/bin/python /home/pi/PDI/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Habilitar servicio
sudo systemctl daemon-reload
sudo systemctl enable pokedex-animal.service
sudo systemctl start pokedex-animal.service
```

## Configuración Opcional Edge TPU

### 1. Instalar Edge TPU Runtime

```bash
# Agregar repositorio de Google
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Actualizar repositorios
sudo apt update

# Instalar Edge TPU runtime
sudo apt install -y libedgetpu1-std

# Para máximo rendimiento (puede sobrecalentar):
# sudo apt install -y libedgetpu1-max
```

### 2. Instalar PyCoral

```bash
cd ~/PDI
source venv/bin/activate

# Instalar PyCoral
pip install pycoral

# Verificar instalación
python -c "from pycoral.utils import edgetpu; print('Edge TPU disponible')"
```

### 3. Convertir Modelo para Edge TPU

```bash
# Si tienes un modelo personalizado, convertirlo:
# Esto requiere Edge TPU Compiler (solo en x86-64)
# Para Raspberry Pi, usar modelos precompilados

# Descargar modelo precompilado de ejemplo
wget https://github.com/google-coral/test_data/raw/master/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite -O data/models/animal_edgetpu.tflite
```

## Ejecución y Pruebas

### 1. Verificación del Sistema

```bash
cd ~/PDI
source venv/bin/activate

# Ejecutar verificación completa
python final_check.py

# El script debe mostrar todas las pruebas en verde:
# ✅ OpenCV: PASÓ
# ✅ TensorFlow Lite: PASÓ  
# ✅ Cámara: PASÓ
# ✅ Modelo ML: PASÓ
# ✅ Base de datos: PASÓ
```

### 2. Ejecutar Demo Sin Cámara

```bash
# Probar funcionalidad básica
python demo.py

# Debe mostrar la interfaz y permitir cargar imágenes
```

### 3. Ejecutar Aplicación Principal

```bash
# Ejecutar aplicación completa
python main.py

# La aplicación debe:
# - Abrir interfaz gráfica
# - Detectar cámara automáticamente
# - Mostrar feed de video en tiempo real
# - Permitir capturar y analizar animales
```

### 4. Pruebas de Rendimiento

```bash
# Monitorear rendimiento del sistema
htop

# En otra terminal, ejecutar test de estrés:
python -c "
import cv2
import time
cap = cv2.VideoCapture(0)
start = time.time()
frames = 0
while frames < 100:
    ret, frame = cap.read()
    if ret:
        frames += 1
end = time.time()
fps = frames / (end - start)
print(f'FPS promedio: {fps:.2f}')
cap.release()
"
```

## Resolución de Problemas

### Problema: Error "No module named 'cv2'"

```bash
# Solución:
cd ~/PDI
source venv/bin/activate
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

### Problema: "Camera not found"

```bash
# Verificar cámara:
lsusb  # Para webcam USB
vcgencmd get_camera  # Para cámara Pi

# Si es cámara Pi:
sudo raspi-config
# Interface Options > Camera > Enable

# Si es webcam USB:
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

### Problema: "Out of memory"

```bash
# Aumentar swap:
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048
sudo dphys-swapfile swapoff
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Reducir resolución en config.json:
{
    "camera": {
        "resolution": [320, 240],
        "fps": 10
    }
}
```

### Problema: Baja Performance

```bash
# Optimizar CPU:
sudo nano /boot/firmware/config.txt
# Agregar: arm_freq=1800

# Usar TensorFlow Lite en lugar de TensorFlow completo
pip uninstall tensorflow
pip install tflite-runtime

# Configurar hilos:
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

### Problema: Error de TensorFlow Lite

```bash
# Reinstalar TFLite Runtime:
pip uninstall tflite-runtime
pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

# O usar wheel específico:
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.14.0-cp39-cp39-linux_aarch64.whl
pip install tflite_runtime-2.14.0-cp39-cp39-linux_aarch64.whl
```

### Logs de Depuración

```bash
# Habilitar logs detallados:
export TF_CPP_MIN_LOG_LEVEL=0

# Ver logs del sistema:
journalctl -u pokedex-animal.service -f

# Ver logs de la aplicación:
tail -f ~/PDI/logs/pokedex.log
```

## Comandos de Mantenimiento

### Actualizar el Sistema

```bash
# Actualizar Raspberry Pi OS
sudo apt update && sudo apt upgrade -y

# Actualizar Python packages
cd ~/PDI
source venv/bin/activate
pip list --outdated
pip install --upgrade pip setuptools wheel
```

### Respaldo de Datos

```bash
# Crear respaldo de la base de datos Pokédx
cp ~/PDI/data/pokedx.db ~/PDI/backups/pokedx_$(date +%Y%m%d).db

# Crear respaldo completo
tar -czf ~/pokedx_backup_$(date +%Y%m%d).tar.gz ~/PDI/data
```

### Monitoreo del Sistema

```bash
# Temperatura de CPU
vcgencmd measure_temp

# Uso de memoria
free -h

# Espacio en disco
df -h

# Procesos de la aplicación
ps aux | grep python
```

## Configuración de Red (Opcional)

### Acceso Remoto

```bash
# Habilitar SSH
sudo systemctl enable ssh
sudo systemctl start ssh

# Configurar VNC para acceso gráfico remoto
sudo raspi-config
# Interface Options > VNC > Enable

# Instalar VNC Server
sudo apt install realvnc-vnc-server realvnc-vnc-viewer
```

### Configurar Red WiFi

```bash
# Configurar WiFi manualmente
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf

# Agregar:
network={
    ssid="TuRedWiFi"
    psk="TuContraseña"
}

# Reiniciar WiFi
sudo wpa_cli -i wlan0 reconfigure
```

Con esta guía completa, deberías poder instalar y ejecutar exitosamente la Pokédx Animal en tu Raspberry Pi. El sistema está optimizado para funcionar eficientemente en hardware limitado mientras mantiene todas las funcionalidades avanzadas.

Para soporte adicional o problemas específicos, revisa la documentación técnica en `TECHNICAL_DOCS.md` o ejecuta el script de verificación `final_check.py`.