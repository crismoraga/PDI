# Guía Completa de Instalación - Pokédx Animal en Raspberry Pi

Esta guía proporciona instrucciones paso a paso extremadamente detalladas para instalar, configurar y ejecutar la Pokédx Animal en una Raspberry Pi con Raspberry Pi OS (Raspbian). Optimizado para reconocimiento de animales en tiempo real.

## Tabla de Contenidos

1. [Requisitos del Hardware](#1-requisitos-del-hardware)
2. [Preparación del Sistema Operativo](#2-preparación-del-sistema-operativo)
3. [Configuración Inicial del Sistema](#3-configuración-inicial-del-sistema)
4. [Configuración de la Cámara](#4-configuración-de-la-cámara)
5. [Instalación de Dependencias del Sistema](#5-instalación-de-dependencias-del-sistema)
6. [Configuración del Entorno Python](#6-configuración-del-entorno-python)
7. [Instalación de OpenCV](#7-instalación-de-opencv)
8. [Instalación de TensorFlow Lite](#8-instalación-de-tensorflow-lite)
9. [Configuración del Proyecto Pokédx](#9-configuración-del-proyecto-pokédx)
10. [Optimizaciones de Rendimiento](#10-optimizaciones-de-rendimiento)
11. [Configuración Opcional Edge TPU](#11-configuración-opcional-edge-tpu)
12. [Ejecución y Pruebas](#12-ejecución-y-pruebas)
13. [Resolución de Problemas](#13-resolución-de-problemas)
14. [Mantenimiento y Actualizaciones](#14-mantenimiento-y-actualizaciones)

---

## 1. Requisitos del Hardware

### 1.1 Componentes Requeridos

**Mínimo (Funcional básico):**
- Raspberry Pi 4 Model B (4GB RAM)
- Tarjeta microSD 32GB Clase 10
- Fuente de alimentación USB-C 3A oficial
- Cámara Raspberry Pi V2.1 o Webcam USB compatible

**Recomendado (Óptimo rendimiento):**
- Raspberry Pi 4 Model B (8GB RAM)
- Tarjeta microSD 64GB Clase A2 U3 (Samsung EVO Select)
- Fuente de alimentación USB-C 3A oficial con cable de 1.5m
- Cámara Raspberry Pi HQ o V2.1
- Disipadores de calor de aluminio
- Ventilador activo 5V
- Carcasa con ventilación

**Opcional (Rendimiento máximo):**
- Google Coral USB Accelerator (Edge TPU)
- SSD USB 3.0 como almacenamiento principal
- Hub USB 3.0 alimentado

### 1.2 Verificación de Compatibilidad

**Modelos de Raspberry Pi Soportados:**
- ✅ Raspberry Pi 4 Model B (4GB/8GB) - **RECOMENDADO**
- ✅ Raspberry Pi 400 - **RECOMENDADO**  
- ⚠️ Raspberry Pi 4 Model B (2GB) - Funcional con limitaciones
- ⚠️ Raspberry Pi 3B+ - Rendimiento reducido
- ❌ Raspberry Pi Zero/3B - No recomendado

**Cámaras Compatibles:**
- ✅ Raspberry Pi Camera V2.1 (8MP)
- ✅ Raspberry Pi Camera HQ (12MP)
- ✅ Webcams USB UVC estándar
- ⚠️ Cámaras USB económicas (pueden tener problemas de drivers)

---

## 2. Preparación del Sistema Operativo

### 2.1 Descarga e Instalación de Raspberry Pi OS

#### Método 1: Raspberry Pi Imager (Recomendado)

1. **Descargar Raspberry Pi Imager:**
   ```bash
   # En tu computadora principal, descarga desde:
   # https://www.raspberrypi.org/software/
   ```

2. **Configurar imagen del SO:**
   - Abrir Raspberry Pi Imager
   - Seleccionar "Raspberry Pi OS (64-bit)" - **IMPORTANTE: Usar versión 64-bit**
   - Hacer clic en el ícono de engranaje (configuración avanzada)

3. **Configuración avanzada:**
   ```
   ✅ Habilitar SSH
   ✅ Configurar nombre de usuario: pi
   ✅ Establecer contraseña segura
   ✅ Configurar WiFi (nombre y contraseña)
   ✅ Configurar región: ES (España) o tu país
   ✅ Habilitar SSH con autenticación por contraseña
   ```

4. **Escribir imagen:**
   - Seleccionar tarjeta microSD
   - Escribir imagen (proceso toma 10-20 minutos)
   - Verificar escritura cuando termine

#### Método 2: Descarga Manual

```bash
# Descargar imagen oficial (en computadora principal)
wget https://downloads.raspberrypi.org/raspios_arm64/images/raspios_arm64-2023-05-03/2023-05-03-raspios-bullseye-arm64.img.xz

# Escribir a tarjeta SD (Linux/macOS)
sudo dd bs=4M if=2023-05-03-raspios-bullseye-arm64.img of=/dev/sdX conv=fsync
```

### 2.2 Primer Arranque

1. **Insertar componentes:**
   - Insertar tarjeta microSD en Raspberry Pi
   - Conectar cámara (si es CSI)
   - Conectar HDMI, teclado, mouse
   - Conectar alimentación (último paso)

2. **Configuración inicial:**
   - El sistema arrancará automáticamente
   - Completar asistente de configuración inicial
   - Seleccionar idioma y región
   - Actualizar software cuando se solicite

---

## 3. Configuración Inicial del Sistema

### 3.1 Actualización del Sistema

```bash
# Actualizar lista de paquetes
sudo apt update

# Actualizar todos los paquetes instalados (proceso puede tomar 30-60 minutos)
sudo apt full-upgrade -y

# Limpiar paquetes innecesarios
sudo apt autoremove -y
sudo apt autoclean

# Reiniciar para aplicar cambios del kernel
sudo reboot
```

### 3.2 Configuración de raspi-config

```bash
# Abrir herramienta de configuración
sudo raspi-config
```

**Configuraciones requeridas:**

1. **1 System Options → S5 Boot / Auto Login**
   - Seleccionar "B2 Console Autologin" (opcional, para headless)

2. **3 Interface Options:**
   - **P1 Camera → Yes** (habilitar cámara CSI)
   - **P2 SSH → Yes** (habilitar SSH para acceso remoto)
   - **P3 VNC → Yes** (opcional, para GUI remoto)
   - **P4 SPI → Yes** (requerido para algunos sensores)
   - **P5 I2C → Yes** (requerido para algunos sensores)

3. **4 Performance Options:**
   - **P1 Overclock → Modest** (solo si tienes enfriamiento adecuado)
   - **P2 GPU Memory → 128MB** (para procesamiento de video)

4. **5 Localisation Options:**
   - Configurar según tu ubicación
   - **L1 Locale → es_ES.UTF-8** (para España)
   - **L2 Timezone → Europe/Madrid** (ajustar según ubicación)

5. **6 Advanced Options:**
   - **A1 Expand Filesystem → Yes** (usar toda la tarjeta SD)

```bash
# Aplicar cambios y reiniciar
# Seleccionar "Finish" y "Yes" para reiniciar
sudo reboot
```

### 3.3 Verificación de Configuración

```bash
# Verificar información del sistema
cat /proc/cpuinfo | grep "model name"
cat /proc/meminfo | grep "MemTotal"
df -h

# Verificar que la cámara fue detectada
vcgencmd get_camera

# Resultado esperado: supported=1 detected=1
```

---

## 4. Configuración de la Cámara

### 4.1 Cámara CSI (Raspberry Pi Camera)

#### Verificación de Conexión

```bash
# Verificar detección de cámara
libcamera-hello --list-cameras

# Resultado esperado:
# Available cameras:
# 0 : imx219 [3280x2464] (/base/soc/i2c0mux/i2c@1/imx219@10)
```

#### Prueba de Funcionamiento

```bash
# Captura de foto de prueba (libcamera - nueva interfaz)
libcamera-still -o test_photo.jpg --width 1920 --height 1080

# Captura de video de prueba (10 segundos)
libcamera-vid -t 10000 -o test_video.h264

# Verificar archivos creados
ls -la test_*
```

#### Configuración de Calidad

```bash
# Crear archivo de configuración personalizada
sudo nano /boot/config.txt

# Añadir al final del archivo:
# Configuración optimizada para cámara
gpu_mem=128
camera_auto_detect=1
dtoverlay=vc4-kms-v3d
max_framebuffers=2

# Para Raspberry Pi HQ Camera, añadir también:
# dtoverlay=imx477
```

### 4.2 Cámara USB (Webcam)

#### Verificación de Dispositivos USB

```bash
# Listar dispositivos USB conectados
lsusb

# Verificar dispositivos de video
ls -la /dev/video*

# Instalar utilidades v4l2
sudo apt install v4l-utils -y

# Verificar capacidades de la cámara
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext
```

#### Prueba con OpenCV

```bash
# Instalar OpenCV básico para pruebas
sudo apt install python3-opencv -y

# Crear script de prueba
cat > test_camera.py << 'EOF'
#!/usr/bin/env python3
import cv2

# Probar cámara (0 = primera cámara)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

# Configurar resolución
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Capturar un frame
ret, frame = cap.read()

if ret:
    # Guardar imagen
    cv2.imwrite('camera_test.jpg', frame)
    print(f"Imagen guardada: camera_test.jpg")
    print(f"Resolución: {frame.shape[1]}x{frame.shape[0]}")
else:
    print("Error: No se pudo capturar imagen")

cap.release()
EOF

# Ejecutar prueba
python3 test_camera.py
```

---

## 5. Instalación de Dependencias del Sistema

### 5.1 Bibliotecas de Sistema Esenciales

```bash
# Actualizar repositorios
sudo apt update

# Instalar dependencias de compilación
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    wget \
    curl \
    unzip

# Dependencias para Python
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel
```

### 5.2 Bibliotecas para OpenCV y Procesamiento de Imágenes

```bash
# Bibliotecas de imágenes y video
sudo apt install -y \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev

# Bibliotecas de interfaz gráfica
sudo apt install -y \
    libgtk-3-dev \
    libcanberra-gtk3-dev \
    libqt5gui5 \
    libqt5webkit5-dev \
    libqt5test5

# Bibliotecas matemáticas optimizadas
sudo apt install -y \
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev
```

### 5.3 Bibliotecas Adicionales para Machine Learning

```bash
# Dependencias para TensorFlow Lite
sudo apt install -y \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg8-dev \
    zlib1g-dev

# Herramientas de desarrollo
sudo apt install -y \
    htop \
    tree \
    nano \
    vim \
    screen \
    tmux
```

### 5.4 Verificación de Instalación

```bash
# Verificar versiones instaladas
python3 --version
pip3 --version
gcc --version
cmake --version

# Verificar bibliotecas críticas
pkg-config --exists opencv4 && echo "OpenCV4 disponible" || echo "OpenCV4 no encontrado"
pkg-config --modversion gtk+-3.0
```

---

## 6. Configuración del Entorno Python

### 6.1 Actualización de pip y herramientas

```bash
# Actualizar pip a la última versión
python3 -m pip install --upgrade pip

# Instalar herramientas de entorno virtual
python3 -m pip install --upgrade \
    setuptools \
    wheel \
    virtualenv \
    virtualenvwrapper
```

### 6.2 Creación del Entorno Virtual

```bash
# Navegar al directorio del proyecto
cd ~
git clone https://github.com/tu-usuario/PDI.git  # Reemplazar con tu repositorio
cd PDI

# Crear entorno virtual específico para el proyecto
python3 -m venv venv_pokedex

# Activar entorno virtual
source venv_pokedex/bin/activate

# Verificar activación (debe aparecer (venv_pokedex) al inicio del prompt)
which python
python --version
```

### 6.3 Configuración de Variables de Entorno

```bash
# Crear archivo de configuración de entorno
nano ~/.bashrc

# Añadir al final del archivo:
# Configuración para Pokédx Animal
export POKEDEX_HOME="$HOME/PDI"
export POKEDEX_VENV="$POKEDEX_HOME/venv_pokedex"

# Alias útiles
alias activate_pokedex="source $POKEDEX_VENV/bin/activate"
alias pokedex_home="cd $POKEDEX_HOME"

# Optimizaciones para Raspberry Pi
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Aplicar cambios
source ~/.bashrc
```

### 6.4 Script de Activación Automática

```bash
# Crear script de activación rápida
cat > ~/activate_pokedex.sh << 'EOF'
#!/bin/bash
echo "Activando entorno Pokédx Animal..."
cd ~/PDI
source venv_pokedex/bin/activate
echo "Entorno activo: $(which python)"
echo "Directorio actual: $(pwd)"
echo ""
echo "Comandos disponibles:"
echo "  python pokedex_realtime.py  - Ejecutar Pokédx en tiempo real"
echo "  python demo.py              - Ejecutar modo demo"
echo "  python test_all.py          - Ejecutar pruebas"
echo ""
EOF

chmod +x ~/activate_pokedex.sh

# Uso: ./activate_pokedex.sh
```

---

## 7. Instalación de OpenCV

### 7.1 Método 1: Instalación desde Repositorios (Recomendado)

```bash
# Activar entorno virtual
source venv_pokedex/bin/activate

# Instalar OpenCV con soporte completo
pip install opencv-contrib-python==4.8.1.78

# Instalar dependencias adicionales
pip install opencv-python-headless==4.8.1.78  # Sin GUI (opcional)
```

### 7.2 Verificación de Instalación

```bash
# Crear script de verificación
cat > test_opencv.py << 'EOF'
#!/usr/bin/env python3
import cv2
import numpy as np

print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)

# Verificar backends disponibles
print("Available Video Backends:")
for backend in cv2.videoio_registry.getBackends():
    print(f"  {backend}")

# Crear imagen de prueba
img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.putText(img, 'OpenCV Test', (50, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Guardar imagen
cv2.imwrite('opencv_test.png', img)
print("Imagen de prueba guardada: opencv_test.png")

# Probar captura de cámara
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Cámara funcional - Frame shape: {frame.shape}")
            cv2.imwrite('camera_opencv_test.jpg', frame)
        else:
            print("Error: No se pudo capturar frame")
        cap.release()
    else:
        print("Error: No se pudo abrir cámara")
except Exception as e:
    print(f"Error con cámara: {e}")

print("Verificación de OpenCV completada")
EOF

# Ejecutar verificación
python test_opencv.py
```

### 7.3 Método 2: Compilación desde Fuente (Opcional - Máximo Rendimiento)

**⚠️ Advertencia: Este proceso toma 2-4 horas en Raspberry Pi**

```bash
# Solo si necesitas máximo rendimiento y características específicas
cd ~
mkdir opencv_build && cd opencv_build

# Descargar OpenCV y contrib
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.8.1.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.8.1.zip

unzip opencv.zip
unzip opencv_contrib.zip

# Crear directorio de compilación
cd opencv-4.8.1
mkdir build && cd build

# Configurar compilación optimizada para Raspberry Pi
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib-4.8.1/modules \
      -D ENABLE_NEON=ON \
      -D ENABLE_VFPV3=ON \
      -D BUILD_TESTS=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D CMAKE_SHARED_LINKER_FLAGS=-latomic \
      -D BUILD_EXAMPLES=OFF ..

# Compilar (usar todos los cores disponibles)
make -j$(nproc)

# Instalar
sudo make install
sudo ldconfig
```

---

## 8. Instalación de TensorFlow Lite

### 8.1 Instalación de tflite-runtime

```bash
# Activar entorno virtual
source venv_pokedx/bin/activate

# Instalar TensorFlow Lite Runtime optimizado para Raspberry Pi
pip install tflite-runtime

# Alternativamente, instalar desde wheels precompilados específicos:
# Para Raspberry Pi 4 (ARM64)
# pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
```

### 8.2 Instalación de Dependencias de Machine Learning

```bash
# Instalar NumPy optimizado
pip install numpy==1.24.3

# Instalar SciPy (toma tiempo en compilar)
pip install scipy==1.10.1

# Instalar scikit-learn
pip install scikit-learn==1.3.0

# Instalar Pillow para procesamiento de imágenes
pip install Pillow==10.0.0

# Instalar pandas para manejo de datos
pip install pandas==2.0.3
```

### 8.3 Instalación de Dependencias de la Aplicación

```bash
# Instalar desde requirements.txt del proyecto
pip install -r requirements.txt

# O instalar manualmente:
pip install \
    requests==2.31.0 \
    beautifulsoup4==4.12.2 \
    wikipedia==1.4.0 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    urllib3==2.0.4
```

### 8.4 Verificación de TensorFlow Lite

```bash
# Crear script de verificación
cat > test_tflite.py << 'EOF'
#!/usr/bin/env python3
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
    print("✅ tflite_runtime importado correctamente")
    
    # Crear un modelo dummy para probar
    # (En la práctica, cargarás tu modelo real)
    print("TensorFlow Lite Runtime funcional")
    
except ImportError as e:
    print(f"❌ Error importando tflite_runtime: {e}")
    
    # Fallback a TensorFlow completo
    try:
        import tensorflow as tf
        print("✅ TensorFlow completo disponible como fallback")
        print(f"Versión TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ Ni tflite_runtime ni TensorFlow están disponibles")

print("Verificación de ML completada")
EOF

# Ejecutar verificación
python test_tflite.py
```

---

## 9. Configuración del Proyecto Pokédx

### 9.1 Descarga e Instalación del Código Fuente

```bash
# Si aún no has clonado el repositorio
cd ~
git clone https://github.com/tu-usuario/PDI.git pokedx_animal
cd pokedx_animal

# O si ya existe, actualizar
cd ~/pokedx_animal
git pull origin main

# Activar entorno virtual
source venv_pokedex/bin/activate
```

### 9.2 Configuración de Directorios

```bash
# Crear estructura de directorios necesaria
mkdir -p data/{snapshots,exports,logs,backups}
mkdir -p model
mkdir -p logs

# Establecer permisos correctos
chmod 755 data
chmod 755 data/{snapshots,exports,logs,backups}
chmod 755 model

# Verificar estructura
tree data/ model/
```

### 9.3 Configuración de Base de Datos

```bash
# La base de datos se creará automáticamente al primer uso
# Pero podemos probar la conexión:

python3 -c "
from pokedx.db import PokedxRepository
repo = PokedxRepository('data/pokedx.db')
print('Base de datos inicializada correctamente')
stats = repo.get_statistics()
print(f'Estadísticas: {stats}')
"
```

### 9.4 Descarga de Modelo TensorFlow Lite

```bash
# Usar script incluido para descargar modelo optimizado
python scripts/download_tflite_model.py list

# Descargar modelo recomendado para Raspberry Pi
python scripts/download_tflite_model.py download mobilenet_v2_1.0_224

# Verificar archivos descargados
ls -la model/
```

### 9.5 Configuración Específica para Raspberry Pi

```bash
# Crear archivo de configuración personalizada
cat > config_raspberry_pi.json << 'EOF'
{
  "project": {
    "name": "Pokédx Animal",
    "version": "2.0.0-rpi",
    "platform": "raspberry_pi",
    "optimized_for": "raspberry_pi_4"
  },
  "camera": {
    "default_index": 0,
    "resolution": {
      "width": 640,
      "height": 480
    },
    "fps": 20,
    "buffer_size": 1,
    "fourcc": "MJPG"
  },
  "model": {
    "backend": "tflite",
    "model_path": "model/animal_classifier.tflite",
    "labels_path": "model/labels.txt",
    "confidence_threshold": 0.3,
    "use_edge_tpu": false
  },
  "performance": {
    "video_refresh_ms": 50,
    "prediction_interval_ms": 1500,
    "max_memory_usage": "2GB",
    "enable_threading": true,
    "optimize_for_pi": true
  },
  "ui": {
    "window_size": "1024x768",
    "enable_fullscreen": false,
    "theme": "dark"
  }
}
EOF
```

---

## 10. Optimizaciones de Rendimiento

### 10.1 Optimizaciones del Sistema

```bash
# Configuración de memoria GPU
sudo nano /boot/config.txt

# Añadir/modificar las siguientes líneas:
gpu_mem=128                    # Memoria para GPU (procesamiento video)
gpu_freq=500                   # Frecuencia GPU
arm_freq=1800                  # Frecuencia CPU (solo si tienes enfriamiento)
over_voltage=2                 # Voltaje adicional (solo con enfriamiento)
disable_splash=1               # Deshabilitar splash screen
dtoverlay=vc4-kms-v3d         # Habilitar aceleración GPU
max_framebuffers=2            # Optimización framebuffer
```

### 10.2 Optimizaciones de Red y Storage

```bash
# Optimizar configuración de red
sudo nano /etc/dhcpcd.conf

# Añadir al final:
# interface wlan0
# static ip_address=192.168.1.100/24  # IP fija opcional
# static routers=192.168.1.1
# static domain_name_servers=8.8.8.8

# Optimizar montaje de SD
sudo nano /etc/fstab

# Modificar la línea de la raíz para añadir noatime:
# /dev/mmcblk0p2  /  ext4  defaults,noatime  0  1
```

### 10.3 Optimizaciones de Proceso

```bash
# Crear script de optimización
cat > optimize_pi.sh << 'EOF'
#!/bin/bash

echo "Aplicando optimizaciones para Pokédx Animal..."

# Aumentar prioridad del proceso Python
sudo renice -10 $$

# Configurar variables de entorno para optimización
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0

# Configurar límites de memoria
ulimit -m 1048576  # 1GB límite de memoria

# Optimizar scheduler
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

echo "Optimizaciones aplicadas"
EOF

chmod +x optimize_pi.sh
```

### 10.4 Script de Monitoreo de Rendimiento

```bash
# Crear script de monitoreo
cat > monitor_performance.py << 'EOF'
#!/usr/bin/env python3
import psutil
import time
import os

def monitor_system():
    """Monitorear rendimiento del sistema."""
    print("=" * 50)
    print("MONITOR DE RENDIMIENTO - POKÉDX ANIMAL")
    print("=" * 50)
    
    while True:
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_temp = get_cpu_temp()
        
        # Memoria
        memory = psutil.virtual_memory()
        
        # Información de proceso
        current_process = psutil.Process()
        
        print(f"\r CPU: {cpu_percent:5.1f}% | "
              f"Temp: {cpu_temp:4.1f}°C | "
              f"RAM: {memory.percent:5.1f}% | "
              f"Proc: {current_process.memory_percent():5.1f}%", 
              end="")
        
        time.sleep(2)

def get_cpu_temp():
    """Obtener temperatura del CPU."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
        return temp
    except:
        return 0.0

if __name__ == "__main__":
    try:
        monitor_system()
    except KeyboardInterrupt:
        print("\nMonitoreo detenido")
EOF

chmod +x monitor_performance.py
```

---

## 11. Configuración Opcional Edge TPU

### 11.1 Instalación de Edge TPU Runtime

**Nota: Solo si tienes Google Coral USB Accelerator**

```bash
# Añadir repositorio de Coral
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Actualizar e instalar
sudo apt update
sudo apt install libedgetpu1-std -y

# Instalar Python API
pip install pycoral
```

### 11.2 Verificación de Edge TPU

```bash
# Conectar Coral USB Accelerator y verificar
lsusb | grep "Global Unichip"

# Crear script de verificación
cat > test_edgetpu.py << 'EOF'
#!/usr/bin/env python3
try:
    from pycoral.utils import edgetpu
    
    # Listar dispositivos Edge TPU
    devices = edgetpu.list_edge_tpus()
    
    if devices:
        print(f"✅ Edge TPU encontrados: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"  Dispositivo {i}: {device}")
    else:
        print("❌ No se encontraron dispositivos Edge TPU")
        
except ImportError:
    print("❌ pycoral no está instalado")
except Exception as e:
    print(f"❌ Error: {e}")
EOF

python test_edgetpu.py
```

### 11.3 Conversión de Modelo para Edge TPU

```bash
# Instalar Edge TPU Compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt install edgetpu-compiler -y

# Convertir modelo existente
edgetpu_compiler model/animal_classifier.tflite -o model/

# Verificar modelo compilado
ls -la model/*_edgetpu.tflite
```

---

## 12. Ejecución y Pruebas

### 12.1 Pruebas Individuales de Componentes

```bash
# Activar entorno
source venv_pokedex/bin/activate
cd ~/pokedx_animal

# Probar cámara
python3 -c "
from utils.camera import CameraCapture
camera = CameraCapture()
if camera.start():
    print('✅ Cámara funcional')
    camera.stop()
else:
    print('❌ Error con cámara')
"

# Probar procesamiento de imágenes
python3 -c "
from utils.image_processing import ImageProcessor
processor = ImageProcessor()
print('✅ Procesamiento de imágenes funcional')
"

# Probar clasificador
python3 -c "
try:
    from model.tflite_classifier import TFLiteAnimalClassifier
    classifier = TFLiteAnimalClassifier()
    print('✅ Clasificador TensorFlow Lite funcional')
except:
    from model.animal_classifier import AnimalClassifier
    classifier = AnimalClassifier()
    print('✅ Clasificador Keras funcional')
"

# Probar base de datos
python3 -c "
from pokedx.db import PokedxRepository
repo = PokedxRepository('data/pokedx.db')
stats = repo.get_statistics()
print(f'✅ Base de datos funcional: {stats}')
"
```

### 12.2 Prueba Integral del Sistema

```bash
# Ejecutar suite de pruebas completa
python test_all.py

# El resultado debe mostrar:
# ✅ Procesamiento de Imágenes: PASÓ
# ✅ Clasificador de Animales: PASÓ  
# ✅ Módulo de API: PASÓ
# ✅ Integración Completa: PASÓ
```

### 12.3 Ejecución del Modo Demo

```bash
# Ejecutar modo demo (sin cámara en tiempo real)
python demo.py

# Debería abrir una ventana GUI donde puedes:
# - Cargar imágenes de prueba
# - Probar el clasificador
# - Ver información de animales
```

### 12.4 Ejecución en Tiempo Real

```bash
# Ejecutar aplicación principal con tiempo real
python pokedx_realtime.py

# La aplicación debe:
# - Mostrar video de cámara en tiempo real
# - Detectar y clasificar animales automáticamente
# - Guardar entradas en la base de datos
# - Permitir búsqueda y gestión de entradas
```

### 12.5 Script de Inicio Automático

```bash
# Crear script de inicio
cat > start_pokedx.sh << 'EOF'
#!/bin/bash

echo "Iniciando Pokédx Animal..."

# Navegar al directorio del proyecto
cd ~/pokedx_animal

# Activar entorno virtual
source venv_pokedx/bin/activate

# Aplicar optimizaciones
./optimize_pi.sh

# Verificar cámara
if ! python3 -c "from utils.camera import CameraCapture; c = CameraCapture(); print('OK' if c.start() else 'FAIL'); c.stop()" | grep -q "OK"; then
    echo "Error: Cámara no disponible"
    exit 1
fi

# Iniciar aplicación
echo "Iniciando aplicación en tiempo real..."
python pokedx_realtime.py

EOF

chmod +x start_pokedx.sh

# Uso: ./start_pokedx.sh
```

### 12.6 Servicio Systemd (Inicio Automático en Boot)

```bash
# Crear archivo de servicio
sudo nano /etc/systemd/system/pokedx.service

# Contenido del archivo:
[Unit]
Description=Pokédx Animal Real-time Application
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/pokedx_animal
Environment=PATH=/home/pi/pokedx_animal/venv_pokedx/bin:/usr/bin:/bin
ExecStart=/home/pi/pokedx_animal/venv_pokedx/bin/python /home/pi/pokedx_animal/pokedx_realtime.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Habilitar servicio
sudo systemctl enable pokedx.service

# Iniciar servicio
sudo systemctl start pokedx.service

# Verificar estado
sudo systemctl status pokedx.service
```

---

## 13. Resolución de Problemas

### 13.1 Problemas Comunes con Cámara

**Error: "Cannot open camera"**

```bash
# Verificar permisos
sudo usermod -a -G video pi
sudo reboot

# Verificar que no haya otros procesos usando la cámara
sudo lsof /dev/video0

# Verificar configuración de libcamera
libcamera-hello --list-cameras

# Si usa cámara CSI, verificar conexión física
vcgencmd get_camera
```

**Error: "Framerate muy bajo"**

```bash
# Reducir resolución en config
# Modificar config_raspberry_pi.json:
{
  "camera": {
    "resolution": {"width": 480, "height": 360},
    "fps": 15
  }
}
```

### 13.2 Problemas con TensorFlow Lite

**Error: "tflite_runtime not found"**

```bash
# Reinstalar tflite-runtime
pip uninstall tflite-runtime -y
pip install tflite-runtime

# Verificar arquitectura
uname -m
# Debe mostrar aarch64 para Pi 4 con OS 64-bit
```

**Error: "Model file not found"**

```bash
# Verificar modelo
ls -la model/
python scripts/download_tflite_model.py download mobilenet_v2_1.0_224
```

### 13.3 Problemas de Rendimiento

**Aplicación muy lenta:**

```bash
# Verificar temperatura
vcgencmd measure_temp

# Si > 80°C, mejorar enfriamiento y reducir frecuencia
sudo nano /boot/config.txt
# Comentar líneas de overclock

# Reducir calidad de procesamiento
# Modificar config_raspberry_pi.json:
{
  "performance": {
    "prediction_interval_ms": 2000,
    "video_refresh_ms": 100
  }
}
```

**Error: "Out of memory"**

```bash
# Verificar memoria disponible
free -h

# Aumentar swap
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Reducir buffer de cámara
{
  "camera": {
    "buffer_size": 1
  }
}
```

### 13.4 Problemas de Base de Datos

**Error: "Database locked"**

```bash
# Verificar permisos
ls -la data/pokedx.db
chmod 664 data/pokedx.db

# Verificar procesos que usan la BD
sudo lsof data/pokedx.db
```

### 13.5 Logs de Debugging

```bash
# Habilitar logs detallados
export PYTHONPATH=~/pokedx_animal
export POKEDEX_DEBUG=1

# Ejecutar con logs
python pokedx_realtime.py 2>&1 | tee logs/debug.log

# Ver logs en tiempo real
tail -f logs/debug.log
```

---

## 14. Mantenimiento y Actualizaciones

### 14.1 Actualización del Sistema

```bash
# Script de mantenimiento semanal
cat > maintenance.sh << 'EOF'
#!/bin/bash

echo "Iniciando mantenimiento del sistema..."

# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Limpiar paquetes
sudo apt autoremove -y
sudo apt autoclean

# Actualizar Python packages
source ~/pokedx_animal/venv_pokedx/bin/activate
pip list --outdated
# pip install --upgrade package_name  # Actualizar individualmente

# Verificar espacio en disco
df -h

# Limpiar logs antiguos
find ~/pokedx_animal/data/logs -name "*.log" -mtime +30 -delete

# Backup de base de datos
cp ~/pokedx_animal/data/pokedx.db ~/pokedx_animal/data/backups/pokedx_backup_$(date +%Y%m%d).db

echo "Mantenimiento completado"
EOF

chmod +x maintenance.sh

# Programar ejecución semanal
crontab -e
# Añadir: 0 2 * * 0 /home/pi/maintenance.sh
```

### 14.2 Monitoreo de Salud del Sistema

```bash
# Crear script de monitoreo
cat > health_check.py << 'EOF'
#!/usr/bin/env python3
import psutil
import os
import sqlite3
from datetime import datetime

def health_check():
    """Verificación de salud del sistema."""
    print("VERIFICACIÓN DE SALUD - POKÉDX ANIMAL")
    print("=" * 40)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Sistema
    print("SISTEMA:")
    print(f"  CPU: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"  RAM: {psutil.virtual_memory().percent:.1f}%")
    print(f"  Disco: {psutil.disk_usage('/').percent:.1f}%")
    
    # Temperatura
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read()) / 1000.0
        print(f"  Temperatura: {temp:.1f}°C")
        if temp > 80:
            print("  ⚠️ TEMPERATURA ALTA - Verificar enfriamiento")
    except:
        print("  Temperatura: No disponible")
    
    # Base de datos
    print("\nBASE DE DATOS:")
    try:
        db_path = "data/pokedx.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM entries")
            total_entries = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM entries WHERE captured = 1")
            captured_entries = cursor.fetchone()[0]
            conn.close()
            
            print(f"  Total entradas: {total_entries}")
            print(f"  Capturados: {captured_entries}")
            print(f"  Completado: {(captured_entries/total_entries*100 if total_entries > 0 else 0):.1f}%")
        else:
            print("  ⚠️ Base de datos no encontrada")
    except Exception as e:
        print(f"  ❌ Error accediendo BD: {e}")
    
    # Cámara
    print("\nCÁMARA:")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  ✅ Funcional - Resolución: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("  ⚠️ No se pudo capturar frame")
            cap.release()
        else:
            print("  ❌ No se pudo abrir cámara")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\nVerificación completada")

if __name__ == "__main__":
    os.chdir("/home/pi/pokedx_animal")
    health_check()
EOF

chmod +x health_check.py

# Ejecutar verificación
python health_check.py
```

### 14.3 Backup y Recuperación

```bash
# Script de backup completo
cat > backup.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="$HOME/pokedx_backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="pokedx_backup_$DATE"

echo "Creando backup: $BACKUP_NAME"

# Crear directorio de backup
mkdir -p "$BACKUP_DIR"

# Backup de base de datos
cp ~/pokedx_animal/data/pokedx.db "$BACKUP_DIR/${BACKUP_NAME}_database.db"

# Backup de configuración
cp ~/pokedx_animal/config*.json "$BACKUP_DIR/" 2>/dev/null

# Backup de snapshots (solo últimos 100 archivos)
mkdir -p "$BACKUP_DIR/snapshots"
ls -t ~/pokedx_animal/data/snapshots/*.jpg 2>/dev/null | head -100 | xargs -I {} cp {} "$BACKUP_DIR/snapshots/"

# Crear archivo tar comprimido
cd "$BACKUP_DIR"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}_database.db" config*.json snapshots/

# Limpiar archivos temporales
rm -rf "${BACKUP_NAME}_database.db" config*.json snapshots/

echo "Backup completado: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"

# Limpiar backups antiguos (mantener últimos 10)
ls -t "$BACKUP_DIR"/*.tar.gz | tail -n +11 | xargs -r rm

EOF

chmod +x backup.sh

# Ejecutar backup
./backup.sh
```

---

## Conclusión

Has completado la instalación y configuración completa de la Pokédx Animal en Raspberry Pi. El sistema está ahora optimizado para:

- ✅ Reconocimiento de animales en tiempo real
- ✅ Base de datos completa tipo Pokédx
- ✅ Optimización específica para Raspberry Pi 4
- ✅ Soporte para TensorFlow Lite
- ✅ Interfaz gráfica responsiva
- ✅ Sistema de monitoreo y mantenimiento

### Comandos de Uso Rápido

```bash
# Activar entorno y navegar al proyecto
source ~/pokedx_animal/venv_pokedx/bin/activate && cd ~/pokedx_animal

# Ejecutar aplicación principal
python pokedx_realtime.py

# Ejecutar modo demo
python demo.py

# Verificar salud del sistema
python health_check.py

# Crear backup
./backup.sh
```

### Recursos Adicionales

- **Documentación técnica**: `TECHNICAL_DOCS.md`
- **Logs del sistema**: `data/logs/`
- **Base de datos**: `data/pokedx.db`
- **Configuración**: `config_raspberry_pi.json`

Para soporte adicional, consulta los logs de la aplicación o ejecuta las verificaciones de salud incluidas.

**¡Tu Pokédx Animal está lista para descubrir y catalogar la fauna que te rodea en tiempo real!**