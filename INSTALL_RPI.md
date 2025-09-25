# Guía de Instalación y Optimización en Raspberry Pi

Este documento cubre la instalación paso a paso de Pokedex Animal en Raspberry Pi, incluyendo configuración de cámara, dependencias del sistema, Python, OpenCV, TensorFlow Lite (tflite_runtime) y recomendaciones de rendimiento. También incluye la opción de Edge TPU.

## Hardware soportado

- Raspberry Pi 4/400/5 (recomendado)
- Cámara oficial Raspberry Pi (CSI) o cámara USB compatible UVC
- Opcional: Coral USB Accelerator (Edge TPU)
- Tarjeta microSD clase A2 U3 (recomendado)

## 1) Preparación del sistema

1. Instala Raspberry Pi OS (64-bit recomendado).
2. Conecta la Raspberry a internet, y actualiza el sistema:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

1. Habilita la cámara:

- Para cámaras CSI (libcamera):

```bash
sudo raspi-config
# Interfacing Options -> Camera -> Enable
# Reboot si lo solicita
```

- Verifica la cámara:

```bash
libcamera-hello -t 2000
```

Para cámaras USB UVC, conecta y valida con `v4l2-ctl --list-devices` (instalar con `sudo apt install v4l-utils`).

## 2) Dependencias del sistema

Instala librerías necesarias para OpenCV, NumPy y PIL:

```bash
sudo apt install -y python3-venv python3-dev build-essential \
    libatlas-base-dev libblas-dev liblapack-dev gfortran \
    libjpeg-dev libpng-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev \
    libgtk-3-dev libcanberra-gtk3-module libdc1394-22-dev libv4l-dev \
    libxvidcore-dev libx264-dev libopenexr-dev libtbb2 libtbb-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

## 3) Entorno Python

Crea un entorno virtual en la carpeta del proyecto:

```bash
cd ~/PDI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4) OpenCV y dependencias Python

Instala las dependencias del proyecto excepto TensorFlow completo (en Pi usamos tflite_runtime):

```bash
pip install --no-cache-dir -r requirements.txt
```

Si `opencv-python` falla o es lento, prueba `opencv-python-headless`:

```bash
pip install --no-cache-dir opencv-python-headless
```

Nota: Si usas `headless`, Tkinter/GUI no utilizará las bindings de HighGUI, pero nuestra app usa Tkinter; es compatible.

## 5) Instalar tflite_runtime

Selecciona la rueda (wheel) adecuada para tu arquitectura y versión de Python. Ejemplos (pueden cambiar según releases):

- Para Raspberry Pi OS 64-bit (aarch64), Python 3.11:

```bash
pip install https://github.com/google-coral/pycoral/releases/download/release-fp/tflite_runtime-2.14.0-cp311-cp311-linux_aarch64.whl
```

- Para Raspberry Pi OS 32-bit (armv7l), Python 3.9/3.10, busca la rueda adecuada:

```bash
# Ejemplo genérico; revisa versiones en:
# https://github.com/google-coral/pycoral/releases
# o en repositorios de tflite_runtime
```

Verifica instalación:

```bash
python -c "import tflite_runtime; print('tflite_runtime OK')"
```

Nuestra app detecta automáticamente tflite_runtime y utilizará TFLite; si no está presente, usará el clasificador Keras.

## 6) Edge TPU (opcional)

1. Instala las librerías del Coral USB Accelerator:

```bash
echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install -y libedgetpu1-std
```

1. Instala pycoral si tu modelo lo requiere:

```bash
pip install pycoral
```

1. Usa modelos _edgetpu.tflite y activa `use_edgetpu=True` en el constructor del clasificador TFLite (o ajusta el código para intentar delegados automáticamente).

## 7) Descargar modelo TFLite y labels

Incluimos un script de utilidad:

```bash
python scripts/download_tflite_model.py --model-url <URL_DEL_MODELO_TFLITE> --labels-url <URL_DE_LABELS>
```
Por defecto descarga a `model/animal_classifier.tflite` y `model/labels.txt`.


## 8) Ejecutar la aplicación

Desde el directorio del proyecto con el entorno activado:

```bash
python setup.py  # opcional: verifica entorno y cámara
python main.py
```

Sugerencias de rendimiento:

- Resolución: 640x480 y 15 FPS están configurados por defecto para Pi.
- Evita operaciones pesadas por frame; nuestro pipeline usa preprocesamiento ligero.
- Si el rendimiento no es suficiente, reduce a 320x240.
- Asegúrate de disipación térmica para evitar throttling.

## 9) Solución de problemas

- Cámara no se abre:
  - Verifica con `libcamera-hello` (CSI) o `v4l2-ctl` (USB).
  - Comprueba permisos y que no esté en uso por otro proceso.
- tflite_runtime no carga:
  - Confirma la rueda correcta para tu Python/arquitectura.
  - Prueba `python -c "import platform,sys; print(platform.machine(), sys.version)"` para validar.
- OpenCV lento:
  - Usa `opencv-python-headless` o recompila con NEON/VFPv4 optimizaciones.

## 10) Notas finales

- La app guarda cada análisis en `data/snapshots` y en `data/pokedex.db`.
- La búsqueda y listado están accesibles desde la UI.
- Para una experiencia tipo Pokédex 1:1, en próximos pasos añadiremos vistas detalladas y navegación por entradas con imágenes y datos completos.
