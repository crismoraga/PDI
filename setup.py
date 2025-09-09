#!/usr/bin/env python3
"""
Script de configuración y prueba del entorno para Pokedex Animal
"""

import sys
import os
import subprocess
import importlib
import platform

def check_python_version():
    """Verificar versión de Python"""
    print("🐍 Verificando versión de Python...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} no es compatible")
        print("   Se requiere Python 3.8 o superior")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True

def install_requirements():
    """Instalar dependencias desde requirements.txt"""
    print("\n📦 Instalando dependencias...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Dependencias instaladas correctamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al instalar dependencias: {e}")
        print(f"   Salida: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False

def check_dependencies():
    """Verificar que las dependencias están instaladas"""
    print("\n🔍 Verificando dependencias...")
    
    dependencies = [
        ('cv2', 'opencv-python'),
        ('tensorflow', 'tensorflow'),
        ('numpy', 'numpy'),
        ('requests', 'requests'),
        ('PIL', 'Pillow'),
        ('tkinter', 'tkinter (built-in)'),
        ('wikipedia', 'wikipedia'),
        ('bs4', 'beautifulsoup4')
    ]
    
    missing = []
    
    for module, package in dependencies:
        try:
            importlib.import_module(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NO ENCONTRADO")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_camera():
    """Verificar acceso a la cámara"""
    print("\n📹 Verificando acceso a la cámara...")
    
    try:
        import cv2
        
        # Intentar abrir cámara
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ No se puede acceder a la cámara")
            print("   Verifica que tu cámara esté conectada y no esté siendo usada por otra aplicación")
            return False
        
        # Intentar leer un frame
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print(f"✅ Cámara funcional - Resolución: {frame.shape[1]}x{frame.shape[0]}")
            return True
        else:
            print("❌ Error al leer de la cámara")
            return False
            
    except Exception as e:
        print(f"❌ Error al verificar cámara: {e}")
        return False

def check_tensorflow():
    """Verificar instalación de TensorFlow"""
    print("\n🧠 Verificando TensorFlow...")
    
    try:
        import tensorflow as tf
        
        print(f"✅ TensorFlow {tf.__version__}")
        
        # Verificar si hay GPU disponible
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU disponible: {len(gpus)} dispositivo(s)")
        else:
            print("💻 Usando CPU (normal para proyectos educativos)")
            
        # Probar carga de modelo
        try:
            from tensorflow.keras.applications import MobileNetV2
            model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
            print("✅ MobileNetV2 cargado correctamente")
            return True
        except Exception as e:
            print(f"⚠️ Advertencia al cargar MobileNetV2: {e}")
            return True
            
    except Exception as e:
        print(f"❌ Error con TensorFlow: {e}")
        return False

def test_modules():
    """Probar módulos individuales del proyecto"""
    print("\n🧪 Probando módulos del proyecto...")
    
    modules_to_test = [
        ('utils.camera', 'Módulo de cámara'),
        ('utils.image_processing', 'Módulo de procesamiento'),
        ('utils.api', 'Módulo de API'),
        ('model.animal_classifier', 'Módulo de clasificación')
    ]
    
    all_passed = True
    
    for module_name, description in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"✅ {description}")
        except Exception as e:
            print(f"❌ {description}: {e}")
            all_passed = False
    
    return all_passed

def create_directories():
    """Crear directorios necesarios"""
    print("\n📁 Verificando estructura de directorios...")
    
    directories = [
        'data',
        'model',
        'utils',
        'logs'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Creado: {directory}/")
        else:
            print(f"✅ Existe: {directory}/")

def run_system_diagnostics():
    """Ejecutar diagnósticos del sistema"""
    print("\n💻 Información del sistema:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Arquitectura: {platform.machine()}")
    print(f"   Procesador: {platform.processor()}")
    print(f"   Python: {platform.python_version()}")

def main():
    """Función principal de configuración"""
    print("=" * 60)
    print("🐾 CONFIGURACIÓN DE POKEDEX ANIMAL")
    print("=" * 60)
    
    success = True
    
    # Diagnósticos del sistema
    run_system_diagnostics()
    
    # Verificar Python
    if not check_python_version():
        success = False
    
    # Crear directorios
    create_directories()
    
    # Instalar dependencias
    if not install_requirements():
        success = False
    
    # Verificar dependencias
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"\n❌ Dependencias faltantes: {', '.join(missing)}")
        success = False
    
    # Verificar cámara
    if not check_camera():
        success = False
    
    # Verificar TensorFlow
    if not check_tensorflow():
        success = False
    
    # Probar módulos
    if not test_modules():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
        print("✅ Todos los componentes están funcionando correctamente")
        print("\n🚀 Para ejecutar la aplicación:")
        print("   python main.py")
    else:
        print("❌ CONFIGURACIÓN INCOMPLETA")
        print("⚠️ Algunos componentes tienen problemas")
        print("   Revisa los errores anteriores y corrígelos")
    print("=" * 60)

if __name__ == "__main__":
    main()
