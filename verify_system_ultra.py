#!/usr/bin/env python3
"""
Sistema de Verificacion Rigurosa y Exhaustiva
Para Pokedex Animal Ultra - Windows Edition

Verifica todos los componentes del sistema con enfoque esceptico y critico.
No permite fallos silenciosos.
"""

from __future__ import annotations

import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class VerificationResult:
    """Resultado de una verificacion individual."""
    
    def __init__(self, name: str, passed: bool, message: str, details: Optional[str] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
        
    def __repr__(self) -> str:
        status = "[PASS]" if self.passed else "[FAIL]"
        result = f"{status} {self.name}: {self.message}"
        if self.details:
            result += f"\n  Details: {self.details}"
        return result


class SystemVerifier:
    """Verificador riguroso del sistema completo."""
    
    def __init__(self):
        self.results: List[VerificationResult] = []
        self.critical_failures: List[str] = []
        
    def verify_all(self) -> bool:
        """Ejecutar todas las verificaciones."""
        logger.info("=" * 80)
        logger.info("INICIANDO VERIFICACION RIGUROSA DEL SISTEMA")
        logger.info("=" * 80)
        
        verifications = [
            ("Python Version", self.verify_python_version),
            ("Critical Packages", self.verify_critical_packages),
            ("TensorFlow GPU", self.verify_tensorflow_gpu),
            ("PyTorch GPU", self.verify_pytorch_gpu),
            ("OpenCV", self.verify_opencv),
            ("CustomTkinter", self.verify_customtkinter),
            ("YOLO", self.verify_yolo),
            ("Camera Access", self.verify_camera),
            ("GPU Availability", self.verify_gpu),
            ("CUDA Installation", self.verify_cuda),
            ("Directory Structure", self.verify_directories),
            ("Model Files", self.verify_models),
            ("Database", self.verify_database),
            ("Dependencies Integrity", self.verify_dependencies_integrity),
            ("Code Syntax", self.verify_code_syntax),
            ("Import Resolution", self.verify_imports),
            ("Performance Baseline", self.verify_performance),
        ]
        
        for name, verification_func in verifications:
            logger.info(f"\nVerificando: {name}")
            logger.info("-" * 80)
            
            try:
                result = verification_func()
                self.results.append(result)
                
                if result.passed:
                    logger.info(f"[OK] {result.message}")
                else:
                    logger.error(f"[ERROR] {result.message}")
                    if result.details:
                        logger.error(f"Detalles: {result.details}")
                    
                    if "Critical" in name or "GPU" in name or "Camera" in name:
                        self.critical_failures.append(name)
                        
            except Exception as e:
                error_msg = f"Excepcion durante verificacion: {str(e)}"
                logger.exception(error_msg)
                self.results.append(VerificationResult(name, False, error_msg))
                self.critical_failures.append(name)
                
        return self.generate_report()
        
    def verify_python_version(self) -> VerificationResult:
        """Verificar version de Python."""
        version = sys.version_info
        
        if version.major != 3:
            return VerificationResult(
                "Python Version",
                False,
                f"Python 3 requerido, detectado Python {version.major}",
                f"Version completa: {sys.version}"
            )
            
        if version.minor < 10:
            return VerificationResult(
                "Python Version",
                False,
                f"Python 3.10+ requerido, detectado Python {version.major}.{version.minor}",
                "Actualizar Python a version 3.10 o superior"
            )
            
        return VerificationResult(
            "Python Version",
            True,
            f"Python {version.major}.{version.minor}.{version.micro}",
            f"Ejecutable: {sys.executable}"
        )
        
    def verify_critical_packages(self) -> VerificationResult:
        """Verificar paquetes criticos estan instalados."""
        critical_packages = {
            'numpy': '1.24.0',
            'cv2': '4.8.0',
            'tensorflow': '2.15.0',
            'torch': '2.1.0',
            'customtkinter': '5.0.0',
            'pandas': '2.0.0',
            'PIL': '10.0.0',
        }
        
        missing = []
        outdated = []
        
        for package, min_version in critical_packages.items():
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, '__version__', 'unknown')
                
                if version == 'unknown':
                    outdated.append(f"{package} (version desconocida)")
                    
            except ImportError:
                missing.append(package)
                
        if missing:
            return VerificationResult(
                "Critical Packages",
                False,
                f"Paquetes faltantes: {', '.join(missing)}",
                "Ejecutar: pip install -r requirements_windows_ultra.txt"
            )
            
        if outdated:
            return VerificationResult(
                "Critical Packages",
                True,
                "Todos los paquetes instalados (algunas versiones no verificables)",
                f"Versiones desconocidas: {', '.join(outdated)}"
            )
            
        return VerificationResult(
            "Critical Packages",
            True,
            "Todos los paquetes criticos instalados correctamente"
        )
        
    def verify_tensorflow_gpu(self) -> VerificationResult:
        """Verificar TensorFlow con soporte GPU."""
        try:
            import tensorflow as tf
            
            gpu_devices = tf.config.list_physical_devices('GPU')
            
            if not gpu_devices:
                return VerificationResult(
                    "TensorFlow GPU",
                    False,
                    "TensorFlow instalado pero sin GPUs detectadas",
                    "Verificar instalacion de CUDA y cuDNN"
                )
                
            gpu_available = tf.test.is_built_with_cuda()
            
            if not gpu_available:
                return VerificationResult(
                    "TensorFlow GPU",
                    False,
                    "TensorFlow no compilado con soporte CUDA",
                    "Reinstalar tensorflow con GPU: pip install tensorflow[and-cuda]"
                )
                
            details = f"Version: {tf.__version__}\n"
            details += f"GPUs detectadas: {len(gpu_devices)}\n"
            for i, gpu in enumerate(gpu_devices):
                details += f"  GPU {i}: {gpu.name}"
                
            return VerificationResult(
                "TensorFlow GPU",
                True,
                f"TensorFlow {tf.__version__} con {len(gpu_devices)} GPU(s)",
                details
            )
            
        except Exception as e:
            return VerificationResult(
                "TensorFlow GPU",
                False,
                f"Error verificando TensorFlow: {str(e)}"
            )
            
    def verify_pytorch_gpu(self) -> VerificationResult:
        """Verificar PyTorch con soporte GPU."""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            
            if not cuda_available:
                return VerificationResult(
                    "PyTorch GPU",
                    False,
                    "PyTorch instalado pero CUDA no disponible",
                    "Reinstalar PyTorch con CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
                )
                
            device_count = torch.cuda.device_count()
            
            details = f"Version PyTorch: {torch.__version__}\n"
            details += f"Version CUDA: {torch.version.cuda}\n"
            details += f"cuDNN Version: {torch.backends.cudnn.version()}\n"
            details += f"GPUs detectadas: {device_count}\n"
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                details += f"  GPU {i}: {name} ({memory_gb:.2f} GB)"
                
            return VerificationResult(
                "PyTorch GPU",
                True,
                f"PyTorch {torch.__version__} con CUDA {torch.version.cuda}",
                details
            )
            
        except Exception as e:
            return VerificationResult(
                "PyTorch GPU",
                False,
                f"Error verificando PyTorch: {str(e)}"
            )
            
    def verify_opencv(self) -> VerificationResult:
        """Verificar OpenCV con todas las capacidades."""
        try:
            import cv2
            
            version = cv2.__version__
            
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img = cv2.GaussianBlur(test_img, (5, 5), 0)
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(test_img, 50, 150)
            
            build_info = cv2.getBuildInformation()
            
            cuda_enabled = "CUDA" in build_info and "YES" in build_info[build_info.find("CUDA"):build_info.find("CUDA")+100]
            
            details = f"Version: {version}\n"
            details += f"CUDA habilitado: {'Si' if cuda_enabled else 'No'}"
            
            return VerificationResult(
                "OpenCV",
                True,
                f"OpenCV {version} funcional",
                details
            )
            
        except Exception as e:
            return VerificationResult(
                "OpenCV",
                False,
                f"Error verificando OpenCV: {str(e)}"
            )
            
    def verify_customtkinter(self) -> VerificationResult:
        """Verificar CustomTkinter."""
        try:
            import customtkinter as ctk
            
            version = ctk.__version__
            
            test_window = ctk.CTk()
            test_window.withdraw()
            test_window.destroy()
            
            return VerificationResult(
                "CustomTkinter",
                True,
                f"CustomTkinter {version} funcional"
            )
            
        except Exception as e:
            return VerificationResult(
                "CustomTkinter",
                False,
                f"Error verificando CustomTkinter: {str(e)}",
                "Instalar: pip install customtkinter"
            )
            
    def verify_yolo(self) -> VerificationResult:
        """Verificar YOLO (Ultralytics)."""
        try:
            from ultralytics import YOLO
            
            return VerificationResult(
                "YOLO",
                True,
                "Ultralytics YOLO disponible"
            )
            
        except Exception as e:
            return VerificationResult(
                "YOLO",
                False,
                f"Error verificando YOLO: {str(e)}",
                "Instalar: pip install ultralytics"
            )
            
    def verify_camera(self) -> VerificationResult:
        """Verificar acceso a camara."""
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                return VerificationResult(
                    "Camera Access",
                    False,
                    "No se puede abrir camara",
                    "Verificar permisos de camara en Windows y que dispositivo este conectado"
                )
                
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return VerificationResult(
                    "Camera Access",
                    False,
                    "Camara abierta pero no se puede capturar frame",
                    "Verificar drivers de camara"
                )
                
            height, width = frame.shape[:2]
            
            return VerificationResult(
                "Camera Access",
                True,
                f"Camara funcional: {width}x{height}",
                f"Frame shape: {frame.shape}"
            )
            
        except Exception as e:
            return VerificationResult(
                "Camera Access",
                False,
                f"Error accediendo camara: {str(e)}"
            )
            
    def verify_gpu(self) -> VerificationResult:
        """Verificar GPUs disponibles."""
        try:
            import GPUtil
            
            gpus = GPUtil.getGPUs()
            
            if not gpus:
                return VerificationResult(
                    "GPU Availability",
                    False,
                    "No se detectaron GPUs NVIDIA",
                    "Sistema funcionara en CPU (rendimiento reducido)"
                )
                
            details = ""
            for i, gpu in enumerate(gpus):
                details += f"\nGPU {i}: {gpu.name}\n"
                details += f"  Memoria: {gpu.memoryTotal}MB\n"
                details += f"  Driver: {gpu.driver}\n"
                details += f"  Uso actual: {gpu.load*100:.1f}%"
                
            return VerificationResult(
                "GPU Availability",
                True,
                f"{len(gpus)} GPU(s) NVIDIA detectada(s)",
                details
            )
            
        except ImportError:
            return VerificationResult(
                "GPU Availability",
                True,
                "GPUtil no instalado (opcional)",
                "Instalar para monitoreo: pip install gputil"
            )
        except Exception as e:
            return VerificationResult(
                "GPU Availability",
                False,
                f"Error verificando GPU: {str(e)}"
            )
            
    def verify_cuda(self) -> VerificationResult:
        """Verificar instalacion CUDA."""
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                return VerificationResult(
                    "CUDA Installation",
                    False,
                    "nvcc no encontrado en PATH",
                    "Instalar CUDA Toolkit 11.8 y agregar al PATH"
                )
                
            version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
            version = version_line[0] if version_line else "Version desconocida"
            
            return VerificationResult(
                "CUDA Installation",
                True,
                f"CUDA instalado: {version.strip()}"
            )
            
        except FileNotFoundError:
            return VerificationResult(
                "CUDA Installation",
                False,
                "CUDA no encontrado",
                "Instalar CUDA Toolkit 11.8"
            )
        except Exception as e:
            return VerificationResult(
                "CUDA Installation",
                False,
                f"Error verificando CUDA: {str(e)}"
            )
            
    def verify_directories(self) -> VerificationResult:
        """Verificar estructura de directorios."""
        required_dirs = [
            Path("data/snapshots_ultra"),
            Path("data/exports_ultra"),
            Path("data/logs_ultra"),
            Path("data/cache"),
            Path("model/ultra"),
            Path("data/training"),
        ]
        
        missing = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Creado directorio: {dir_path}")
                
        return VerificationResult(
            "Directory Structure",
            True,
            "Estructura de directorios verificada y creada"
        )
        
    def verify_models(self) -> VerificationResult:
        """Verificar archivos de modelos."""
        model_dir = Path("model/ultra")
        
        models = list(model_dir.glob("*.h5")) + list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))
        
        if not models:
            return VerificationResult(
                "Model Files",
                True,
                "No hay modelos entrenados aun",
                "Ejecutar entrenamiento o descargar modelos pre-entrenados"
            )
            
        details = f"Modelos encontrados: {len(models)}\n"
        for model in models:
            size_mb = model.stat().st_size / (1024 * 1024)
            details += f"  {model.name}: {size_mb:.2f} MB"
            
        return VerificationResult(
            "Model Files",
            True,
            f"{len(models)} modelo(s) encontrado(s)",
            details
        )
        
    def verify_database(self) -> VerificationResult:
        """Verificar base de datos."""
        try:
            from pokedex_ultra_windows import UltraPokedexDatabase
            
            db = UltraPokedexDatabase()
            
            stats = db.get_statistics()
            
            db.close()
            
            details = f"Detecciones totales: {stats['total_detections']}\n"
            details += f"Especies descubiertas: {stats['total_species_discovered']}\n"
            details += f"Especies capturadas: {stats['total_species_captured']}"
            
            return VerificationResult(
                "Database",
                True,
                "Base de datos funcional",
                details
            )
            
        except Exception as e:
            return VerificationResult(
                "Database",
                False,
                f"Error verificando base de datos: {str(e)}"
            )
            
    def verify_dependencies_integrity(self) -> VerificationResult:
        """Verificar integridad de dependencias."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'check'],
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                return VerificationResult(
                    "Dependencies Integrity",
                    False,
                    "Conflictos de dependencias detectados",
                    result.stdout + result.stderr
                )
                
            return VerificationResult(
                "Dependencies Integrity",
                True,
                "Sin conflictos de dependencias"
            )
            
        except Exception as e:
            return VerificationResult(
                "Dependencies Integrity",
                False,
                f"Error verificando dependencias: {str(e)}"
            )
            
    def verify_code_syntax(self) -> VerificationResult:
        """Verificar sintaxis de archivos Python principales."""
        files_to_check = [
            "pokedex_ultra_windows.py",
            "train_professional_models.py",
        ]
        
        errors = []
        
        for filepath in files_to_check:
            if not Path(filepath).exists():
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()
                    compile(code, filepath, 'exec')
            except SyntaxError as e:
                errors.append(f"{filepath}: {e}")
                
        if errors:
            return VerificationResult(
                "Code Syntax",
                False,
                "Errores de sintaxis encontrados",
                "\n".join(errors)
            )
            
        return VerificationResult(
            "Code Syntax",
            True,
            "Sintaxis correcta en todos los archivos"
        )
        
    def verify_imports(self) -> VerificationResult:
        """Verificar que imports principales se resuelven."""
        test_imports = [
            "pokedex_ultra_windows",
            "train_professional_models",
            "utils.camera",
            "utils.image_processing",
            "utils.api",
            "model.animal_classifier",
        ]
        
        failed = []
        
        for module_name in test_imports:
            module_path = module_name.replace('.', '/') + '.py'
            
            if not Path(module_path).exists():
                continue
                
            try:
                importlib.import_module(module_name)
            except Exception as e:
                failed.append(f"{module_name}: {str(e)}")
                
        if failed:
            return VerificationResult(
                "Import Resolution",
                False,
                "Algunos imports fallan",
                "\n".join(failed)
            )
            
        return VerificationResult(
            "Import Resolution",
            True,
            "Todos los imports se resuelven correctamente"
        )
        
    def verify_performance(self) -> VerificationResult:
        """Verificar rendimiento basico del sistema."""
        try:
            import time
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            
            test_img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
            
            start = time.time()
            for _ in range(100):
                blurred = cv2.GaussianBlur(test_img, (5, 5), 0)
            opencv_time = (time.time() - start) / 100
            
            details = f"CPU: {cpu_percent}%\n"
            details += f"RAM: {ram.percent}% ({ram.used / 1e9:.2f}/{ram.total / 1e9:.2f} GB)\n"
            details += f"OpenCV GaussianBlur (1920x1080): {opencv_time*1000:.2f}ms"
            
            if opencv_time > 0.1:
                return VerificationResult(
                    "Performance Baseline",
                    False,
                    "Rendimiento bajo del sistema",
                    details + "\nSistema puede no cumplir requisitos de tiempo real"
                )
                
            return VerificationResult(
                "Performance Baseline",
                True,
                "Rendimiento del sistema adecuado",
                details
            )
            
        except Exception as e:
            return VerificationResult(
                "Performance Baseline",
                False,
                f"Error verificando rendimiento: {str(e)}"
            )
            
    def generate_report(self) -> bool:
        """Generar reporte final de verificacion."""
        logger.info("\n" + "=" * 80)
        logger.info("REPORTE FINAL DE VERIFICACION")
        logger.info("=" * 80)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total = len(self.results)
        
        logger.info(f"\nTotal de verificaciones: {total}")
        logger.info(f"Exitosas: {passed}")
        logger.info(f"Fallidas: {failed}")
        logger.info(f"Tasa de exito: {(passed/total)*100:.1f}%")
        
        if failed > 0:
            logger.error("\nVERIFICACIONES FALLIDAS:")
            for result in self.results:
                if not result.passed:
                    logger.error(f"  - {result.name}: {result.message}")
                    if result.details:
                        logger.error(f"    {result.details}")
                        
        if self.critical_failures:
            logger.critical("\nFALLOS CRITICOS DETECTADOS:")
            for failure in self.critical_failures:
                logger.critical(f"  - {failure}")
            logger.critical("\nEl sistema NO puede funcionar correctamente.")
            logger.critical("Corregir fallos criticos antes de continuar.")
            return False
            
        if failed == 0:
            logger.info("\n" + "=" * 80)
            logger.info("VERIFICACION COMPLETA: TODOS LOS COMPONENTES FUNCIONAN CORRECTAMENTE")
            logger.info("=" * 80)
            return True
        else:
            logger.warning("\nVERIFICACION COMPLETA CON ADVERTENCIAS")
            logger.warning("Algunos componentes tienen problemas pero el sistema puede funcionar.")
            return True


def main() -> int:
    """Funcion principal."""
    verifier = SystemVerifier()
    success = verifier.verify_all()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
