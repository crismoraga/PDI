#!/usr/bin/env python3
"""
Utilidades de configuración para detectar plataforma y optimizar rendimiento.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional


class PlatformDetector:
    """Detector de plataforma y configurador automático."""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        
    def _detect_platform(self) -> Dict[str, Any]:
        """Detecta la plataforma actual y sus características."""
        info = {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "is_raspberry_pi": self._is_raspberry_pi(),
            "has_camera": self._detect_camera(),
            "memory_gb": self._get_memory_info(),
            "cpu_cores": os.cpu_count() or 1,
            "has_gpu": self._detect_gpu(),
            "display_available": self._has_display()
        }
        return info
    
    def _is_raspberry_pi(self) -> bool:
        """Detecta si estamos ejecutando en Raspberry Pi."""
        try:
            # Método 1: Verificar archivo /proc/cpuinfo
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read()
                    if "Raspberry Pi" in content or "BCM" in content:
                        return True
                        
            # Método 2: Verificar modelo específico
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read()
                    if "Raspberry Pi" in model:
                        return True
                        
            # Método 3: Verificar arquitectura ARM
            arch = platform.machine().lower()
            return arch in ["armv6l", "armv7l", "aarch64", "arm64"]
            
        except Exception:
            return False
    
    def _detect_camera(self) -> bool:
        """Detecta si hay cámara disponible."""
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
        except ImportError:
            pass
        
        # Verificar dispositivos v4l2 en Linux
        if platform.system() == "Linux":
            return len(list(Path("/dev").glob("video*"))) > 0
        
        return False
    
    def _get_memory_info(self) -> float:
        """Obtiene información de memoria del sistema."""
        try:
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            return kb / 1024 / 1024  # Convertir a GB
            
            # Fallback para Windows/Mac
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            return 1.0  # Valor por defecto conservador
    
    def _detect_gpu(self) -> bool:
        """Detecta si hay GPU disponible."""
        try:
            # Verificar GPU NVIDIA
            result = subprocess.run(["nvidia-smi"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Verificar GPU VideoCore en Raspberry Pi
        if self._is_raspberry_pi():
            try:
                result = subprocess.run(["vcgencmd", "get_mem", "gpu"], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=5)
                if result.returncode == 0 and "gpu=" in result.stdout:
                    gpu_mem = int(result.stdout.split("=")[1].replace("M", ""))
                    return gpu_mem >= 64  # Al menos 64MB GPU
            except (subprocess.SubprocessError, FileNotFoundError, ValueError):
                pass
        
        return False
    
    def _has_display(self) -> bool:
        """Detecta si hay display disponible."""
        if os.name == "posix":
            return "DISPLAY" in os.environ
        return True  # Asumir que Windows/Mac tienen display


class ConfigManager:
    """Gestor de configuración inteligente basado en plataforma."""
    
    def __init__(self, config_path: str = "config_advanced.json"):
        self.config_path = Path(config_path)
        self.detector = PlatformDetector()
        self.config = self._load_config()
        self.optimized_config = self._optimize_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga configuración desde archivo."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error cargando configuración: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto mínima."""
        return {
            "camera": {"resolution": {"width": 640, "height": 480}, "fps": 15},
            "model": {"confidence_threshold": 0.7},
            "ui": {"theme": "dark"}
        }
    
    def _optimize_config(self) -> Dict[str, Any]:
        """Optimiza configuración según la plataforma detectada."""
        config = self.config.copy()
        platform_info = self.detector.platform_info
        
        # Optimizaciones para Raspberry Pi
        if platform_info.get("is_raspberry_pi"):
            config = self._apply_raspberry_pi_optimizations(config, platform_info)
        else:
            config = self._apply_desktop_optimizations(config, platform_info)
        
        # Optimizaciones de memoria
        memory_gb = platform_info.get("memory_gb", 1.0)
        if memory_gb < 2.0:
            config = self._apply_low_memory_optimizations(config)
        elif memory_gb >= 8.0:
            config = self._apply_high_memory_optimizations(config)
        
        # Optimizaciones de CPU
        cpu_cores = platform_info.get("cpu_cores", 1)
        config["model"]["inference"]["num_threads"] = min(cpu_cores, 4)
        
        return config
    
    def _apply_raspberry_pi_optimizations(self, config: Dict[str, Any], 
                                        platform_info: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica optimizaciones específicas para Raspberry Pi."""
        
        # Configuración de cámara optimizada
        if "camera" in config:
            config["camera"]["resolution"] = config["camera"]["resolution"]["raspberry_pi"]
            config["camera"]["fps"] = config["camera"]["fps"]["raspberry_pi"]
            config["camera"]["buffer_size"] = 1
        
        # Configuración de modelo optimizada
        if "model" in config:
            # Preferir TensorFlow Lite
            config["model"]["primary"]["use_edge_tpu"] = platform_info.get("has_gpu", False)
            config["model"]["inference"]["batch_size"] = 1
            config["model"]["inference"]["enable_gpu_acceleration"] = False
            
        # Configuración de procesamiento optimizada
        if "image_processing" in config:
            # Reducir parámetros computacionalmente intensivos
            config["image_processing"]["enhancement"]["clahe_clip_limit"] = 2.0
            config["image_processing"]["segmentation"]["kmeans_k"] = 3
            config["image_processing"]["segmentation"]["kmeans_iterations"] = 10
        
        # Configuración de UI optimizada
        if "ui" in config:
            config["ui"]["performance"]["video_refresh_ms"] = 66  # ~15 FPS
            config["ui"]["performance"]["enable_vsync"] = False
        
        # Configuración de rendimiento
        config["performance"] = config["performance"]["raspberry_pi"]
        
        return config
    
    def _apply_desktop_optimizations(self, config: Dict[str, Any], 
                                   platform_info: Dict[str, Any]) -> Dict[str, Any]:
        """Aplica optimizaciones para desktop."""
        
        # Configuración de cámara de alta calidad
        if "camera" in config:
            config["camera"]["resolution"] = config["camera"]["resolution"]["desktop"]
            config["camera"]["fps"] = config["camera"]["fps"]["desktop"]
        
        # Configuración de modelo con mejor calidad
        if "model" in config:
            config["model"]["inference"]["enable_gpu_acceleration"] = platform_info.get("has_gpu", False)
            config["model"]["inference"]["batch_size"] = 2
        
        # Configuración de rendimiento desktop
        config["performance"] = config["performance"]["desktop"]
        
        return config
    
    def _apply_low_memory_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizaciones para sistemas con poca memoria."""
        
        # Reducir cache y buffers
        config["performance"]["cache_size"] = 25
        config["camera"]["buffer_size"] = 1
        config["performance"]["memory_limit_mb"] = 512
        
        # Reducir calidad de procesamiento
        config["image_processing"]["segmentation"]["kmeans_k"] = 3
        config["api"]["wikipedia"]["summary_length"] = 400
        
        return config
    
    def _apply_high_memory_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimizaciones para sistemas con mucha memoria."""
        
        # Aumentar cache y buffers
        config["performance"]["cache_size"] = 500
        config["camera"]["buffer_size"] = 4
        config["performance"]["memory_limit_mb"] = 4096
        
        # Mejorar calidad de procesamiento
        config["image_processing"]["segmentation"]["kmeans_k"] = 8
        config["api"]["wikipedia"]["summary_length"] = 1000
        
        return config
    
    def get_optimized_config(self) -> Dict[str, Any]:
        """Devuelve la configuración optimizada."""
        return self.optimized_config
    
    def save_optimized_config(self, path: Optional[str] = None) -> None:
        """Guarda la configuración optimizada."""
        output_path = Path(path) if path else Path("config_optimized.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.optimized_config, f, indent=2, ensure_ascii=False)
    
    def print_platform_info(self) -> None:
        """Imprime información de la plataforma detectada."""
        info = self.detector.platform_info
        
        print("INFORMACIÓN DE PLATAFORMA DETECTADA")
        print("=" * 50)
        print(f"Sistema Operativo: {info['os']}")
        print(f"Arquitectura: {info['architecture']}")
        print(f"Python: {info['python_version']}")
        print(f"Raspberry Pi: {'Sí' if info['is_raspberry_pi'] else 'No'}")
        print(f"Cámara disponible: {'Sí' if info['has_camera'] else 'No'}")
        print(f"Memoria RAM: {info['memory_gb']:.1f} GB")
        print(f"Cores CPU: {info['cpu_cores']}")
        print(f"GPU disponible: {'Sí' if info['has_gpu'] else 'No'}")
        print(f"Display disponible: {'Sí' if info['display_available'] else 'No'}")
        print("=" * 50)


def main():
    """Función principal para configuración automática."""
    print("CONFIGURADOR AUTOMÁTICO POKÉDEX ANIMAL")
    print("=" * 50)
    
    # Detectar plataforma
    manager = ConfigManager()
    manager.print_platform_info()
    
    # Generar configuración optimizada
    print("\nGenerando configuración optimizada...")
    manager.save_optimized_config()
    
    print("✅ Configuración optimizada guardada en 'config_optimized.json'")
    print("\nPara usar la configuración optimizada:")
    print("1. Renombra 'config.json' a 'config_backup.json'")
    print("2. Renombra 'config_optimized.json' a 'config.json'")
    print("3. Ejecuta la aplicación normalmente")
    
    # Mostrar configuraciones clave aplicadas
    config = manager.get_optimized_config()
    print(f"\nCONFIGURACIÓN APLICADA:")
    print(f"- Resolución de cámara: {config['camera']['resolution']['width']}x{config['camera']['resolution']['height']}")
    print(f"- FPS: {config['camera']['fps']}")
    print(f"- Modelo: {'TensorFlow Lite' if 'tflite' in config['model']['primary']['model_path'] else 'Keras'}")
    print(f"- Hilos: {config['model']['inference']['num_threads']}")
    print(f"- Cache: {config['performance']['cache_size']} entradas")


if __name__ == "__main__":
    main()