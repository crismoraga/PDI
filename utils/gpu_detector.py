"""
Detector Universal de GPU para AMD y NVIDIA
Configura automaticamente TensorFlow y PyTorch para usar GPU disponible
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class GPUDetector:
    """Detector y configurador de GPU universal."""
    
    def __init__(self):
        self.gpu_type: Optional[str] = None
        self.gpu_name: Optional[str] = None
        self.gpu_available: bool = False
        self.device_name: str = "cpu"
        
        self._detect_gpu()
        
    def _detect_gpu(self) -> None:
        """Detectar tipo de GPU disponible."""
        
        if self._check_nvidia_gpu():
            self.gpu_type = "NVIDIA"
            self.gpu_available = True
            logger.info(f"GPU NVIDIA detectada: {self.gpu_name}")
            
        elif self._check_amd_gpu():
            self.gpu_type = "AMD"
            self.gpu_available = True
            logger.info(f"GPU AMD detectada: {self.gpu_name}")
            
        else:
            self.gpu_type = None
            self.gpu_available = False
            logger.warning("No se detectó GPU. Usando CPU.")
            
    def _check_nvidia_gpu(self) -> bool:
        """Verificar disponibilidad de GPU NVIDIA."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self.gpu_name = result.stdout.strip()
                return True
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        return False
        
    def _check_amd_gpu(self) -> bool:
        """Verificar disponibilidad de GPU AMD."""
        if platform.system() != "Windows":
            try:
                result = subprocess.run(
                    ['rocm-smi', '--showproductname'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'GPU' in line:
                            self.gpu_name = line.strip()
                            return True
                            
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
                
        try:
            import wmi
            c = wmi.WMI()
            
            for gpu in c.Win32_VideoController():
                if 'AMD' in gpu.Name or 'Radeon' in gpu.Name:
                    self.gpu_name = gpu.Name
                    return True
                    
        except ImportError:
            pass
            
        return False
        
    def configure_tensorflow(self) -> None:
        """Configurar TensorFlow para usar GPU disponible."""
        try:
            import tensorflow as tf
            
            if self.gpu_type == "NVIDIA":
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"TensorFlow configurado para {len(gpus)} GPU(s) NVIDIA")
                    self.device_name = f"GPU:0"
                    
            elif self.gpu_type == "AMD":
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                
                try:
                    import tensorflow_rocm
                    logger.info("TensorFlow-ROCm disponible para GPU AMD")
                    self.device_name = f"GPU:0"
                    
                except ImportError:
                    logger.warning(
                        "TensorFlow-ROCm no instalado. GPU AMD no disponible para TensorFlow. "
                        "Instalar: pip install tensorflow-rocm"
                    )
                    
        except Exception as e:
            logger.error(f"Error configurando TensorFlow: {e}")
            
    def configure_pytorch(self) -> str:
        """
        Configurar PyTorch para usar GPU disponible.
        
        Returns:
            str: Nombre del dispositivo ('cuda', 'hip', 'cpu')
        """
        try:
            import torch
            
            if self.gpu_type == "NVIDIA":
                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info(f"PyTorch usando CUDA GPU: {torch.cuda.get_device_name(0)}")
                    torch.cuda.empty_cache()
                    return device
                    
            elif self.gpu_type == "AMD":
                try:
                    if torch.cuda.is_available():
                        device = "cuda"
                        logger.info("PyTorch con soporte ROCm detectado")
                        return device
                        
                except Exception:
                    logger.warning(
                        "PyTorch con ROCm no configurado correctamente. "
                        "Instalar: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7"
                    )
                    
            logger.info("PyTorch usando CPU")
            return "cpu"
            
        except Exception as e:
            logger.error(f"Error configurando PyTorch: {e}")
            return "cpu"
            
    def get_device_info(self) -> Dict[str, any]:
        """Obtener informacion detallada del dispositivo."""
        info = {
            "type": self.gpu_type,
            "name": self.gpu_name,
            "device": self.device_name,
            "gpu_available": self.gpu_available,
            "cuda_available": self.gpu_type == "NVIDIA",
            "rocm_available": self.gpu_type == "AMD",
        }
        
        if self.gpu_type == "NVIDIA":
            info.update(self._get_nvidia_info())
        elif self.gpu_type == "AMD":
            info.update(self._get_amd_info())
            
        return info
        
    def _get_nvidia_info(self) -> Dict[str, any]:
        """Obtener informacion detallada de GPU NVIDIA."""
        info = {}
        
        try:
            import torch
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                
        except Exception:
            pass
            
        return info
        
    def _get_amd_info(self) -> Dict[str, any]:
        """Obtener informacion detallada de GPU AMD."""
        info = {}
        
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                info["rocm_available"] = True
                
        except Exception:
            info["rocm_available"] = False
            
        return info
        
    def print_config(self) -> None:
        """Imprimir configuracion de GPU."""
        print("\n" + "="*60)
        print("CONFIGURACION DE GPU")
        print("="*60)
        
        if self.gpu_available:
            print(f"Tipo: {self.gpu_type}")
            print(f"Nombre: {self.gpu_name}")
            print(f"Dispositivo: {self.device_name}")
            
            info = self.get_device_info()
            
            if self.gpu_type == "NVIDIA":
                if "cuda_version" in info:
                    print(f"CUDA Version: {info['cuda_version']}")
                if "gpu_count" in info:
                    print(f"GPUs disponibles: {info['gpu_count']}")
                if "gpu_memory_total" in info:
                    print(f"Memoria GPU: {info['gpu_memory_total']} GB")
                    
            elif self.gpu_type == "AMD":
                if info.get("rocm_available"):
                    print("ROCm: Disponible")
                else:
                    print("ROCm: No detectado")
                    print("Instalar: https://rocm.docs.amd.com/")
        else:
            print("GPU: No disponible")
            print("Dispositivo: CPU")
            
        print("="*60 + "\n")


def get_optimal_device() -> Tuple[str, GPUDetector]:
    """
    Obtener dispositivo optimo y detector configurado.
    
    Returns:
        Tuple[str, GPUDetector]: (nombre_dispositivo, detector)
    """
    detector = GPUDetector()
    detector.configure_tensorflow()
    device = detector.configure_pytorch()
    
    return device, detector


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    device, detector = get_optimal_device()
    detector.print_config()
    
    print(f"\nDispositivo optimo para PyTorch: {device}")
    print(f"Dispositivo optimo para TensorFlow: {detector.device_name}")
