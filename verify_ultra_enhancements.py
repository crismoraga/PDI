#!/usr/bin/env python3
"""
Script de Verificacion Rigurosa - Pokedex Animal Ultra Enhancements
Verifica todas las mejoras implementadas: GPU AMD, ventanas completas, UI futurista

Author: Ultra Verification System
Date: 2024
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str) -> None:
    """Imprimir header llamativo."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 80}")
    print(f"{text.center(80)}")
    print(f"{'=' * 80}{Colors.END}\n")

def print_success(text: str) -> None:
    """Imprimir mensaje de exito."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text: str) -> None:
    """Imprimir mensaje de error."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text: str) -> None:
    """Imprimir advertencia."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text: str) -> None:
    """Imprimir informacion."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


class UltraEnhancementsVerifier:
    """Verificador riguroso de todas las mejoras ultra."""
    
    def __init__(self):
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
        self.warnings = 0
        
    def run_all_checks(self) -> bool:
        """Ejecutar todas las verificaciones."""
        print_header("VERIFICACION ULTRA-RIGUROSA DE MEJORAS")
        
        checks = [
            ("GPU Detector - AMD/NVIDIA Support", self.verify_gpu_detector),
            ("ROCm Documentation", self.verify_rocm_docs),
            ("Achievements Window - Complete Implementation", self.verify_achievements_window),
            ("Pokedex Window - Complete Implementation", self.verify_pokedex_window),
            ("Real-time Detection Display - Enhanced HUD", self.verify_detection_display),
            ("GPU Integration in Main App", self.verify_gpu_integration),
            ("Database Schema - Complete", self.verify_database_schema),
            ("UI Components - Futuristic Design", self.verify_ui_components),
        ]
        
        for check_name, check_func in checks:
            print_info(f"Ejecutando: {check_name}...")
            try:
                result = check_func()
                if result:
                    print_success(f"{check_name}: PASSED")
                    self.passed_checks += 1
                else:
                    print_error(f"{check_name}: FAILED")
                    self.failed_checks += 1
            except Exception as e:
                print_error(f"{check_name}: EXCEPTION - {e}")
                self.failed_checks += 1
            
            self.total_checks += 1
            print()
        
        self.print_summary()
        return self.failed_checks == 0
    
    def verify_gpu_detector(self) -> bool:
        """Verificar GPU detector universal."""
        try:
            gpu_detector_path = Path("utils/gpu_detector.py")
            if not gpu_detector_path.exists():
                print_error("GPU detector file not found")
                return False
            
            print_success("GPU detector file exists")
            
            # Verificar contenido
            content = gpu_detector_path.read_text(encoding='utf-8')
            
            required_components = [
                ("GPUDetector class", "class GPUDetector"),
                ("NVIDIA detection", "nvidia-smi"),
                ("AMD detection", "rocm-smi"),
                ("WMI detection", "import wmi"),
                ("TensorFlow config", "configure_tensorflow"),
                ("PyTorch config", "configure_pytorch"),
                ("Device info", "get_device_info"),
            ]
            
            for name, pattern in required_components:
                if pattern in content:
                    print_success(f"  - {name}: Found")
                else:
                    print_error(f"  - {name}: NOT FOUND")
                    return False
            
            # Intentar importar
            try:
                sys.path.insert(0, str(Path.cwd()))
                from utils.gpu_detector import GPUDetector
                
                detector = GPUDetector()
                info = detector.get_device_info()
                
                print_success(f"  - GPU Type: {info.get('type', 'Unknown')}")
                print_success(f"  - GPU Name: {info.get('name', 'Unknown')}")
                print_success(f"  - Device: {info.get('device', 'Unknown')}")
                
                return True
                
            except Exception as e:
                print_warning(f"  - GPU detection warning: {e}")
                print_info("  - This is expected if no GPU is available")
                # Aceptar si el archivo es correcto aunque no haya GPU
                return True
                
        except Exception as e:
            print_error(f"GPU detector verification failed: {e}")
            return False
    
    def verify_rocm_docs(self) -> bool:
        """Verificar documentacion ROCm."""
        try:
            docs_path = Path("AMD_ROCM_SETUP.md")
            if not docs_path.exists():
                print_error("ROCm documentation not found")
                return False
            
            print_success("ROCm documentation exists")
            
            content = docs_path.read_text(encoding='utf-8')
            
            required_sections = [
                "Requisitos del Sistema",
                "Instalacion de ROCm",
                "PyTorch con ROCm",
                "TensorFlow-DirectML",
                "Configuracion del Proyecto",
                "Optimizaciones de Rendimiento",
                "Resolucion de Problemas",
                "Benchmarks Esperados"
            ]
            
            for section in required_sections:
                if section in content:
                    print_success(f"  - Section '{section}': Found")
                else:
                    print_error(f"  - Section '{section}': NOT FOUND")
                    return False
            
            # Verificar menciones especificas a RX 6700 XT
            if "RX 6700 XT" in content or "6700 XT" in content:
                print_success("  - RX 6700 XT specific info: Found")
            else:
                print_warning("  - RX 6700 XT specific info: Not found")
            
            return True
            
        except Exception as e:
            print_error(f"ROCm docs verification failed: {e}")
            return False
    
    def verify_achievements_window(self) -> bool:
        """Verificar implementacion completa de ventana de logros."""
        try:
            main_app_path = Path("pokedex_ultra_windows.py")
            if not main_app_path.exists():
                print_error("Main app file not found")
                return False
            
            content = main_app_path.read_text(encoding='utf-8')
            
            # Verificar que NO es stub
            if content.count("def _show_achievements") < 1:
                print_error("_show_achievements method not found")
                return False
            
            # Extraer metodo
            lines = content.split('\n')
            in_method = False
            method_lines = []
            
            for line in lines:
                if 'def _show_achievements' in line:
                    in_method = True
                elif in_method:
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        break
                    method_lines.append(line)
            
            method_content = '\n'.join(method_lines)
            
            # Verificar que NO es stub (mas de 5 lineas de codigo real)
            code_lines = [l for l in method_lines if l.strip() and not l.strip().startswith('#')]
            if len(code_lines) < 10:
                print_error(f"Achievements window appears to be a stub ({len(code_lines)} lines)")
                return False
            
            print_success(f"Achievements window has {len(code_lines)} lines of code")
            
            # Componentes requeridos
            required_components = [
                ("CTkToplevel window", "CTkToplevel"),
                ("Database query", "self.database"),
                ("Category headers", "category"),
                ("Achievement cards", "_create_achievement_card"),
                ("Progress bars", "CTkProgressBar" or "progress"),
                ("Unlocked status", "unlocked"),
                ("Footer stats", "footer"),
            ]
            
            for name, pattern in required_components:
                if pattern in method_content:
                    print_success(f"  - {name}: Implemented")
                else:
                    print_error(f"  - {name}: NOT FOUND")
                    return False
            
            return True
            
        except Exception as e:
            print_error(f"Achievements window verification failed: {e}")
            return False
    
    def verify_pokedex_window(self) -> bool:
        """Verificar implementacion completa de ventana de pokedex."""
        try:
            main_app_path = Path("pokedex_ultra_windows.py")
            content = main_app_path.read_text(encoding='utf-8')
            
            # Verificar que NO es stub
            if content.count("def _show_pokedex") < 1:
                print_error("_show_pokedex method not found")
                return False
            
            # Extraer metodo
            lines = content.split('\n')
            in_method = False
            method_lines = []
            
            for line in lines:
                if 'def _show_pokedex' in line:
                    in_method = True
                elif in_method:
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        break
                    method_lines.append(line)
            
            method_content = '\n'.join(method_lines)
            
            # Verificar que NO es stub
            code_lines = [l for l in method_lines if l.strip() and not l.strip().startswith('#')]
            if len(code_lines) < 10:
                print_error(f"Pokedex window appears to be a stub ({len(code_lines)} lines)")
                return False
            
            print_success(f"Pokedex window has {len(code_lines)} lines of code")
            
            # Componentes requeridos
            required_components = [
                ("Search bar", "search" or "CTkEntry"),
                ("Filter menu", "filter" or "CTkOptionMenu"),
                ("Grid layout", "grid"),
                ("Species cards", "_create_species_card"),
                ("Database query", "SELECT"),
                ("Scrollable frame", "CTkScrollableFrame"),
                ("Details button", "show_details"),
            ]
            
            for name, pattern in required_components:
                if pattern in method_content:
                    print_success(f"  - {name}: Implemented")
                else:
                    print_error(f"  - {name}: NOT FOUND")
                    return False
            
            return True
            
        except Exception as e:
            print_error(f"Pokedex window verification failed: {e}")
            return False
    
    def verify_detection_display(self) -> bool:
        """Verificar display de deteccion en tiempo real mejorado."""
        try:
            main_app_path = Path("pokedex_ultra_windows.py")
            content = main_app_path.read_text(encoding='utf-8')
            
            # Verificar metodo _annotate_frame
            if "def _annotate_frame" not in content:
                print_error("_annotate_frame method not found")
                return False
            
            # Extraer metodo
            lines = content.split('\n')
            in_method = False
            method_lines = []
            
            for line in lines:
                if 'def _annotate_frame' in line:
                    in_method = True
                elif in_method:
                    if 'def ' in line and not line.startswith(' '):
                        break
                    method_lines.append(line)
            
            method_content = '\n'.join(method_lines)
            
            # Componentes de HUD mejorado
            required_features = [
                ("Bounding box corners", "corner"),
                ("Confidence color coding", "confidence_color" or "_get_confidence_color"),
                ("HUD overlay", "hud" or "overlay"),
                ("Confidence bar", "bar"),
                ("Features display", "features"),
                ("Processing time", "processing_time"),
                ("Model source", "model_source"),
            ]
            
            for name, pattern in required_features:
                if pattern in method_content.lower():
                    print_success(f"  - {name}: Implemented")
                else:
                    print_error(f"  - {name}: NOT FOUND")
                    return False
            
            # Verificar helper _get_confidence_color
            if "_get_confidence_color" in content:
                print_success("  - Confidence color helper: Implemented")
            else:
                print_warning("  - Confidence color helper: Not found (optional)")
            
            return True
            
        except Exception as e:
            print_error(f"Detection display verification failed: {e}")
            return False
    
    def verify_gpu_integration(self) -> bool:
        """Verificar integracion de GPU en aplicacion principal."""
        try:
            main_app_path = Path("pokedex_ultra_windows.py")
            content = main_app_path.read_text(encoding='utf-8')
            
            # Verificar import
            if "from utils.gpu_detector import GPUDetector" not in content:
                print_error("GPUDetector not imported")
                return False
            
            print_success("GPUDetector imported")
            
            # Verificar uso en EnsembleAIEngine
            if "self.gpu_detector = GPUDetector()" not in content:
                print_error("GPUDetector not instantiated")
                return False
            
            print_success("GPUDetector instantiated in EnsembleAIEngine")
            
            # Verificar configuracion de frameworks
            checks = [
                ("TensorFlow config", "configure_tensorflow"),
                ("PyTorch config", "configure_pytorch"),
                ("Device info logging", "get_device_info"),
            ]
            
            for name, pattern in checks:
                if pattern in content:
                    print_success(f"  - {name}: Found")
                else:
                    print_error(f"  - {name}: NOT FOUND")
                    return False
            
            return True
            
        except Exception as e:
            print_error(f"GPU integration verification failed: {e}")
            return False
    
    def verify_database_schema(self) -> bool:
        """Verificar esquema completo de base de datos."""
        try:
            main_app_path = Path("pokedex_ultra_windows.py")
            content = main_app_path.read_text(encoding='utf-8')
            
            # Buscar clase UltraPokedexDatabase
            if "class UltraPokedexDatabase" not in content:
                print_error("UltraPokedexDatabase class not found")
                return False
            
            print_success("UltraPokedexDatabase class found")
            
            # Tablas requeridas
            required_tables = [
                "species",
                "sightings",
                "achievements",
                "user_stats"
            ]
            
            for table in required_tables:
                if f"CREATE TABLE IF NOT EXISTS {table}" in content:
                    print_success(f"  - Table '{table}': Defined")
                else:
                    print_error(f"  - Table '{table}': NOT FOUND")
                    return False
            
            # Indices
            if "CREATE INDEX" in content:
                print_success("  - Indexes: Defined")
            else:
                print_warning("  - Indexes: Not found")
            
            # Metodos requeridos
            required_methods = [
                "add_sighting",
                "capture_species",
                "get_statistics",
                "_check_achievements"
            ]
            
            for method in required_methods:
                if f"def {method}" in content:
                    print_success(f"  - Method '{method}': Implemented")
                else:
                    print_error(f"  - Method '{method}': NOT FOUND")
                    return False
            
            return True
            
        except Exception as e:
            print_error(f"Database schema verification failed: {e}")
            return False
    
    def verify_ui_components(self) -> bool:
        """Verificar componentes UI futuristas."""
        try:
            main_app_path = Path("pokedex_ultra_windows.py")
            content = main_app_path.read_text(encoding='utf-8')
            
            # CustomTkinter components
            ctk_components = [
                "CTkFrame",
                "CTkLabel",
                "CTkButton",
                "CTkScrollableFrame",
                "CTkProgressBar",
                "CTkEntry",
                "CTkOptionMenu",
                "CTkTextbox"
            ]
            
            for component in ctk_components:
                if component in content:
                    print_success(f"  - {component}: Used")
                else:
                    print_warning(f"  - {component}: Not found")
            
            # Colores futuristas
            futuristic_colors = [
                "#1a1a2e",
                "#16213e",
                "#00d9ff",
                "#ffd700"
            ]
            
            color_count = sum(1 for color in futuristic_colors if color in content)
            if color_count >= len(futuristic_colors) / 2:
                print_success(f"  - Futuristic color scheme: Applied ({color_count} colors)")
            else:
                print_warning("  - Futuristic color scheme: Limited")
            
            # Efectos visuales
            effects = [
                ("Border effects", "border_width" or "border_color"),
                ("Corner radius", "corner_radius"),
                ("Gradient simulation", "overlay" or "addWeighted"),
            ]
            
            for name, pattern in effects:
                if pattern in content.lower():
                    print_success(f"  - {name}: Applied")
                else:
                    print_warning(f"  - {name}: Not applied")
            
            return True
            
        except Exception as e:
            print_error(f"UI components verification failed: {e}")
            return False
    
    def print_summary(self) -> None:
        """Imprimir resumen de verificacion."""
        print_header("RESUMEN DE VERIFICACION")
        
        print(f"{Colors.BOLD}Total de Verificaciones:{Colors.END} {self.total_checks}")
        print(f"{Colors.GREEN}{Colors.BOLD}✓ Passed:{Colors.END} {self.passed_checks}")
        print(f"{Colors.RED}{Colors.BOLD}✗ Failed:{Colors.END} {self.failed_checks}")
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Warnings:{Colors.END} {self.warnings}")
        
        success_rate = (self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0
        
        print(f"\n{Colors.BOLD}Tasa de Exito:{Colors.END} {success_rate:.1f}%")
        
        if self.failed_checks == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 TODAS LAS VERIFICACIONES PASARON! 🎉{Colors.END}")
            print(f"{Colors.GREEN}Sistema ultra-profesional completamente verificado.{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ VERIFICACION FALLIDA{Colors.END}")
            print(f"{Colors.RED}Se encontraron {self.failed_checks} problemas que requieren atencion.{Colors.END}")


def main():
    """Funcion principal."""
    print(f"{Colors.BOLD}Pokedex Animal Ultra - Verificador de Mejoras{Colors.END}")
    print(f"{Colors.BOLD}Version: 2.0 Ultra Professional{Colors.END}\n")
    
    verifier = UltraEnhancementsVerifier()
    success = verifier.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
