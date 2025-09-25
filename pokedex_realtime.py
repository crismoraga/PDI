#!/usr/bin/env python3
"""
Aplicación Pokédx Animal - Versión Profesional de Tiempo Real
Optimizada para Raspberry Pi con funcionalidad Pokédx completa

Esta aplicación replica fielmente el funcionamiento de una Pokédx del mundo Pokémn,
aplicada a animales reales, con reconocimiento en tiempo real y base de datos completa.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import colorsys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk

from utils.camera import CameraCapture
from utils.image_processing import ImageProcessor
from utils.api import AnimalInfoAPI
from utils.platform_config import PlatformDetector
from model.animal_classifier import AnimalClassifier
from pokedex.db import PokedexRepository, PokedexEntry

# Configuración de directorios
SNAPSHOT_DIR = Path("data/snapshots")
EXPORT_DIR = Path("data/exports")
LOGS_DIR = Path("data/logs")

# Configuración de aplicación
APP_TITLE = "Pokédx Animal - Tiempo Real"
WINDOW_SIZE = "1400x900"
VIDEO_REFRESH_MS = 33  # ~30 FPS para tiempo real
PREDICTION_INTERVAL_MS = 1000  # Predicción cada segundo
CONFIDENCE_THRESHOLD = 0.3  # Umbral mínimo para detección
AUTO_CAPTURE_THRESHOLD = 0.7  # Umbral para captura automática


class RealTimePokedexApp:
    """
    Aplicación principal de Pokédx Animal con funcionalidad de tiempo real.
    
    Características principales:
    - Reconocimiento en tiempo real con cámara siempre activa
    - Base de datos completa tipo Pokédx con entradas persistentes
    - Sistema de "visto" vs "capturado" como en Pokémn
    - Análisis visual avanzado (color dominante, tamaño, ubicación)
    - Optimizado para Raspberry Pi con TensorFlow Lite
    - Vista de detalle completa de cada entrada
    """
    
    def __init__(self) -> None:
        """Inicializar la aplicación."""
        self._setup_logging()
        self._setup_directories()
        
        # Detectar plataforma y configurar optimizaciones
        self.platform = PlatformDetector()
        self._configure_platform_optimizations()
        
        # Inicializar componentes principales
        self._init_components()
        
        # Estado de la aplicación
        self.is_running = False
        self.current_frame = None
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_lock = threading.Lock()
        
        # UI y controles
        self._init_ui()
        
        # Hilo de predicción en tiempo real
        self.prediction_thread = None
        
        logging.info("Pokédx Animal inicializada correctamente")
    
    def _setup_logging(self) -> None:
        """Configurar sistema de logging."""
        LOGS_DIR.mkdir(exist_ok=True)
        
        log_file = LOGS_DIR / f"pokedex_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _setup_directories(self) -> None:
        """Crear directorios necesarios."""
        for directory in [SNAPSHOT_DIR, EXPORT_DIR, LOGS_DIR, Path("data")]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _configure_platform_optimizations(self) -> None:
        """Configurar optimizaciones específicas de plataforma."""
        if self.platform.platform_info["is_raspberry_pi"]:
            # Configuración optimizada para Raspberry Pi
            global VIDEO_REFRESH_MS, PREDICTION_INTERVAL_MS
            
            memory_gb = self.platform.platform_info["memory_gb"]
            if memory_gb <= 1:
                VIDEO_REFRESH_MS = 50  # 20 FPS
                PREDICTION_INTERVAL_MS = 2000  # Predicción cada 2 segundos
            elif memory_gb <= 4:
                VIDEO_REFRESH_MS = 40  # 25 FPS
                PREDICTION_INTERVAL_MS = 1500  # Predicción cada 1.5 segundos
            
            logging.info(f"Optimizado para Raspberry Pi: {memory_gb}GB RAM")
        else:
            logging.info("Ejecutándose en plataforma de escritorio")
    
    def _init_components(self) -> None:
        """Inicializar componentes principales."""
        logging.info("Inicializando componentes...")
        
        # Cámara
        self.camera = CameraCapture()
        if self.platform.platform_info["is_raspberry_pi"]:
            # Configuración optimizada para Pi
            self.camera.set_resolution(640, 480)
            self.camera.set_fps(25)
        
        # Procesamiento de imágenes
        self.image_processor = ImageProcessor()
        
        # Clasificador (TensorFlow Lite si está disponible)
        self.classifier_backend = "keras"
        try:
            from model.tflite_classifier import TFLiteAnimalClassifier
            self.classifier = TFLiteAnimalClassifier()
            self.classifier_backend = "tflite"
            logging.info("Clasificador TensorFlow Lite activo")
        except Exception as exc:
            logging.warning(f"TensorFlow Lite no disponible: {exc}")
            self.classifier = AnimalClassifier()
            self.classifier_backend = "keras"
            logging.info("Clasificador Keras activo")
        
        # API de información
        self.animal_api = AnimalInfoAPI()
        
        # Base de datos Pokédx
        db_path = Path("data") / "pokedex.db"
        self.pokedx_repo = PokedexRepository(str(db_path))
        
        logging.info("Componentes inicializados correctamente")
    
    def _init_ui(self) -> None:
        """Inicializar interfaz de usuario."""
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Estilo moderno
        self._setup_style()
        
        # Layout principal
        self._create_main_layout()
        
        # Variables de estado de UI
        self.status_var = tk.StringVar(value="Inicializando...")
        self.detection_var = tk.StringVar(value="Sin detección")
        self.confidence_var = tk.StringVar(value="0%")
        self.entries_count_var = tk.StringVar(value="0")
        self.captured_count_var = tk.StringVar(value="0")
        
        # Actualizar contadores iniciales
        self._update_counters()
    
    def _setup_style(self) -> None:
        """Configurar estilo visual de la aplicación."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colores profesionales sin emojis
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10))
        style.configure('Detection.TLabel', font=('Arial', 14, 'bold'), foreground='blue')
        style.configure('Confidence.TLabel', font=('Arial', 12), foreground='green')
        
        # Botones personalizados
        style.configure('Action.TButton', font=('Arial', 11, 'bold'))
        style.configure('Capture.TButton', font=('Arial', 11, 'bold'), foreground='red')
    
    def _create_main_layout(self) -> None:
        """Crear el layout principal de la interfaz."""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Panel izquierdo - Cámara y controles
        self._create_camera_panel(main_frame)
        
        # Panel derecho - Pokédx y información
        self._create_pokedx_panel(main_frame)
        
        # Panel inferior - Estado y estadísticas
        self._create_status_panel(main_frame)
    
    def _create_camera_panel(self, parent: ttk.Widget) -> None:
        """Crear panel de cámara y controles."""
        camera_frame = ttk.LabelFrame(parent, text="Cámara en Tiempo Real", padding="5")
        camera_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Video feed
        self.video_label = ttk.Label(camera_frame)
        self.video_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        # Controles de cámara
        controls_frame = ttk.Frame(camera_frame)
        controls_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        self.start_button = ttk.Button(
            controls_frame, 
            text="Iniciar Pokédx", 
            command=self._start_real_time,
            style='Action.TButton'
        )
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(
            controls_frame, 
            text="Detener", 
            command=self._stop_real_time,
            style='Action.TButton',
            state='disabled'
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.capture_button = ttk.Button(
            controls_frame, 
            text="Capturar Manualmente", 
            command=self._manual_capture,
            style='Capture.TButton',
            state='disabled'
        )
        self.capture_button.grid(row=0, column=2, padx=5)
        
        # Información de detección actual
        detection_frame = ttk.LabelFrame(camera_frame, text="Detección Actual", padding="5")
        detection_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Label(detection_frame, text="Animal:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(detection_frame, textvariable=self.detection_var, style='Detection.TLabel').grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(detection_frame, text="Confianza:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(detection_frame, textvariable=self.confidence_var, style='Confidence.TLabel').grid(row=1, column=1, sticky=tk.W)
    
    def _create_pokedx_panel(self, parent: ttk.Widget) -> None:
        """Crear panel de Pokédx con entradas y búsqueda."""
        pokedx_frame = ttk.LabelFrame(parent, text="Pokédx Animal", padding="5")
        pokedx_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        pokedx_frame.columnconfigure(0, weight=1)
        pokedx_frame.rowconfigure(1, weight=1)
        
        # Barra de búsqueda
        search_frame = ttk.Frame(pokedx_frame)
        search_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        search_frame.columnconfigure(1, weight=1)
        
        ttk.Label(search_frame, text="Buscar:").grid(row=0, column=0, padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        self.search_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        self.search_entry.bind('<KeyRelease>', self._on_search_change)
        
        search_button = ttk.Button(search_frame, text="Buscar", command=self._search_entries)
        search_button.grid(row=0, column=2)
        
        # Lista de entradas
        list_frame = ttk.Frame(pokedx_frame)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        # Treeview para mostrar entradas
        columns = ('ID', 'Nombre', 'Fecha', 'Confianza', 'Estado')
        self.entries_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        # Configurar columnas
        self.entries_tree.heading('ID', text='#')
        self.entries_tree.heading('Nombre', text='Nombre')
        self.entries_tree.heading('Fecha', text='Fecha')
        self.entries_tree.heading('Confianza', text='Confianza')
        self.entries_tree.heading('Estado', text='Estado')
        
        self.entries_tree.column('ID', width=50)
        self.entries_tree.column('Nombre', width=150)
        self.entries_tree.column('Fecha', width=120)
        self.entries_tree.column('Confianza', width=80)
        self.entries_tree.column('Estado', width=80)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.entries_tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.entries_tree.xview)
        self.entries_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid del treeview y scrollbars
        self.entries_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Eventos de selección
        self.entries_tree.bind('<Double-1>', self._on_entry_double_click)
        
        # Botones de acción
        actions_frame = ttk.Frame(pokedx_frame)
        actions_frame.grid(row=2, column=0, pady=(10, 0))
        
        ttk.Button(actions_frame, text="Ver Detalle", command=self._show_entry_detail).grid(row=0, column=0, padx=5)
        ttk.Button(actions_frame, text="Marcar Capturado", command=self._mark_captured).grid(row=0, column=1, padx=5)
        ttk.Button(actions_frame, text="Exportar", command=self._export_entries).grid(row=0, column=2, padx=5)
        ttk.Button(actions_frame, text="Actualizar Lista", command=self._refresh_entries_list).grid(row=0, column=3, padx=5)
    
    def _create_status_panel(self, parent: ttk.Widget) -> None:
        """Crear panel de estado y estadísticas."""
        status_frame = ttk.LabelFrame(parent, text="Estado del Sistema", padding="5")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Estado general
        ttk.Label(status_frame, text="Estado:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(status_frame, textvariable=self.status_var, style='Status.TLabel').grid(row=0, column=1, sticky=tk.W)
        
        # Estadísticas
        stats_frame = ttk.Frame(status_frame)
        stats_frame.grid(row=0, column=2, columnspan=2, sticky=tk.E)
        
        ttk.Label(stats_frame, text="Total Entradas:").grid(row=0, column=0, padx=(20, 5))
        ttk.Label(stats_frame, textvariable=self.entries_count_var, style='Status.TLabel').grid(row=0, column=1)
        
        ttk.Label(stats_frame, text="Capturados:").grid(row=0, column=2, padx=(20, 5))
        ttk.Label(stats_frame, textvariable=self.captured_count_var, style='Status.TLabel').grid(row=0, column=3)
        
        # Información del sistema
        ttk.Label(status_frame, text=f"Plataforma: {self.classifier_backend.upper()}").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(status_frame, text=f"Cámara: {'Disponible' if self.platform.platform_info['has_camera'] else 'No disponible'}").grid(row=1, column=1, sticky=tk.W)
    
    def _start_real_time(self) -> None:
        """Iniciar el modo de tiempo real."""
        if not self.camera.start():
            messagebox.showerror("Error", "No se pudo inicializar la cámara")
            return
        
        self.is_running = True
        self.status_var.set("Ejecutándose en tiempo real...")
        
        # Actualizar botones
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.capture_button.config(state='normal')
        
        # Iniciar hilos de procesamiento
        self._start_video_thread()
        self._start_prediction_thread()
        
        logging.info("Modo tiempo real iniciado")
    
    def _stop_real_time(self) -> None:
        """Detener el modo de tiempo real."""
        self.is_running = False
        self.camera.stop()
        
        self.status_var.set("Detenido")
        self.detection_var.set("Sin detección")
        self.confidence_var.set("0%")
        
        # Actualizar botones
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.capture_button.config(state='disabled')
        
        logging.info("Modo tiempo real detenido")
    
    def _start_video_thread(self) -> None:
        """Iniciar hilo de actualización de video."""
        def video_loop():
            while self.is_running:
                frame = self.camera.get_frame()
                if frame is not None:
                    self.current_frame = frame
                    self._update_video_display(frame)
                time.sleep(VIDEO_REFRESH_MS / 1000.0)
        
        video_thread = threading.Thread(target=video_loop, daemon=True)
        video_thread.start()
    
    def _start_prediction_thread(self) -> None:
        """Iniciar hilo de predicción en tiempo real."""
        def prediction_loop():
            while self.is_running:
                if self.current_frame is not None:
                    current_time = time.time()
                    
                    # Limitar frecuencia de predicciones
                    if current_time - self.last_prediction_time >= (PREDICTION_INTERVAL_MS / 1000.0):
                        self._process_frame_prediction(self.current_frame.copy())
                        self.last_prediction_time = current_time
                
                time.sleep(0.1)  # Sleep corto para no saturar CPU
        
        self.prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
        self.prediction_thread.start()
    
    def _update_video_display(self, frame: np.ndarray) -> None:
        """Actualizar la visualización de video."""
        try:
            # Redimensionar frame para display
            display_frame = cv2.resize(frame, (640, 480))
            
            # Añadir overlay de información si hay predicción
            if self.last_prediction:
                self._add_prediction_overlay(display_frame, self.last_prediction)
            
            # Convertir a formato Tkinter
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            tk_image = ImageTk.PhotoImage(pil_image)
            
            # Actualizar label (debe ejecutarse en main thread)
            self.root.after(0, lambda: self._update_video_label(tk_image))
            
        except Exception as e:
            logging.error(f"Error actualizando display de video: {e}")
    
    def _update_video_label(self, tk_image: ImageTk.PhotoImage) -> None:
        """Actualizar el label de video en el main thread."""
        self.video_label.configure(image=tk_image)
        self.video_label.image = tk_image  # Mantener referencia
    
    def _add_prediction_overlay(self, frame: np.ndarray, prediction_data: Dict[str, Any]) -> None:
        """Añadir overlay de información de predicción al frame."""
        animal_name = prediction_data.get('name', 'Desconocido')
        confidence = prediction_data.get('confidence', 0.0)
        bbox = prediction_data.get('bbox')
        
        # Dibujar bounding box si existe
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Información de texto
        text = f"{animal_name} ({confidence:.1%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Calcular tamaño del texto para el fondo
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Dibujar fondo del texto
        cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + baseline + 10), 
                     (0, 0, 0), -1)
        
        # Dibujar texto
        cv2.putText(frame, text, (15, 10 + text_height + 5), font, font_scale, (255, 255, 255), thickness)
    
    def _process_frame_prediction(self, frame: np.ndarray) -> None:
        """Procesar frame para predicción de animal."""
        try:
            with self.prediction_lock:
                # Preprocesar imagen
                processed_image = self.image_processor.preprocess_for_classification(frame)
                
                # Realizar predicción
                animal_name, confidence = self.classifier.predict(processed_image)
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Análisis visual adicional
                    visual_analysis = self._analyze_visual_features(frame)
                    
                    prediction_data = {
                        'name': animal_name,
                        'confidence': confidence,
                        'timestamp': time.time(),
                        'visual_analysis': visual_analysis,
                        'frame': frame
                    }
                    
                    self.last_prediction = prediction_data
                    
                    # Actualizar UI
                    self.root.after(0, lambda: self._update_detection_display(animal_name, confidence))
                    
                    # Auto-captura si confianza es muy alta
                    if confidence >= AUTO_CAPTURE_THRESHOLD:
                        self.root.after(0, lambda: self._auto_capture(prediction_data))
                else:
                    self.last_prediction = None
                    self.root.after(0, lambda: self._clear_detection_display())
                    
        except Exception as e:
            logging.error(f"Error en predicción de frame: {e}")
    
    def _analyze_visual_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analizar características visuales del frame."""
        try:
            # Detectar objetos y obtener bounding box principal
            bboxes = self.image_processor.detect_objects(frame)
            main_bbox = bboxes[0] if bboxes else None
            
            # Calcular color dominante
            dominant_color = self._get_dominant_color(frame, main_bbox)
            
            # Calcular tamaño relativo
            relative_size = self._calculate_relative_size(frame, main_bbox)
            
            return {
                'dominant_color': dominant_color,
                'relative_size': relative_size,
                'bbox': main_bbox,
                'frame_size': frame.shape[:2]
            }
        except Exception as e:
            logging.error(f"Error en análisis visual: {e}")
            return {}
    
    def _get_dominant_color(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """Obtener color dominante de la imagen o región."""
        try:
            # Usar región de bounding box si está disponible
            if bbox:
                x1, y1, x2, y2 = bbox
                region = frame[y1:y2, x1:x2]
            else:
                region = frame
            
            # Convertir a RGB
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para acelerar procesamiento
            region_small = cv2.resize(region_rgb, (50, 50))
            
            # Calcular histograma y encontrar color dominante
            pixels = region_small.reshape(-1, 3)
            
            # K-means para encontrar colores dominantes
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Color más frecuente
            dominant_rgb = kmeans.cluster_centers_[0].astype(int)
            
            # Convertir a nombre de color aproximado
            color_name = self._rgb_to_color_name(dominant_rgb)
            
            return {
                'rgb': dominant_rgb.tolist(),
                'hex': f"#{dominant_rgb[0]:02x}{dominant_rgb[1]:02x}{dominant_rgb[2]:02x}",
                'name': color_name
            }
        except Exception as e:
            logging.error(f"Error calculando color dominante: {e}")
            return {'rgb': [128, 128, 128], 'hex': '#808080', 'name': 'Gris'}
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Convertir RGB a nombre de color aproximado."""
        color_map = {
            'Rojo': [255, 0, 0],
            'Verde': [0, 255, 0],
            'Azul': [0, 0, 255],
            'Amarillo': [255, 255, 0],
            'Magenta': [255, 0, 255],
            'Cian': [0, 255, 255],
            'Naranja': [255, 165, 0],
            'Rosa': [255, 192, 203],
            'Morado': [128, 0, 128],
            'Marrón': [165, 42, 42],
            'Negro': [0, 0, 0],
            'Blanco': [255, 255, 255],
            'Gris': [128, 128, 128]
        }
        
        min_distance = float('inf')
        closest_color = 'Gris'
        
        for color_name, color_rgb in color_map.items():
            distance = np.linalg.norm(rgb - np.array(color_rgb))
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        
        return closest_color
    
    def _calculate_relative_size(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> float:
        """Calcular tamaño relativo del objeto detectado."""
        if not bbox:
            return 0.0
        
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]
        
        return min(1.0, bbox_area / frame_area)
    
    def _update_detection_display(self, animal_name: str, confidence: float) -> None:
        """Actualizar display de detección en UI."""
        self.detection_var.set(animal_name)
        self.confidence_var.set(f"{confidence:.1%}")
    
    def _clear_detection_display(self) -> None:
        """Limpiar display de detección."""
        self.detection_var.set("Sin detección")
        self.confidence_var.set("0%")
    
    def _auto_capture(self, prediction_data: Dict[str, Any]) -> None:
        """Captura automática cuando la confianza es muy alta."""
        # Verificar si ya existe una entrada reciente del mismo animal
        recent_entries = self.pokedx_repo.search_by_name(prediction_data['name'], limit=1)
        
        # Solo auto-capturar si no hay entradas recientes (menos de 30 segundos)
        if not recent_entries or (time.time() - recent_entries[0].timestamp > 30):
            self._create_pokedx_entry(prediction_data, auto_captured=True)
            self.status_var.set(f"Auto-capturado: {prediction_data['name']}")
            logging.info(f"Auto-captura realizada: {prediction_data['name']}")
    
    def _manual_capture(self) -> None:
        """Captura manual del animal actual."""
        if not self.last_prediction:
            messagebox.showwarning("Aviso", "No hay detección activa para capturar")
            return
        
        self._create_pokedx_entry(self.last_prediction, auto_captured=False)
        messagebox.showinfo("Éxito", f"Animal capturado: {self.last_prediction['name']}")
        logging.info(f"Captura manual realizada: {self.last_prediction['name']}")
    
    def _create_pokedx_entry(self, prediction_data: Dict[str, Any], auto_captured: bool = False) -> None:
        """Crear entrada en la Pokédx."""
        try:
            animal_name = prediction_data['name']
            confidence = prediction_data['confidence']
            timestamp = prediction_data['timestamp']
            visual_analysis = prediction_data.get('visual_analysis', {})
            frame = prediction_data['frame']
            
            # Guardar snapshot
            snapshot_filename = f"{animal_name}_{int(timestamp)}.jpg"
            snapshot_path = SNAPSHOT_DIR / snapshot_filename
            cv2.imwrite(str(snapshot_path), frame)
            
            # Obtener información adicional del animal
            animal_info = self._get_animal_info_async(animal_name)
            
            # Preparar datos visuales
            dominant_color = visual_analysis.get('dominant_color', {})
            bbox = visual_analysis.get('bbox')
            
            # Crear entrada de Pokédx
            entry = PokedexEntry(
                id=None,
                timestamp=timestamp,
                name=animal_name,
                confidence=confidence,
                summary=animal_info.get('summary', ''),
                habitat=animal_info.get('habitat', ''),
                diet=animal_info.get('diet', ''),
                characteristics=animal_info.get('characteristics', ''),
                conservation_status=animal_info.get('conservation_status', ''),
                scientific_name=animal_info.get('scientific_name', ''),
                source_url=animal_info.get('source_url', ''),
                image_path=str(snapshot_path),
                nickname=None,  # Usuario puede establecer después
                captured=1 if not auto_captured else 0,  # Captura manual = capturado
                notes=f"{'Auto-capturado' if auto_captured else 'Capturado manualmente'}",
                dominant_color=dominant_color.get('name', ''),
                dominant_color_rgb=','.join(map(str, dominant_color.get('rgb', []))),
                relative_size=visual_analysis.get('relative_size', 0.0),
                bbox=','.join(map(str, bbox)) if bbox else None,
                features_json=json.dumps(visual_analysis)
            )
            
            # Guardar en base de datos
            entry_id = self.pokedx_repo.add_entry(entry)
            logging.info(f"Entrada #{entry_id} creada para {animal_name}")
            
            # Actualizar UI
            self._refresh_entries_list()
            self._update_counters()
            
        except Exception as e:
            logging.error(f"Error creando entrada de Pokédx: {e}")
            messagebox.showerror("Error", f"Error guardando captura: {e}")
    
    def _get_animal_info_async(self, animal_name: str) -> Dict[str, Any]:
        """Obtener información del animal de forma asíncrona."""
        try:
            # Ejecutar en hilo separado para no bloquear UI
            info = self.animal_api.get_animal_info(animal_name)
            return info if info else {}
        except Exception as e:
            logging.error(f"Error obteniendo información de {animal_name}: {e}")
            return {}
    
    def _refresh_entries_list(self) -> None:
        """Actualizar lista de entradas en la UI."""
        try:
            # Limpiar lista actual
            for item in self.entries_tree.get_children():
                self.entries_tree.delete(item)
            
            # Obtener entradas más recientes
            entries = self.pokedx_repo.list_entries(limit=50)
            
            # Llenar treeview
            for entry in entries:
                date_str = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M')
                status = "Capturado" if entry.captured else "Visto"
                
                self.entries_tree.insert('', 'end', values=(
                    entry.id,
                    entry.name,
                    date_str,
                    f"{entry.confidence:.1%}",
                    status
                ))
        
        except Exception as e:
            logging.error(f"Error actualizando lista de entradas: {e}")
    
    def _update_counters(self) -> None:
        """Actualizar contadores de estadísticas."""
        try:
            entries = self.pokedx_repo.list_entries(limit=1000)  # Obtener todas para contar
            total_count = len(entries)
            captured_count = sum(1 for entry in entries if entry.captured)
            
            self.entries_count_var.set(str(total_count))
            self.captured_count_var.set(str(captured_count))
        
        except Exception as e:
            logging.error(f"Error actualizando contadores: {e}")
    
    def _on_search_change(self, event: tk.Event) -> None:
        """Manejar cambios en el campo de búsqueda."""
        # Búsqueda en tiempo real con delay
        self.root.after(500, self._search_entries)
    
    def _search_entries(self) -> None:
        """Buscar entradas por nombre."""
        search_term = self.search_var.get().strip()
        
        try:
            # Limpiar lista actual
            for item in self.entries_tree.get_children():
                self.entries_tree.delete(item)
            
            if search_term:
                # Buscar por nombre
                entries = self.pokedx_repo.search_by_name(search_term, limit=50)
            else:
                # Mostrar todas las entradas recientes
                entries = self.pokedx_repo.list_entries(limit=50)
            
            # Llenar treeview con resultados
            for entry in entries:
                date_str = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M')
                status = "Capturado" if entry.captured else "Visto"
                
                self.entries_tree.insert('', 'end', values=(
                    entry.id,
                    entry.name,
                    date_str,
                    f"{entry.confidence:.1%}",
                    status
                ))
        
        except Exception as e:
            logging.error(f"Error en búsqueda: {e}")
    
    def _on_entry_double_click(self, event: tk.Event) -> None:
        """Manejar doble clic en entrada."""
        self._show_entry_detail()
    
    def _show_entry_detail(self) -> None:
        """Mostrar vista de detalle de la entrada seleccionada."""
        selection = self.entries_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecciona una entrada para ver detalles")
            return
        
        item = self.entries_tree.item(selection[0])
        entry_id = item['values'][0]
        
        try:
            entry = self.pokedx_repo.get_entry_by_id(entry_id)
            if entry:
                self._open_detail_window(entry)
        except Exception as e:
            logging.error(f"Error mostrando detalle: {e}")
            messagebox.showerror("Error", f"Error cargando detalle: {e}")
    
    def _open_detail_window(self, entry: PokedexEntry) -> None:
        """Abrir ventana de detalle de entrada."""
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Detalle - {entry.name}")
        detail_window.geometry("800x600")
        detail_window.resizable(True, True)
        
        # Frame principal con scroll
        canvas = tk.Canvas(detail_window)
        scrollbar = ttk.Scrollbar(detail_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Layout
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Contenido de detalle
        self._populate_detail_content(scrollable_frame, entry)
    
    def _populate_detail_content(self, parent: ttk.Frame, entry: PokedexEntry) -> None:
        """Poblar contenido de la ventana de detalle."""
        # Título
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(title_frame, text=entry.name, style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(title_frame, text=f"#{entry.id}", style='Subtitle.TLabel').pack(side=tk.RIGHT)
        
        # Imagen si existe
        if entry.image_path and os.path.exists(entry.image_path):
            img_frame = ttk.LabelFrame(parent, text="Imagen Capturada", padding="10")
            img_frame.pack(fill=tk.X, padx=10, pady=5)
            
            try:
                # Cargar y redimensionar imagen
                pil_image = Image.open(entry.image_path)
                pil_image.thumbnail((400, 300))
                tk_image = ImageTk.PhotoImage(pil_image)
                
                img_label = ttk.Label(img_frame, image=tk_image)
                img_label.image = tk_image  # Mantener referencia
                img_label.pack()
            except Exception as e:
                ttk.Label(img_frame, text=f"Error cargando imagen: {e}").pack()
        
        # Información básica
        basic_frame = ttk.LabelFrame(parent, text="Información Básica", padding="10")
        basic_frame.pack(fill=tk.X, padx=10, pady=5)
        
        basic_info = [
            ("Fecha de detección", datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M:%S')),
            ("Confianza", f"{entry.confidence:.1%}"),
            ("Estado", "Capturado" if entry.captured else "Visto"),
            ("Nombre científico", entry.scientific_name or "No disponible"),
            ("Color dominante", entry.dominant_color or "No disponible"),
            ("Tamaño relativo", f"{(entry.relative_size or 0):.1%}" if entry.relative_size else "No disponible")
        ]
        
        for i, (label, value) in enumerate(basic_info):
            ttk.Label(basic_frame, text=f"{label}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 10))
            ttk.Label(basic_frame, text=str(value)).grid(row=i, column=1, sticky=tk.W)
        
        # Información detallada
        if entry.summary:
            detail_frame = ttk.LabelFrame(parent, text="Descripción", padding="10")
            detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            text_widget = tk.Text(detail_frame, height=6, wrap=tk.WORD)
            text_scrollbar = ttk.Scrollbar(detail_frame, command=text_widget.yview)
            text_widget.configure(yscrollcommand=text_scrollbar.set)
            
            text_widget.insert(tk.END, entry.summary)
            text_widget.config(state=tk.DISABLED)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Campos adicionales
        additional_info = [
            ("Hábitat", entry.habitat),
            ("Dieta", entry.diet),
            ("Características", entry.characteristics),
            ("Estado de conservación", entry.conservation_status),
            ("Notas", entry.notes)
        ]
        
        for label, value in additional_info:
            if value:
                info_frame = ttk.LabelFrame(parent, text=label, padding="10")
                info_frame.pack(fill=tk.X, padx=10, pady=2)
                ttk.Label(info_frame, text=value, wraplength=600).pack(anchor=tk.W)
        
        # Botones de acción
        actions_frame = ttk.Frame(parent)
        actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        if not entry.captured:
            ttk.Button(actions_frame, text="Marcar como Capturado", 
                      command=lambda: self._mark_entry_captured(entry.id)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(actions_frame, text="Editar Apodo", 
                  command=lambda: self._edit_nickname(entry.id)).pack(side=tk.LEFT, padx=5)
    
    def _mark_captured(self) -> None:
        """Marcar entrada seleccionada como capturada."""
        selection = self.entries_tree.selection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecciona una entrada para marcar como capturada")
            return
        
        item = self.entries_tree.item(selection[0])
        entry_id = item['values'][0]
        
        self._mark_entry_captured(entry_id)
    
    def _mark_entry_captured(self, entry_id: int) -> None:
        """Marcar una entrada específica como capturada."""
        try:
            success = self.pokedx_repo.mark_as_captured(entry_id)
            if success:
                messagebox.showinfo("Éxito", "Animal marcado como capturado")
                self._refresh_entries_list()
                self._update_counters()
            else:
                messagebox.showerror("Error", "No se pudo marcar como capturado")
        except Exception as e:
            logging.error(f"Error marcando como capturado: {e}")
            messagebox.showerror("Error", f"Error: {e}")
    
    def _edit_nickname(self, entry_id: int) -> None:
        """Editar apodo de una entrada."""
        try:
            entry = self.pokedx_repo.get_entry_by_id(entry_id)
            if not entry:
                return
            
            # Diálogo para ingresar apodo
            nickname = tk.simpledialog.askstring(
                "Editar Apodo", 
                f"Apodo para {entry.name}:",
                initialvalue=entry.nickname or ""
            )
            
            if nickname is not None:  # Usuario no canceló
                success = self.pokedx_repo.update_nickname(entry_id, nickname if nickname.strip() else None)
                if success:
                    messagebox.showinfo("Éxito", "Apodo actualizado")
                    self._refresh_entries_list()
        except Exception as e:
            logging.error(f"Error editando apodo: {e}")
            messagebox.showerror("Error", f"Error: {e}")
    
    def _export_entries(self) -> None:
        """Exportar entradas de Pokédx."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Exportar a JSON
            json_path = EXPORT_DIR / f"pokedex_export_{timestamp}.json"
            entries = self.pokedx_repo.list_entries(limit=1000)
            
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_entries': len(entries),
                'captured_count': sum(1 for e in entries if e.captured),
                'entries': [self._entry_to_dict(entry) for entry in entries]
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            # Exportar a Markdown
            md_path = EXPORT_DIR / f"pokedex_report_{timestamp}.md"
            self._export_to_markdown(entries, md_path)
            
            messagebox.showinfo("Éxito", f"Datos exportados:\n- {json_path}\n- {md_path}")
            
        except Exception as e:
            logging.error(f"Error exportando: {e}")
            messagebox.showerror("Error", f"Error exportando: {e}")
    
    def _entry_to_dict(self, entry: PokedexEntry) -> Dict[str, Any]:
        """Convertir entrada a diccionario para exportación."""
        return {
            'id': entry.id,
            'name': entry.name,
            'timestamp': entry.timestamp,
            'date': datetime.fromtimestamp(entry.timestamp).isoformat(),
            'confidence': entry.confidence,
            'captured': bool(entry.captured),
            'nickname': entry.nickname,
            'summary': entry.summary,
            'habitat': entry.habitat,
            'diet': entry.diet,
            'characteristics': entry.characteristics,
            'conservation_status': entry.conservation_status,
            'scientific_name': entry.scientific_name,
            'dominant_color': entry.dominant_color,
            'relative_size': entry.relative_size,
            'notes': entry.notes
        }
    
    def _export_to_markdown(self, entries: List[PokedexEntry], output_path: Path) -> None:
        """Exportar entradas a formato Markdown."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Pokédx Animal - Reporte de Entradas\n\n")
            f.write(f"**Fecha de exportación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total de entradas:** {len(entries)}\n")
            f.write(f"**Animales capturados:** {sum(1 for e in entries if e.captured)}\n\n")
            
            f.write("## Entradas\n\n")
            
            for entry in entries:
                f.write(f"### #{entry.id} - {entry.name}\n\n")
                f.write(f"- **Estado:** {'Capturado' if entry.captured else 'Visto'}\n")
                f.write(f"- **Fecha:** {datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"- **Confianza:** {entry.confidence:.1%}\n")
                
                if entry.nickname:
                    f.write(f"- **Apodo:** {entry.nickname}\n")
                if entry.scientific_name:
                    f.write(f"- **Nombre científico:** {entry.scientific_name}\n")
                if entry.dominant_color:
                    f.write(f"- **Color dominante:** {entry.dominant_color}\n")
                if entry.relative_size:
                    f.write(f"- **Tamaño relativo:** {entry.relative_size:.1%}\n")
                
                if entry.summary:
                    f.write(f"\n**Descripción:** {entry.summary}\n")
                
                f.write("\n---\n\n")
    
    def _on_closing(self) -> None:
        """Manejar cierre de aplicación."""
        if self.is_running:
            self._stop_real_time()
        
        logging.info("Cerrando aplicación Pokédx Animal")
        self.root.destroy()
    
    def run(self) -> None:
        """Ejecutar la aplicación."""
        # Cargar lista inicial de entradas
        self._refresh_entries_list()
        
        # Mostrar mensaje de bienvenida
        self.status_var.set("Listo para iniciar")
        
        logging.info("Pokédx Animal iniciada")
        
        # Ejecutar loop principal de Tkinter
        self.root.mainloop()


def main() -> None:
    """Función principal de la aplicación."""
    print("=" * 60)
    print("POKEDEX ANIMAL - TIEMPO REAL")
    print("Version Profesional Optimizada para Raspberry Pi")
    print("=" * 60)
    
    try:
        app = RealTimePokedexApp()
        app.run()
    except KeyboardInterrupt:
        print("\nAplicación interrumpida por usuario")
    except Exception as e:
        print(f"Error fatal: {e}")
        logging.error(f"Error fatal: {e}")
    finally:
        print("Pokédx Animal terminada")


if __name__ == "__main__":
    main()