#!/usr/bin/env python3
"""
Pokedex Animal Ultra - Version Windows Profesional
Sistema avanzado de reconocimiento de fauna con IA de ultima generacion

Caracteristicas:
- Interfaz futurista con CustomTkinter
- IA multi-modelo con ensemble learning
- Procesamiento en tiempo real optimizado GPU
- Sistema de logros y estadisticas avanzadas
- Visualizaciones dinamicas y animaciones fluidas
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import queue
import sqlite3
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Deque

import cv2
import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageDraw, ImageFilter, ImageTk
from scipy import ndimage
from skimage import color, exposure, filters
from tensorflow import keras
from ultralytics import YOLO

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from utils.camera import CameraCapture
from utils.image_processing import ImageProcessor
from utils.api import AnimalInfoAPI
from utils.gpu_detector import GPUDetector
from model.animal_classifier import AnimalClassifier

SNAPSHOT_DIR = Path("data/snapshots_ultra")
EXPORT_DIR = Path("data/exports_ultra")
LOGS_DIR = Path("data/logs_ultra")
MODELS_DIR = Path("model/ultra")
CACHE_DIR = Path("data/cache")

for directory in [SNAPSHOT_DIR, EXPORT_DIR, LOGS_DIR, MODELS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

APP_TITLE = "Pokedex Animal Ultra - Windows Edition"
WINDOW_SIZE = "1920x1080"
VIDEO_FPS_TARGET = 60
PREDICTION_FPS_TARGET = 10
CONFIDENCE_THRESHOLD = 0.45
ENSEMBLE_THRESHOLD = 0.65

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'ultra_pokedex.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Resultado de prediccion del sistema de IA."""
    species_name: str
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    features: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    model_source: str = "ensemble"
    processing_time_ms: float = 0.0


@dataclass
class SystemMetrics:
    """Metricas del sistema en tiempo real."""
    fps_video: float = 0.0
    fps_prediction: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    frame_count: int = 0
    prediction_count: int = 0
    average_confidence: float = 0.0


class FrameBuffer:
    """Buffer circular optimizado para frames de video."""
    
    def __init__(self, maxsize: int = 30):
        self.maxsize = maxsize
        self.buffer: Deque[np.ndarray] = deque(maxlen=maxsize)
        self.lock = threading.Lock()
        
    def put(self, frame: np.ndarray) -> None:
        with self.lock:
            self.buffer.append(frame.copy())
            
    def get(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.buffer:
                return self.buffer[-1].copy()
            return None
            
    def get_batch(self, n: int) -> List[np.ndarray]:
        with self.lock:
            return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)
            
    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()


class EnsembleAIEngine:
    """Motor de IA con ensemble de multiples modelos para maxima precision."""
    
    def __init__(self):
        # Detectar y configurar GPU (AMD/NVIDIA)
        self.gpu_detector = GPUDetector()
        gpu_info = self.gpu_detector.get_device_info()
        
        logger.info("=" * 60)
        logger.info("GPU CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"GPU Type: {gpu_info['type']}")
        logger.info(f"GPU Name: {gpu_info['name']}")
        logger.info(f"Device: {gpu_info['device']}")
        logger.info(f"CUDA Available: {gpu_info['cuda_available']}")
        logger.info(f"ROCm Available: {gpu_info['rocm_available']}")
        logger.info("=" * 60)
        
        # Configurar frameworks
        self.gpu_detector.configure_tensorflow()
        self.device = self.gpu_detector.configure_pytorch()
        
        logger.info(f"AI Engine initialized on device: {self.device}")
        
        self.models: Dict[str, Any] = {}
        self.model_weights: Dict[str, float] = {
            "yolo": 0.4,
            "efficientnet": 0.35,
            "mobilenet": 0.25
        }
        
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.cache_ttl = 30
        
        self._load_models()
        
    def _load_models(self) -> None:
        """Cargar todos los modelos del ensemble."""
        try:
            yolo_path = MODELS_DIR / "yolov8x.pt"
            if yolo_path.exists():
                self.models["yolo"] = YOLO(str(yolo_path))
                logger.info("YOLOv8 model loaded successfully")
            else:
                logger.warning(f"YOLOv8 model not found at {yolo_path}")
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            
        try:
            efficientnet_path = MODELS_DIR / "efficientnet_b7.h5"
            if efficientnet_path.exists():
                self.models["efficientnet"] = keras.models.load_model(str(efficientnet_path))
                logger.info("EfficientNet model loaded successfully")
            else:
                logger.warning(f"EfficientNet model not found at {efficientnet_path}")
                
        except Exception as e:
            logger.error(f"Failed to load EfficientNet model: {e}")
            
        try:
            self.models["mobilenet"] = AnimalClassifier()
            logger.info("MobileNet model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MobileNet model: {e}")
            
        if not self.models:
            raise RuntimeError("No models could be loaded. Check model files.")
            
    def predict(self, frame: np.ndarray) -> PredictionResult:
        """Realizar prediccion usando ensemble de modelos."""
        start_time = time.time()
        
        cache_key = self._generate_cache_key(frame)
        if cache_key in self.prediction_cache:
            cached = self.prediction_cache[cache_key]
            if (datetime.now() - cached.timestamp).seconds < self.cache_ttl:
                return cached
                
        predictions: List[Tuple[str, float, Optional[Tuple]]] = []
        
        if "yolo" in self.models:
            try:
                yolo_pred = self._predict_yolo(frame)
                if yolo_pred:
                    predictions.append(yolo_pred)
            except Exception as e:
                logger.error(f"YOLO prediction failed: {e}")
                
        if "efficientnet" in self.models:
            try:
                eff_pred = self._predict_efficientnet(frame)
                if eff_pred:
                    predictions.append(eff_pred)
            except Exception as e:
                logger.error(f"EfficientNet prediction failed: {e}")
                
        if "mobilenet" in self.models:
            try:
                mobile_pred = self._predict_mobilenet(frame)
                if mobile_pred:
                    predictions.append(mobile_pred)
            except Exception as e:
                logger.error(f"MobileNet prediction failed: {e}")
                
        if not predictions:
            return PredictionResult(
                species_name="Desconocido",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
        final_pred = self._ensemble_predictions(predictions)
        
        processing_time = (time.time() - start_time) * 1000
        
        features = self._extract_visual_features(frame)
        
        result = PredictionResult(
            species_name=final_pred[0],
            confidence=final_pred[1],
            bounding_box=final_pred[2],
            features=features,
            processing_time_ms=processing_time,
            model_source="ensemble"
        )
        
        self.prediction_cache[cache_key] = result
        
        return result
        
    def _predict_yolo(self, frame: np.ndarray) -> Optional[Tuple[str, float, Tuple]]:
        """Prediccion con YOLOv8."""
        results = self.models["yolo"](frame, verbose=False)
        
        if not results or not results[0].boxes:
            return None
            
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        
        cls_id = int(boxes.cls[best_idx])
        confidence = float(boxes.conf[best_idx])
        bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        
        class_name = self.models["yolo"].names[cls_id]
        
        return (class_name, confidence * self.model_weights["yolo"], tuple(bbox))
        
    def _predict_efficientnet(self, frame: np.ndarray) -> Optional[Tuple[str, float, Optional[Tuple]]]:
        """Prediccion con EfficientNet con deteccion de objetos."""
        img_resized = cv2.resize(frame, (600, 600))
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = img_array / 255.0
        
        predictions = self.models["efficientnet"].predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        class_names = self._load_class_names("efficientnet")
        class_name = class_names.get(class_idx, "Desconocido")
        
        bbox = self._detect_object_opencv(frame)
        
        return (class_name, confidence * self.model_weights["efficientnet"], bbox)
        
    def _predict_mobilenet(self, frame: np.ndarray) -> Optional[Tuple[str, float, Optional[Tuple]]]:
        """Prediccion con MobileNet con deteccion de objetos via OpenCV."""
        name, confidence = self.models["mobilenet"].predict(frame)
        
        if confidence > 0:
            bbox = self._detect_object_opencv(frame)
            return (name, confidence * self.model_weights["mobilenet"], bbox)
        return None
        
    def _detect_object_opencv(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detectar objeto principal usando OpenCV cuando YOLO no esta disponible."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                height, width = frame.shape[:2]
                margin = 50
                return (margin, margin, width - margin, height - margin)
                
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            frame_area = frame.shape[0] * frame.shape[1]
            if area < frame_area * 0.05 or area > frame_area * 0.95:
                height, width = frame.shape[:2]
                margin = 50
                return (margin, margin, width - margin, height - margin)
                
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            return (x1, y1, x2, y2)
            
        except Exception as e:
            logger.warning(f"Object detection fallback failed: {e}")
            height, width = frame.shape[:2]
            margin = 50
            return (margin, margin, width - margin, height - margin)
        
    def _ensemble_predictions(
        self, 
        predictions: List[Tuple[str, float, Optional[Tuple]]]
    ) -> Tuple[str, float, Optional[Tuple]]:
        """Combinar predicciones de multiples modelos."""
        species_scores: Dict[str, List[float]] = {}
        best_bbox = None
        max_conf = 0.0
        
        for species, conf, bbox in predictions:
            if species not in species_scores:
                species_scores[species] = []
            species_scores[species].append(conf)
            
            if bbox and conf > max_conf:
                best_bbox = bbox
                max_conf = conf
                
        if not species_scores:
            return ("Desconocido", 0.0, None)
            
        ensemble_scores = {
            species: np.mean(scores) 
            for species, scores in species_scores.items()
        }
        
        best_species = max(ensemble_scores, key=ensemble_scores.get)
        best_confidence = ensemble_scores[best_species]
        
        return (best_species, best_confidence, best_bbox)
        
    def _extract_visual_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """Extraer caracteristicas visuales avanzadas del frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
        
        dominant_hue = np.argmax(hist_h)
        dominant_color = self._hue_to_color_name(dominant_hue)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        brightness = np.mean(hsv[:, :, 2])
        saturation = np.mean(hsv[:, :, 1])
        
        return {
            "dominant_color": dominant_color,
            "brightness": float(brightness),
            "saturation": float(saturation),
            "edge_density": float(edge_density),
            "texture_complexity": float(np.std(gray))
        }
        
    @staticmethod
    def _hue_to_color_name(hue: int) -> str:
        """Convertir valor de hue a nombre de color."""
        if hue < 15 or hue >= 165:
            return "Rojo"
        elif 15 <= hue < 45:
            return "Naranja"
        elif 45 <= hue < 75:
            return "Amarillo"
        elif 75 <= hue < 105:
            return "Verde"
        elif 105 <= hue < 135:
            return "Cian"
        else:
            return "Azul"
            
    @staticmethod
    def _generate_cache_key(frame: np.ndarray) -> str:
        """Generar clave de cache basada en contenido del frame."""
        small = cv2.resize(frame, (32, 32))
        return str(hash(small.tobytes()))
        
    @staticmethod
    def _load_class_names(model_type: str) -> Dict[int, str]:
        """Cargar nombres de clases para un modelo."""
        labels_file = MODELS_DIR / f"{model_type}_labels.json"
        if labels_file.exists():
            with open(labels_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}


class UltraPokedexDatabase:
    """Sistema de base de datos avanzado con indices y full-text search."""
    
    def __init__(self, db_path: str = "data/pokedex_ultra.db"):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_database()
        
    def _initialize_database(self) -> None:
        """Inicializar esquema de base de datos optimizado."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000")
        
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS species (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                scientific_name TEXT,
                family TEXT,
                habitat TEXT,
                diet TEXT,
                conservation_status TEXT,
                description TEXT,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_sightings INTEGER DEFAULT 0,
                captured BOOLEAN DEFAULT 0,
                nickname TEXT,
                user_notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence FLOAT NOT NULL,
                image_path TEXT,
                bounding_box TEXT,
                visual_features TEXT,
                processing_time_ms FLOAT,
                model_source TEXT,
                location TEXT,
                FOREIGN KEY (species_id) REFERENCES species(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS achievements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                category TEXT,
                unlocked BOOLEAN DEFAULT 0,
                unlock_timestamp TIMESTAMP,
                progress INTEGER DEFAULT 0,
                target INTEGER DEFAULT 100,
                icon TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_detections INTEGER DEFAULT 0,
                total_species_discovered INTEGER DEFAULT 0,
                total_species_captured INTEGER DEFAULT 0,
                session_count INTEGER DEFAULT 0,
                total_playtime_seconds INTEGER DEFAULT 0,
                level INTEGER DEFAULT 1,
                experience INTEGER DEFAULT 0,
                favorite_species_id INTEGER,
                FOREIGN KEY (favorite_species_id) REFERENCES species(id)
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_species_name ON species(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_species ON sightings(species_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sightings_timestamp ON sightings(timestamp)")
        
        cursor.execute("""
            INSERT OR IGNORE INTO user_stats (id) VALUES (1)
        """)
        
        self._initialize_achievements()
        
        self.conn.commit()
        
    def _initialize_achievements(self) -> None:
        """Inicializar sistema de logros."""
        achievements = [
            ("Primer Avistamiento", "Detecta tu primer animal", "milestone", 1),
            ("Explorador Novato", "Descubre 5 especies diferentes", "collection", 5),
            ("Naturalista", "Descubre 10 especies diferentes", "collection", 10),
            ("Biologo Experto", "Descubre 25 especies diferentes", "collection", 25),
            ("Pokedex Completa", "Descubre 50 especies diferentes", "collection", 50),
            ("Cazador Preciso", "Logra 10 detecciones con >90% confianza", "precision", 10),
            ("Ojo de Aguila", "Logra 25 detecciones con >95% confianza", "precision", 25),
            ("Maratonista", "Acumula 1 hora de tiempo de sesion", "playtime", 3600),
            ("Dedicado", "Acumula 5 horas de tiempo de sesion", "playtime", 18000),
            ("Obsesionado", "Acumula 10 horas de tiempo de sesion", "playtime", 36000),
        ]
        
        cursor = self.conn.cursor()
        for name, desc, category, target in achievements:
            cursor.execute("""
                INSERT OR IGNORE INTO achievements (name, description, category, target)
                VALUES (?, ?, ?, ?)
            """, (name, desc, category, target))
        self.conn.commit()
        
    def add_sighting(
        self,
        species_name: str,
        confidence: float,
        image_path: Optional[str] = None,
        prediction_result: Optional[PredictionResult] = None
    ) -> int:
        """Registrar nuevo avistamiento."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT id FROM species WHERE name = ?", (species_name,))
        row = cursor.fetchone()
        
        if row:
            species_id = row[0]
            cursor.execute("""
                UPDATE species 
                SET last_seen = CURRENT_TIMESTAMP, total_sightings = total_sightings + 1
                WHERE id = ?
            """, (species_id,))
        else:
            cursor.execute("""
                INSERT INTO species (name, total_sightings)
                VALUES (?, 1)
            """, (species_name,))
            species_id = cursor.lastrowid
            
        bbox_json = json.dumps(prediction_result.bounding_box) if prediction_result and prediction_result.bounding_box else None
        features_json = json.dumps(prediction_result.features) if prediction_result else None
        
        cursor.execute("""
            INSERT INTO sightings (
                species_id, confidence, image_path, bounding_box, 
                visual_features, processing_time_ms, model_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            species_id, 
            confidence, 
            image_path, 
            bbox_json, 
            features_json,
            prediction_result.processing_time_ms if prediction_result else 0.0,
            prediction_result.model_source if prediction_result else "unknown"
        ))
        
        cursor.execute("""
            UPDATE user_stats 
            SET total_detections = total_detections + 1
            WHERE id = 1
        """)
        
        self.conn.commit()
        self._check_achievements()
        
        return species_id
        
    def capture_species(self, species_name: str) -> bool:
        """Marcar especie como capturada."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE species SET captured = 1 WHERE name = ?
        """, (species_name,))
        
        if cursor.rowcount > 0:
            cursor.execute("""
                UPDATE user_stats 
                SET total_species_captured = total_species_captured + 1
                WHERE id = 1
            """)
            self.conn.commit()
            self._check_achievements()
            return True
        return False
        
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadisticas completas."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM user_stats WHERE id = 1")
        stats_row = cursor.fetchone()
        
        cursor.execute("SELECT COUNT(DISTINCT id) FROM species")
        total_species = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM achievements WHERE unlocked = 1")
        unlocked_achievements = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM achievements")
        total_achievements = cursor.fetchone()[0]
        
        return {
            "total_detections": stats_row[1],
            "total_species_discovered": total_species,
            "total_species_captured": stats_row[3],
            "session_count": stats_row[4],
            "total_playtime_seconds": stats_row[5],
            "level": stats_row[6],
            "experience": stats_row[7],
            "achievements_unlocked": unlocked_achievements,
            "total_achievements": total_achievements
        }
        
    def _check_achievements(self) -> List[str]:
        """Verificar y desbloquear logros."""
        cursor = self.conn.cursor()
        unlocked = []
        
        stats = self.get_statistics()
        
        achievement_checks = [
            ("Primer Avistamiento", stats["total_detections"] >= 1),
            ("Explorador Novato", stats["total_species_discovered"] >= 5),
            ("Naturalista", stats["total_species_discovered"] >= 10),
            ("Biologo Experto", stats["total_species_discovered"] >= 25),
            ("Pokedex Completa", stats["total_species_discovered"] >= 50),
        ]
        
        for achievement_name, condition in achievement_checks:
            if condition:
                cursor.execute("""
                    UPDATE achievements 
                    SET unlocked = 1, unlock_timestamp = CURRENT_TIMESTAMP
                    WHERE name = ? AND unlocked = 0
                """, (achievement_name,))
                
                if cursor.rowcount > 0:
                    unlocked.append(achievement_name)
                    
        self.conn.commit()
        return unlocked
        
    def close(self) -> None:
        """Cerrar conexion a base de datos."""
        if self.conn:
            self.conn.close()


class UltraPokedexUI(ctk.CTk):
    """Interfaz grafica futurista de ultima generacion."""
    
    def __init__(self):
        super().__init__()
        
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.ai_engine = EnsembleAIEngine()
        self.database = UltraPokedexDatabase()
        self.camera = CameraCapture()
        self.api = AnimalInfoAPI()
        
        self.frame_buffer = FrameBuffer(maxsize=60)
        self.prediction_queue: queue.Queue = queue.Queue(maxsize=10)
        
        self.current_prediction: Optional[PredictionResult] = None
        self.metrics = SystemMetrics()
        
        self.is_running = False
        self.video_thread: Optional[threading.Thread] = None
        self.prediction_thread: Optional[threading.Thread] = None
        self.metrics_thread: Optional[threading.Thread] = None
        
        self._setup_ui()
        self._start_threads()
        
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _setup_ui(self) -> None:
        """Configurar interfaz de usuario futurista."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self._create_sidebar()
        self._create_main_panel()
        self._create_info_panel()
        
    def _create_sidebar(self) -> None:
        """Crear barra lateral con controles."""
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(10, weight=1)
        
        logo_label = ctk.CTkLabel(
            self.sidebar,
            text="POKEDEX ULTRA",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        version_label = ctk.CTkLabel(
            self.sidebar,
            text="Windows Professional Edition",
            font=ctk.CTkFont(size=10)
        )
        version_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        self.capture_btn = ctk.CTkButton(
            self.sidebar,
            text="CAPTURAR",
            command=self._capture_specimen,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.capture_btn.grid(row=2, column=0, padx=20, pady=10)
        
        self.stats_btn = ctk.CTkButton(
            self.sidebar,
            text="ESTADISTICAS",
            command=self._show_statistics,
            height=40
        )
        self.stats_btn.grid(row=3, column=0, padx=20, pady=10)
        
        self.achievements_btn = ctk.CTkButton(
            self.sidebar,
            text="LOGROS",
            command=self._show_achievements,
            height=40
        )
        self.achievements_btn.grid(row=4, column=0, padx=20, pady=10)
        
        self.pokedex_btn = ctk.CTkButton(
            self.sidebar,
            text="POKEDEX",
            command=self._show_pokedex,
            height=40
        )
        self.pokedex_btn.grid(row=5, column=0, padx=20, pady=10)
        
        separator = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30")
        separator.grid(row=6, column=0, padx=20, pady=20, sticky="ew")
        
        metrics_label = ctk.CTkLabel(
            self.sidebar,
            text="METRICAS DEL SISTEMA",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        metrics_label.grid(row=7, column=0, padx=20, pady=(0, 10))
        
        self.fps_label = ctk.CTkLabel(
            self.sidebar,
            text="FPS Video: 0 | Pred: 0",
            font=ctk.CTkFont(size=10)
        )
        self.fps_label.grid(row=8, column=0, padx=20, pady=5)
        
        self.cpu_label = ctk.CTkLabel(
            self.sidebar,
            text="CPU: 0%",
            font=ctk.CTkFont(size=10)
        )
        self.cpu_label.grid(row=9, column=0, padx=20, pady=5)
        
        self.gpu_label = ctk.CTkLabel(
            self.sidebar,
            text="GPU: N/A",
            font=ctk.CTkFont(size=10)
        )
        self.gpu_label.grid(row=10, column=0, padx=20, pady=5)
        
    def _create_main_panel(self) -> None:
        """Crear panel principal con video y visualizaciones."""
        self.main_panel = ctk.CTkFrame(self, corner_radius=10)
        self.main_panel.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_panel.grid_rowconfigure(0, weight=3)
        self.main_panel.grid_rowconfigure(1, weight=1)
        self.main_panel.grid_columnconfigure(0, weight=1)
        
        self.video_frame = ctk.CTkFrame(self.main_panel)
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both")
        
        self.viz_frame = ctk.CTkFrame(self.main_panel)
        self.viz_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
    def _create_info_panel(self) -> None:
        """Crear panel de informacion detallada."""
        self.info_panel = ctk.CTkFrame(self, width=350, corner_radius=10)
        self.info_panel.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        self.info_panel.grid_rowconfigure(1, weight=1)
        
        info_title = ctk.CTkLabel(
            self.info_panel,
            text="INFORMACION DE ESPECIE",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        info_title.grid(row=0, column=0, padx=20, pady=20)
        
        self.info_textbox = ctk.CTkTextbox(
            self.info_panel,
            font=ctk.CTkFont(size=12),
            wrap="word"
        )
        self.info_textbox.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
    def _start_threads(self) -> None:
        """Iniciar threads de procesamiento."""
        self.is_running = True
        
        if self.camera.start():
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()
            
            self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
            self.prediction_thread.start()
            
            self.metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
            self.metrics_thread.start()
            
            logger.info("All processing threads started successfully")
        else:
            logger.error("Failed to start camera")
            
    def _video_loop(self) -> None:
        """Loop de captura y visualizacion de video."""
        frame_times = deque(maxlen=30)
        
        while self.is_running:
            start_time = time.time()
            
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
                
            self.frame_buffer.put(frame)
            
            display_frame = frame.copy()
            
            if self.current_prediction:
                display_frame = self._annotate_frame(display_frame, self.current_prediction)
                
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (960, 720))
            
            img = Image.fromarray(display_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.video_label.configure(image=img_tk)
            self.video_label.image = img_tk
            
            frame_times.append(time.time() - start_time)
            if len(frame_times) > 0:
                self.metrics.fps_video = 1.0 / np.mean(frame_times)
                
            self.metrics.frame_count += 1
            
            time.sleep(max(0, 1.0 / VIDEO_FPS_TARGET - (time.time() - start_time)))
            
    def _prediction_loop(self) -> None:
        """Loop de predicciones de IA."""
        pred_times = deque(maxlen=10)
        
        while self.is_running:
            start_time = time.time()
            
            frame = self.frame_buffer.get()
            if frame is None:
                time.sleep(0.1)
                continue
                
            try:
                prediction = self.ai_engine.predict(frame)
                
                if prediction.confidence >= CONFIDENCE_THRESHOLD:
                    self.current_prediction = prediction
                    self._update_info_panel(prediction)
                    
                    if prediction.confidence >= ENSEMBLE_THRESHOLD:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        species_safe = prediction.species_name.replace(" ", "_").replace("/", "-")
                        image_path = SNAPSHOT_DIR / f"{species_safe}_{timestamp}.jpg"
                        
                        annotated_frame = frame.copy()
                        if prediction.bounding_box:
                            x1, y1, x2, y2 = prediction.bounding_box
                            color = self._get_confidence_color(prediction.confidence)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                            
                            label = f"{prediction.species_name} {prediction.confidence:.1%}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                      cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
                        
                        cv2.imwrite(str(image_path), annotated_frame)
                        
                        self.database.add_sighting(
                            prediction.species_name,
                            prediction.confidence,
                            str(image_path),
                            prediction
                        )
                        
                        logger.info(f"AUTO-CAPTURED: {prediction.species_name} at {prediction.confidence:.1%} confidence")
                        
                        self.after(0, lambda: self._show_capture_notification(prediction.species_name, prediction.confidence))
                        
                self.metrics.prediction_count += 1
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                
            pred_times.append(time.time() - start_time)
            if len(pred_times) > 0:
                self.metrics.fps_prediction = 1.0 / np.mean(pred_times)
                
            time.sleep(max(0, 1.0 / PREDICTION_FPS_TARGET - (time.time() - start_time)))
            
    def _metrics_loop(self) -> None:
        """Loop de actualizacion de metricas."""
        while self.is_running:
            self.metrics.cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.ram_percent = psutil.virtual_memory().percent
            
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.metrics.gpu_percent = gpus[0].load * 100
                        self.metrics.gpu_memory_percent = gpus[0].memoryUtil * 100
                except Exception:
                    pass
                    
            self._update_metrics_display()
            
            time.sleep(1)
            
    def _update_metrics_display(self) -> None:
        """Actualizar visualizacion de metricas."""
        self.fps_label.configure(
            text=f"FPS Video: {self.metrics.fps_video:.1f} | Pred: {self.metrics.fps_prediction:.1f}"
        )
        self.cpu_label.configure(
            text=f"CPU: {self.metrics.cpu_percent:.1f}% | RAM: {self.metrics.ram_percent:.1f}%"
        )
        
        if GPU_AVAILABLE and self.metrics.gpu_percent > 0:
            self.gpu_label.configure(
                text=f"GPU: {self.metrics.gpu_percent:.1f}% | VRAM: {self.metrics.gpu_memory_percent:.1f}%"
            )
        else:
            self.gpu_label.configure(text="GPU: No disponible")
            
    def _annotate_frame(self, frame: np.ndarray, prediction: PredictionResult) -> np.ndarray:
        """Anotar frame con informacion futurista de alta calidad."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        
        # Overlay semi-transparente para mejor contraste
        overlay = annotated.copy()
        
        # Bounding box con efecto glow
        if prediction.bounding_box:
            x1, y1, x2, y2 = prediction.bounding_box
            
            # Calcular color segun confianza
            confidence_color = self._get_confidence_color(prediction.confidence)
            
            # Box principal
            cv2.rectangle(annotated, (x1, y1), (x2, y2), confidence_color, 3)
            
            # Esquinas destacadas (efecto futurista)
            corner_length = 20
            cv2.line(annotated, (x1, y1), (x1 + corner_length, y1), confidence_color, 5)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_length), confidence_color, 5)
            cv2.line(annotated, (x2, y1), (x2 - corner_length, y1), confidence_color, 5)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_length), confidence_color, 5)
            cv2.line(annotated, (x1, y2), (x1 + corner_length, y2), confidence_color, 5)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_length), confidence_color, 5)
            cv2.line(annotated, (x2, y2), (x2 - corner_length, y2), confidence_color, 5)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_length), confidence_color, 5)
            
            # Etiqueta arriba del bounding box
            label_text = f"{prediction.species_name}"
            conf_text = f"{prediction.confidence:.1%}"
            
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            thickness = 2
            
            (label_w, label_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale * 0.7, thickness - 1)
            
            # Fondo de etiqueta con gradiente simulado
            label_bg_y1 = max(0, y1 - label_h - 40)
            label_bg_y2 = y1 - 5
            label_bg_x1 = x1
            label_bg_x2 = x1 + max(label_w, conf_w) + 30
            
            cv2.rectangle(overlay, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            
            # Borde de etiqueta
            cv2.rectangle(annotated, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), confidence_color, 2)
            
            # Texto de especie
            cv2.putText(annotated, label_text, (x1 + 10, y1 - 25), font, font_scale, (255, 255, 255), thickness)
            
            # Texto de confianza
            cv2.putText(annotated, conf_text, (x1 + 10, y1 - 8), font, font_scale * 0.7, confidence_color, thickness - 1)
        
        # Panel de información HUD (Head-Up Display) en la parte superior
        hud_height = 120
        hud_overlay = annotated.copy()
        cv2.rectangle(hud_overlay, (0, 0), (width, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(hud_overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Linea separadora
        cv2.line(annotated, (0, hud_height), (width, hud_height), (0, 255, 255), 2)
        
        # Texto principal
        main_text = f"DETECCION: {prediction.species_name.upper()}"
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(annotated, main_text, (15, 30), font, 0.9, (0, 255, 255), 2)
        
        # Barra de confianza
        bar_x = 15
        bar_y = 50
        bar_width = 300
        bar_height = 20
        
        # Fondo de barra
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Relleno de barra segun confianza
        fill_width = int(bar_width * prediction.confidence)
        confidence_color = self._get_confidence_color(prediction.confidence)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), confidence_color, -1)
        
        # Borde de barra
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Texto de confianza
        conf_text = f"CONFIANZA: {prediction.confidence:.1%}"
        cv2.putText(annotated, conf_text, (bar_x, bar_y - 5), font, 0.5, (255, 255, 255), 1)
        
        # Informacion adicional
        info_y = 90
        info_items = [
            f"MODELO: {prediction.model_source.upper()}",
            f"TIEMPO: {prediction.processing_time_ms:.1f}ms",
            f"FPS: {self.metrics.fps_prediction:.1f}"
        ]
        
        x_offset = 15
        for item in info_items:
            cv2.putText(annotated, item, (x_offset, info_y), font, 0.45, (200, 200, 200), 1)
            x_offset += 250
        
        # Features en el lado derecho si existen
        if prediction.features and len(prediction.features) > 0:
            feature_x = width - 250
            feature_y = hud_height + 30
            
            # Fondo para features
            feature_bg_h = min(len(prediction.features) * 25 + 20, height - hud_height - 20)
            feature_overlay = annotated.copy()
            cv2.rectangle(feature_overlay, (feature_x - 10, hud_height + 10), (width - 10, hud_height + feature_bg_h), (0, 0, 0), -1)
            cv2.addWeighted(feature_overlay, 0.5, annotated, 0.5, 0, annotated)
            
            # Titulo de features
            cv2.putText(annotated, "FEATURES:", (feature_x, feature_y), font, 0.5, (0, 255, 255), 1)
            feature_y += 20
            
            # Lista de features
            for key, value in list(prediction.features.items())[:5]:  # Max 5 features
                if isinstance(value, float):
                    feature_text = f"{key[:12]}: {value:.2f}"
                else:
                    feature_text = f"{key[:12]}: {str(value)[:8]}"
                    
                cv2.putText(annotated, feature_text, (feature_x, feature_y), font, 0.4, (200, 200, 200), 1)
                feature_y += 22
        
        return annotated
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """Obtener color BGR segun nivel de confianza."""
        if confidence >= 0.9:
            return (0, 255, 0)  # Verde brillante
        elif confidence >= 0.75:
            return (0, 255, 255)  # Amarillo
        elif confidence >= 0.6:
            return (0, 165, 255)  # Naranja
        else:
            return (0, 0, 255)  # Rojo
    
    def _show_capture_notification(self, species_name: str, confidence: float) -> None:
        """Mostrar notificacion de captura automatica."""
        notification = ctk.CTkToplevel(self)
        notification.title("Captura Automatica")
        notification.geometry("400x200")
        notification.attributes("-topmost", True)
        notification.configure(fg_color=("#1a1a2e", "#0f0f1e"))
        
        icon_label = ctk.CTkLabel(
            notification,
            text="CAPTURA EXITOSA",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=("#00ff00", "#00ff00")
        )
        icon_label.pack(pady=20)
        
        species_label = ctk.CTkLabel(
            notification,
            text=f"Especie: {species_name}",
            font=ctk.CTkFont(size=18)
        )
        species_label.pack(pady=10)
        
        conf_label = ctk.CTkLabel(
            notification,
            text=f"Confianza: {confidence:.1%}",
            font=ctk.CTkFont(size=14)
        )
        conf_label.pack(pady=5)
        
        close_btn = ctk.CTkButton(
            notification,
            text="OK",
            command=notification.destroy,
            width=100
        )
        close_btn.pack(pady=20)
        
        notification.after(3000, notification.destroy)
        
    def _update_info_panel(self, prediction: PredictionResult) -> None:
        """Actualizar panel de informacion con datos de prediccion."""
        self.info_textbox.delete("1.0", "end")
        
        info_text = f"""ESPECIE DETECTADA

Nombre: {prediction.species_name}
Confianza: {prediction.confidence:.1%}
Tiempo de procesamiento: {prediction.processing_time_ms:.2f}ms
Modelo: {prediction.model_source}

CARACTERISTICAS VISUALES
"""
        
        if prediction.features:
            for key, value in prediction.features.items():
                if isinstance(value, float):
                    info_text += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
                else:
                    info_text += f"{key.replace('_', ' ').title()}: {value}\n"
                    
        self.info_textbox.insert("1.0", info_text)
        
    def _capture_specimen(self) -> None:
        """Capturar espe
cimen actual."""
        if self.current_prediction and self.current_prediction.confidence >= CONFIDENCE_THRESHOLD:
            success = self.database.capture_species(self.current_prediction.species_name)
            if success:
                logger.info(f"Captured species: {self.current_prediction.species_name}")
                
    def _show_statistics(self) -> None:
        """Mostrar ventana de estadisticas."""
        stats_window = ctk.CTkToplevel(self)
        stats_window.title("Estadisticas")
        stats_window.geometry("800x600")
        
        stats = self.database.get_statistics()
        
        stats_text = f"""
ESTADISTICAS GLOBALES

Total de Detecciones: {stats['total_detections']}
Especies Descubiertas: {stats['total_species_discovered']}
Especies Capturadas: {stats['total_species_captured']}
Nivel: {stats['level']}
Experiencia: {stats['experience']}
Logros Desbloqueados: {stats['achievements_unlocked']}/{stats['total_achievements']}
"""
        
        text_widget = ctk.CTkTextbox(stats_window, font=ctk.CTkFont(size=14))
        text_widget.pack(expand=True, fill="both", padx=20, pady=20)
        text_widget.insert("1.0", stats_text)
        
    def _show_achievements(self) -> None:
        """Mostrar ventana de logros con interfaz futurista."""
        achievements_window = ctk.CTkToplevel(self)
        achievements_window.title("🏆 Logros Desbloqueados")
        achievements_window.geometry("900x700")
        achievements_window.configure(fg_color=("#1a1a2e", "#0f0f1e"))
        
        # Header con gradiente
        header = ctk.CTkFrame(achievements_window, height=80, fg_color=("#16213e", "#0d1117"))
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)
        
        title_label = ctk.CTkLabel(
            header,
            text="🏆 LOGROS Y MEDALLAS",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=("#ffd700", "#ffed4e")
        )
        title_label.pack(pady=20)
        
        # Contenedor scrollable
        scroll_frame = ctk.CTkScrollableFrame(
            achievements_window,
            fg_color=("#1a1a2e", "#0f0f1e")
        )
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Obtener logros de la base de datos
        cursor = self.database.conn.cursor()
        cursor.execute("""
            SELECT name, description, category, unlocked, unlock_timestamp, progress, target
            FROM achievements
            ORDER BY unlocked DESC, category, name
        """)
        
        achievements_by_category = {}
        for row in cursor.fetchall():
            name, desc, category, unlocked, unlock_time, progress, target = row
            if category not in achievements_by_category:
                achievements_by_category[category] = []
            achievements_by_category[category].append({
                "name": name,
                "description": desc,
                "unlocked": bool(unlocked),
                "unlock_time": unlock_time,
                "progress": progress or 0,
                "target": target
            })
        
        # Iconos por categoria
        category_icons = {
            "milestone": "🎯",
            "collection": "📚",
            "precision": "🎯",
            "playtime": "⏱️"
        }
        
        category_names = {
            "milestone": "HITOS",
            "collection": "COLECCION",
            "precision": "PRECISION",
            "playtime": "DEDICACION"
        }
        
        # Renderizar logros por categoria
        for category, achievements in achievements_by_category.items():
            # Separador de categoria
            cat_frame = ctk.CTkFrame(scroll_frame, fg_color=("#16213e", "#0d1117"), corner_radius=10)
            cat_frame.pack(fill="x", pady=(10, 5))
            
            cat_label = ctk.CTkLabel(
                cat_frame,
                text=f"{category_icons.get(category, '🏆')} {category_names.get(category, category.upper())}",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=("#00d9ff", "#00bfff")
            )
            cat_label.pack(pady=10, padx=15, anchor="w")
            
            # Logros de la categoria
            for achievement in achievements:
                self._create_achievement_card(scroll_frame, achievement)
        
        # Footer con stats globales
        stats = self.database.get_statistics()
        footer = ctk.CTkFrame(achievements_window, height=60, fg_color=("#16213e", "#0d1117"))
        footer.pack(fill="x", padx=0, pady=0)
        footer.pack_propagate(False)
        
        stats_text = f"Desbloqueados: {stats['achievements_unlocked']}/{stats['total_achievements']} | Nivel: {stats['level']} | XP: {stats['experience']}"
        stats_label = ctk.CTkLabel(
            footer,
            text=stats_text,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#ffd700", "#ffed4e")
        )
        stats_label.pack(pady=15)
        
    def _create_achievement_card(self, parent: ctk.CTkFrame, achievement: Dict[str, Any]) -> None:
        """Crear tarjeta de logro con animacion."""
        unlocked = achievement["unlocked"]
        progress = achievement["progress"]
        target = achievement["target"]
        progress_pct = min(100, int((progress / target) * 100)) if target > 0 else 0
        
        # Card con efecto hover
        card_color = ("#2a2a4e", "#1a1a3e") if unlocked else ("#1e1e1e", "#0a0a0a")
        card = ctk.CTkFrame(parent, fg_color=card_color, corner_radius=8, border_width=2)
        card.configure(border_color=("#ffd700", "#ffed4e") if unlocked else ("#444", "#222"))
        card.pack(fill="x", pady=5, padx=10)
        
        # Contenido del card
        content_frame = ctk.CTkFrame(card, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=15, pady=12)
        
        # Icono y titulo
        header_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
        header_frame.pack(fill="x")
        
        icon = "✅" if unlocked else "🔒"
        title_text = f"{icon} {achievement['name']}"
        title_color = ("#00ff00", "#00dd00") if unlocked else ("#666", "#444")
        
        title_label = ctk.CTkLabel(
            header_frame,
            text=title_text,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=title_color,
            anchor="w"
        )
        title_label.pack(side="left")
        
        # Timestamp si esta desbloqueado
        if unlocked and achievement["unlock_time"]:
            time_label = ctk.CTkLabel(
                header_frame,
                text=f"🕐 {achievement['unlock_time']}",
                font=ctk.CTkFont(size=10),
                text_color=("#888", "#666")
            )
            time_label.pack(side="right")
        
        # Descripcion
        desc_label = ctk.CTkLabel(
            content_frame,
            text=achievement['description'],
            font=ctk.CTkFont(size=12),
            text_color=("#ccc", "#888"),
            anchor="w"
        )
        desc_label.pack(fill="x", pady=(5, 8))
        
        # Barra de progreso
        if not unlocked and target > 0:
            progress_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
            progress_frame.pack(fill="x")
            
            progress_bar = ctk.CTkProgressBar(
                progress_frame,
                height=8,
                corner_radius=4,
                progress_color=("#00d9ff", "#00bfff")
            )
            progress_bar.pack(side="left", fill="x", expand=True)
            progress_bar.set(progress_pct / 100)
            
            progress_label = ctk.CTkLabel(
                progress_frame,
                text=f"{progress}/{target}",
                font=ctk.CTkFont(size=10),
                text_color=("#888", "#666"),
                width=60
            )
            progress_label.pack(side="right", padx=(10, 0))
        
    def _show_pokedex(self) -> None:
        """Mostrar ventana de pokedex completa con grid de especies."""
        pokedex_window = ctk.CTkToplevel(self)
        pokedex_window.title("📖 Pokedex Animal")
        pokedex_window.geometry("1200x800")
        pokedex_window.configure(fg_color=("#1a1a2e", "#0f0f1e"))
        
        # Header
        header = ctk.CTkFrame(pokedex_window, height=80, fg_color=("#16213e", "#0d1117"))
        header.pack(fill="x")
        header.pack_propagate(False)
        
        title_label = ctk.CTkLabel(
            header,
            text="📖 POKEDEX ANIMAL - ESPECIES REGISTRADAS",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=("#00d9ff", "#00bfff")
        )
        title_label.pack(side="left", padx=30, pady=20)
        
        # Barra de busqueda y filtros
        search_frame = ctk.CTkFrame(pokedex_window, height=60, fg_color=("#16213e", "#0d1117"))
        search_frame.pack(fill="x", padx=20, pady=(10, 0))
        search_frame.pack_propagate(False)
        
        search_var = ctk.StringVar()
        search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="🔍 Buscar especie...",
            textvariable=search_var,
            font=ctk.CTkFont(size=14),
            height=40,
            width=300
        )
        search_entry.pack(side="left", padx=10, pady=10)
        
        filter_var = ctk.StringVar(value="all")
        filter_menu = ctk.CTkOptionMenu(
            search_frame,
            variable=filter_var,
            values=["Todas", "Capturadas", "No Capturadas"],
            font=ctk.CTkFont(size=12),
            height=40,
            width=150
        )
        filter_menu.pack(side="left", padx=5)
        
        # Contenedor scrollable para grid
        scroll_frame = ctk.CTkScrollableFrame(
            pokedex_window,
            fg_color=("#1a1a2e", "#0f0f1e")
        )
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Obtener especies
        cursor = self.database.conn.cursor()
        cursor.execute("""
            SELECT 
                id, name, scientific_name, family, total_sightings, 
                captured, first_seen, last_seen, nickname
            FROM species
            ORDER BY name
        """)
        
        species_list = []
        for row in cursor.fetchall():
            species_list.append({
                "id": row[0],
                "name": row[1],
                "scientific_name": row[2] or "Desconocido",
                "family": row[3] or "Desconocido",
                "total_sightings": row[4],
                "captured": bool(row[5]),
                "first_seen": row[6],
                "last_seen": row[7],
                "nickname": row[8]
            })
        
        def filter_and_display():
            """Filtrar y mostrar especies."""
            # Limpiar grid
            for widget in scroll_frame.winfo_children():
                widget.destroy()
            
            search_text = search_var.get().lower()
            filter_mode = filter_var.get()
            
            filtered_species = []
            for species in species_list:
                # Filtro de busqueda
                if search_text and search_text not in species["name"].lower():
                    continue
                
                # Filtro de captura
                if filter_mode == "Capturadas" and not species["captured"]:
                    continue
                elif filter_mode == "No Capturadas" and species["captured"]:
                    continue
                
                filtered_species.append(species)
            
            # Mostrar en grid (4 columnas)
            columns = 4
            for idx, species in enumerate(filtered_species):
                row = idx // columns
                col = idx % columns
                self._create_species_card(scroll_frame, species, row, col)
            
            # Mensaje si no hay resultados
            if not filtered_species:
                no_results = ctk.CTkLabel(
                    scroll_frame,
                    text="❌ No se encontraron especies",
                    font=ctk.CTkFont(size=16),
                    text_color=("#888", "#666")
                )
                no_results.grid(row=0, column=0, columnspan=4, pady=50)
        
        # Eventos de busqueda y filtro
        search_var.trace_add("write", lambda *args: filter_and_display())
        filter_var.trace_add("write", lambda *args: filter_and_display())
        
        # Footer con estadisticas
        footer = ctk.CTkFrame(pokedex_window, height=60, fg_color=("#16213e", "#0d1117"))
        footer.pack(fill="x")
        footer.pack_propagate(False)
        
        stats = self.database.get_statistics()
        captured_count = sum(1 for s in species_list if s["captured"])
        total_count = len(species_list)
        
        stats_text = f"📊 Especies Registradas: {total_count} | Capturadas: {captured_count} | Descubiertas: {stats['total_species_discovered']}"
        stats_label = ctk.CTkLabel(
            footer,
            text=stats_text,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#ffd700", "#ffed4e")
        )
        stats_label.pack(pady=15)
        
        # Mostrar inicial
        filter_and_display()
        
    def _create_species_card(self, parent: ctk.CTkScrollableFrame, species: Dict[str, Any], row: int, col: int) -> None:
        """Crear tarjeta de especie en el grid."""
        captured = species["captured"]
        
        # Card
        card_color = ("#2a2a4e", "#1a1a3e") if captured else ("#1e1e1e", "#0a0a0a")
        border_color = ("#00ff00", "#00dd00") if captured else ("#444", "#222")
        
        card = ctk.CTkFrame(
            parent,
            fg_color=card_color,
            corner_radius=10,
            border_width=2,
            border_color=border_color,
            width=260,
            height=180
        )
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        card.grid_propagate(False)
        
        # Configurar peso de columnas
        parent.grid_columnconfigure(col, weight=1)
        
        # Contenido
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=12, pady=12)
        
        # Icono y estado
        status_icon = "✅" if captured else "❓"
        status_color = ("#00ff00", "#00dd00") if captured else ("#888", "#666")
        
        status_label = ctk.CTkLabel(
            content,
            text=status_icon,
            font=ctk.CTkFont(size=24),
            text_color=status_color
        )
        status_label.pack(anchor="ne")
        
        # Nombre
        name_text = species["nickname"] if species["nickname"] else species["name"]
        name_label = ctk.CTkLabel(
            content,
            text=name_text,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#ffffff", "#dddddd")
        )
        name_label.pack(pady=(0, 5))
        
        # Nombre cientifico
        sci_label = ctk.CTkLabel(
            content,
            text=species["scientific_name"],
            font=ctk.CTkFont(size=10, slant="italic"),
            text_color=("#888", "#666")
        )
        sci_label.pack()
        
        # Familia
        family_label = ctk.CTkLabel(
            content,
            text=f"🏛️ {species['family']}",
            font=ctk.CTkFont(size=11),
            text_color=("#00d9ff", "#00bfff")
        )
        family_label.pack(pady=(10, 5))
        
        # Estadisticas
        stats_frame = ctk.CTkFrame(content, fg_color="transparent")
        stats_frame.pack(fill="x", pady=(10, 0))
        
        sightings_label = ctk.CTkLabel(
            stats_frame,
            text=f"👁️ {species['total_sightings']}",
            font=ctk.CTkFont(size=10),
            text_color=("#ccc", "#888")
        )
        sightings_label.pack(side="left")
        
        # Boton de detalles
        def show_details():
            self._show_species_details(species)
        
        details_btn = ctk.CTkButton(
            stats_frame,
            text="ℹ️",
            width=30,
            height=24,
            font=ctk.CTkFont(size=12),
            fg_color=("#16213e", "#0d1117"),
            hover_color=("#00d9ff", "#00bfff"),
            command=show_details
        )
        details_btn.pack(side="right")
        
    def _show_species_details(self, species: Dict[str, Any]) -> None:
        """Mostrar ventana de detalles de especie."""
        details_window = ctk.CTkToplevel(self)
        details_window.title(f"📋 {species['name']}")
        details_window.geometry("600x500")
        details_window.configure(fg_color=("#1a1a2e", "#0f0f1e"))
        
        # Header
        header = ctk.CTkFrame(details_window, fg_color=("#16213e", "#0d1117"))
        header.pack(fill="x", padx=0, pady=0)
        
        title = ctk.CTkLabel(
            header,
            text=f"📋 {species['name'].upper()}",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=("#00d9ff", "#00bfff")
        )
        title.pack(pady=20)
        
        # Contenido scrollable
        scroll = ctk.CTkScrollableFrame(details_window, fg_color=("#1a1a2e", "#0f0f1e"))
        scroll.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Informacion detallada
        details_text = f"""
🔬 Nombre Científico: {species['scientific_name']}
🏛️ Familia: {species['family']}
👁️ Avistamientos Totales: {species['total_sightings']}
✅ Estado: {'CAPTURADO' if species['captured'] else 'NO CAPTURADO'}

📅 Primer Avistamiento: {species['first_seen']}
📅 Último Avistamiento: {species['last_seen']}
"""
        
        if species['nickname']:
            details_text += f"\n🏷️ Apodo: {species['nickname']}"
        
        info_label = ctk.CTkLabel(
            scroll,
            text=details_text,
            font=ctk.CTkFont(size=14),
            text_color=("#ffffff", "#dddddd"),
            justify="left"
        )
        info_label.pack(pady=10, anchor="w")
        
        # Boton cerrar
        close_btn = ctk.CTkButton(
            details_window,
            text="Cerrar",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=("#e74c3c", "#c0392b"),
            hover_color=("#c0392b", "#a93226"),
            command=details_window.destroy
        )
        close_btn.pack(pady=10)
        
    def _on_closing(self) -> None:
        """Manejar cierre de aplicacion."""
        logger.info("Shutting down application")
        
        self.is_running = False
        
        if self.video_thread:
            self.video_thread.join(timeout=2)
        if self.prediction_thread:
            self.prediction_thread.join(timeout=2)
        if self.metrics_thread:
            self.metrics_thread.join(timeout=2)
            
        self.camera.stop()
        self.database.close()
        
        self.destroy()


def main() -> None:
    """Funcion principal."""
    logger.info("Starting Pokedex Ultra - Windows Edition")
    
    app = UltraPokedexUI()
    app.mainloop()
    
    logger.info("Application terminated")


if __name__ == "__main__":
    main()
