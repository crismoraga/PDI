#!/usr/bin/env python3
"""Aplicación principal de la Pokédex Animal."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageTk
from tensorflow import keras
import tkinter as tk
from tkinter import ttk

from utils.camera import CameraCapture
from utils.image_processing import ImageProcessor
from utils.api import AnimalInfoAPI
from model.animal_classifier import AnimalClassifier

SNAPSHOT_DIR = Path("data/snapshots")
EXPORT_DIR = Path("data/exports")
APP_TITLE = "Pokedex Animal - PDI Project"
WINDOW_SIZE = "1280x820"
VIDEO_REFRESH_MS = 40


class AnimalPokedexApp:
    """Interfaz principal de la Pokédex Animal."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(1100, 720)
        self.root.configure(bg="#0f172a")

        self.status_var = tk.StringVar(value="Sistema listo")
        self.search_var = tk.StringVar()
        self.nick_var = tk.StringVar()
        self.notes_var = tk.StringVar()
        self.filter_captured_var = tk.BooleanVar(value=False)

        self.camera_active = False
        self.fullscreen = False
        self.current_prediction: str = ""
        self.confidence_score: float = 0.0
        self.last_entry_id: Optional[int] = None
        self.last_snapshot_path: Optional[str] = None
        self.last_bbox: Optional[str] = None

        self._video_lock = threading.Lock()
        self.snapshot_photo: Optional[ImageTk.PhotoImage] = None

        self._configure_logging()
        self._configure_style()
        self._setup_components()
        self._setup_ui()
        self._refresh_tree()

    # ------------------------------------------------------------------
    # Configuración
    # ------------------------------------------------------------------
    def _configure_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        palette = {
            "bg": "#0f172a",
            "panel": "#111c2f",
            "accent": "#1f2d49",
            "focus": "#3a6ff7",
            "text": "#f1f5f9",
            "muted": "#94a3b8",
        }

        style.configure("Root.TFrame", background=palette["bg"])
        style.configure("Panel.TLabelframe", background=palette["panel"], foreground=palette["text"], borderwidth=0)
        style.configure("Panel.TLabelframe.Label", background=palette["panel"], foreground=palette["muted"], font=("Segoe UI", 11, "bold"))
        style.configure("Panel.TFrame", background=palette["panel"])
        style.configure("Title.TLabel", background=palette["bg"], foreground=palette["text"], font=("Segoe UI", 22, "bold"))
        style.configure("Header.TLabel", background=palette["panel"], foreground=palette["text"], font=("Segoe UI", 14, "bold"))
        style.configure("Info.TLabel", background=palette["panel"], foreground=palette["muted"], font=("Segoe UI", 11))
        style.configure("Status.TLabel", background=palette["accent"], foreground=palette["text"], anchor="w", padding=6, font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.configure("Plain.TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Treeview", background=palette["accent"], foreground=palette["text"], fieldbackground=palette["accent"])
        style.map("Accent.TButton", foreground=[("active", palette["text"])], background=[("active", palette["focus"]), ("!disabled", palette["accent"])])
        style.map("Treeview", background=[("selected", palette["focus"])])

        self.root.option_add("*TEntry*Background", palette["accent"])
        self.root.option_add("*TEntry*Foreground", palette["text"])
        self.root.option_add("*TEntry*Font", "Segoe UI 11")

    def _setup_components(self) -> None:
        logging.info("Inicializando componentes")
        self.camera = CameraCapture()
        self.image_processor = ImageProcessor()

        self.classifier_backend = "keras"
        try:
            from model.tflite_classifier import TFLiteAnimalClassifier

            self.classifier = TFLiteAnimalClassifier()
            self.classifier_backend = "tflite"
            logging.info("Clasificador TFLite activo")
        except Exception as exc:  # pragma: no cover - fallback control
            logging.warning("Fallo al cargar TFLite: %s", exc)
            self.classifier = AnimalClassifier()
            self.classifier_backend = "keras"
            logging.info("Clasificador Keras activo")

        try:
            from pokedex.db import PokedexRepository, PokedexEntry

            self.pokedex_repo = PokedexRepository(db_path=os.path.join("data", "pokedex.db"))
            self.PokedexEntry = PokedexEntry
        except Exception as exc:  # pragma: no cover - fallback control
            logging.error("No se pudo inicializar la Pokédex: %s", exc)
            self.pokedex_repo = None
            self.PokedexEntry = None

        self.animal_api = AnimalInfoAPI()
        logging.info("Componentes listos")

    # ------------------------------------------------------------------
    # Interfaz de usuario
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=20)
        root_frame.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(root_frame, text="Pokedex Animal", style="Title.TLabel")
        title.pack(anchor="w", pady=(0, 16))

        main_pane = ttk.Panedwindow(root_frame, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_pane, style="Panel.TFrame", padding=12)
        right_frame = ttk.Frame(main_pane, style="Panel.TFrame", padding=12)
        main_pane.add(left_frame, weight=3)
        main_pane.add(right_frame, weight=2)

        self._build_video_panel(left_frame)
        self._build_pokedex_panel(right_frame)
        self._build_status_bar()

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.root.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda _event: self.disable_fullscreen())

    def _build_video_panel(self, parent: ttk.Frame) -> None:
        video_container = ttk.Labelframe(parent, text="Cámara en vivo", style="Panel.TLabelframe", padding=12)
        video_container.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(video_container, bg="#0f172a", fg="#94a3b8", width=640, height=480)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.configure(text="Video no disponible")

        controls = ttk.Frame(parent, style="Panel.TFrame", padding=(0, 16, 0, 0))
        controls.pack(fill=tk.X)

        self.start_button = ttk.Button(controls, text="Iniciar cámara", style="Accent.TButton", command=self.toggle_camera)
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))

        self.capture_button = ttk.Button(controls, text="Capturar y analizar", style="Accent.TButton", command=self.capture_and_analyze)
        self.capture_button.pack(side=tk.LEFT, padx=8)

        self.save_button = ttk.Button(controls, text="Guardar captura", style="Plain.TButton", command=self.save_current_entry)
        self.save_button.pack(side=tk.LEFT, padx=8)

        self.fullscreen_button = ttk.Button(controls, text="Pantalla completa", style="Plain.TButton", command=self.toggle_fullscreen)
        self.fullscreen_button.pack(side=tk.LEFT, padx=8)

        self.stop_button = ttk.Button(controls, text="Detener cámara", style="Plain.TButton", command=self.stop_camera)
        self.stop_button.pack(side=tk.LEFT, padx=8)

        self.quit_button = ttk.Button(controls, text="Salir", style="Plain.TButton", command=self.quit_app)
        self.quit_button.pack(side=tk.RIGHT)

    def _build_pokedex_panel(self, parent: ttk.Frame) -> None:
        info_frame = ttk.Labelframe(parent, text="Estado del análisis", style="Panel.TLabelframe", padding=12)
        info_frame.pack(fill=tk.X, expand=False)

        self.prediction_label = ttk.Label(info_frame, text="Animal no detectado", style="Header.TLabel")
        self.prediction_label.grid(row=0, column=0, sticky="w")

        self.confidence_label = ttk.Label(info_frame, text="Confianza: 0%", style="Info.TLabel")
        self.confidence_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        backend_text = "Clasificador: TensorFlow Lite" if self.classifier_backend == "tflite" else "Clasificador: Keras"
        self.backend_label = ttk.Label(info_frame, text=backend_text, style="Info.TLabel")
        self.backend_label.grid(row=2, column=0, sticky="w", pady=(2, 0))

        info_frame.columnconfigure(0, weight=1)

        snapshot_frame = ttk.Labelframe(parent, text="Última captura", style="Panel.TLabelframe", padding=12)
        snapshot_frame.pack(fill=tk.BOTH, expand=False, pady=(12, 12))

        self.snapshot_label = tk.Label(snapshot_frame, bg="#111c2f", fg="#94a3b8", width=480, height=320)
        self.snapshot_label.pack(fill=tk.BOTH, expand=True)
        self.snapshot_label.configure(text="Sin captura disponible")

        summary_frame = ttk.Labelframe(parent, text="Resumen del animal", style="Panel.TLabelframe", padding=12)
        summary_frame.pack(fill=tk.BOTH, expand=True)

        self.info_text = tk.Text(summary_frame, wrap=tk.WORD, height=10, bg="#111c2f", fg="#f1f5f9", insertbackground="#f1f5f9", font=("Segoe UI", 11))
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll = ttk.Scrollbar(summary_frame, command=self.info_text.yview)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=info_scroll.set, state=tk.DISABLED)

        pokedex_frame = ttk.Labelframe(parent, text="Pokédex", style="Panel.TLabelframe", padding=12)
        pokedex_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        controls = ttk.Frame(pokedex_frame, style="Panel.TFrame")
        controls.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(controls, text="Buscar especie", style="Info.TLabel").pack(side=tk.LEFT)
        search_entry = ttk.Entry(controls, textvariable=self.search_var, width=24)
        search_entry.pack(side=tk.LEFT, padx=8)

        ttk.Button(controls, text="Buscar", style="Plain.TButton", command=self.search_pokedex).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Listar recientes", style="Plain.TButton", command=self.list_recent_entries).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Exportar JSON", style="Plain.TButton", command=lambda: self.export_selected("json")).pack(side=tk.LEFT, padx=4)
        ttk.Button(controls, text="Exportar Markdown", style="Plain.TButton", command=lambda: self.export_selected("md")).pack(side=tk.LEFT, padx=4)

        ttk.Checkbutton(
            controls,
            text="Solo capturados",
            variable=self.filter_captured_var,
            style="Info.TLabel",
            command=self._refresh_tree,
        ).pack(side=tk.RIGHT)

        data_entry_frame = ttk.Frame(pokedex_frame, style="Panel.TFrame")
        data_entry_frame.pack(fill=tk.X, pady=(8, 8))

        ttk.Label(data_entry_frame, text="Nombre asignado", style="Info.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(data_entry_frame, textvariable=self.nick_var, width=20).grid(row=0, column=1, padx=6)
        ttk.Label(data_entry_frame, text="Notas", style="Info.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Entry(data_entry_frame, textvariable=self.notes_var, width=32).grid(row=0, column=3, padx=6)

        columns = ("id", "fecha", "nombre", "conf", "capturado")
        self.entries_tree = ttk.Treeview(pokedex_frame, columns=columns, show="headings", height=8)
        headings = {
            "id": ("ID", 60),
            "fecha": ("Fecha", 150),
            "nombre": ("Nombre", 160),
            "conf": ("Conf", 80),
            "capturado": ("Estado", 80),
        }
        for col, (title, width) in headings.items():
            self.entries_tree.heading(col, text=title)
            self.entries_tree.column(col, width=width, anchor=tk.W)
        self.entries_tree.pack(fill=tk.BOTH, expand=True)
        self.entries_tree.bind("<<TreeviewSelect>>", self.on_select_entry)

        detail_frame = ttk.Labelframe(parent, text="Detalle de la entrada", style="Panel.TLabelframe", padding=12)
        detail_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.detail_text = tk.Text(detail_frame, wrap=tk.WORD, height=10, bg="#111c2f", fg="#f1f5f9", insertbackground="#f1f5f9", font=("Segoe UI", 11))
        self.detail_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scroll = ttk.Scrollbar(detail_frame, command=self.detail_text.yview)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.detail_text.configure(yscrollcommand=detail_scroll.set, state=tk.DISABLED)

    def _build_status_bar(self) -> None:
        status_bar = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------
    # Cámara
    # ------------------------------------------------------------------
    def toggle_camera(self) -> None:
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self) -> None:
        if self.camera_active:
            return
        if self.camera.start():
            self.camera_active = True
            self.start_button.configure(text="Cámara activa")
            self.set_status("Cámara iniciada")
            self.root.after(VIDEO_REFRESH_MS, self._update_video_feed)
        else:
            self.set_status("No se pudo iniciar la cámara")

    def stop_camera(self) -> None:
        if not self.camera_active:
            return
        self.camera_active = False
        self.camera.stop()
        self.start_button.configure(text="Iniciar cámara")
        self.video_label.configure(image=None, text="Video detenido")
        self.set_status("Cámara detenida")

    def _update_video_feed(self) -> None:
        if not self.camera_active:
            return
        frame = self.camera.get_frame()
        if frame is not None:
            frame_resized = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
        self.root.after(VIDEO_REFRESH_MS, self._update_video_feed)

    # ------------------------------------------------------------------
    # Captura y análisis
    # ------------------------------------------------------------------
    def capture_and_analyze(self) -> None:
        if not self.camera_active:
            self.set_status("La cámara no está activa")
            return

        frame = self.camera.get_frame()
        if frame is None:
            self.set_status("No se pudo capturar el frame")
            return

        processed_frame = self.image_processor.preprocess_for_classification(frame)
        if processed_frame is None:
            self.set_status("Error al preprocesar la imagen")
            return

        prediction, confidence = self.classifier.predict(processed_frame)
        self.current_prediction = prediction
        self.confidence_score = float(confidence)

        self.prediction_label.configure(text=f"Especie: {prediction}")
        self.confidence_label.configure(text=f"Confianza: {confidence:.1%}")

        if confidence > 0.3:
            self.search_animal_info_async(prediction)
        else:
            self._set_info_text("Confianza insuficiente. Ajusta iluminación o distancia.")

        features = self.image_processor.compute_visual_features(frame)
        timestamp = int(time.time())
        safe_name = prediction.replace(" ", "_").lower() or "desconocido"
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_path = SNAPSHOT_DIR / f"{timestamp}_{safe_name}.jpg"
        cv2.imwrite(str(snapshot_path), frame)

        bbox_str = features.get("bbox") if features else None
        self.update_snapshot_preview(str(snapshot_path), bbox_str)

        if self.pokedex_repo and self.PokedexEntry:
            entry = self.PokedexEntry(
                id=None,
                timestamp=time.time(),
                name=prediction,
                confidence=float(confidence),
                summary=None,
                habitat=None,
                diet=None,
                characteristics=None,
                conservation_status=None,
                scientific_name=None,
                source_url=None,
                image_path=str(snapshot_path),
                nickname=None,
                captured=0,
                notes=None,
                dominant_color=features.get("dominant_color_hex") if features else None,
                dominant_color_rgb=features.get("dominant_color_rgb") if features else None,
                relative_size=features.get("relative_size") if features else None,
                bbox=bbox_str,
                features_json=json.dumps(features, ensure_ascii=False) if features else None,
            )
            try:
                entry_id = self.pokedex_repo.add_entry(entry)
                self.last_entry_id = entry_id
                self.last_snapshot_path = str(snapshot_path)
                self.last_bbox = bbox_str
                self.set_status(f"Entrada registrada (ID {entry_id})")
                self._refresh_tree()
            except Exception as exc:  # pragma: no cover
                logging.error("No se pudo registrar la entrada: %s", exc)
                self.set_status("Error al registrar la entrada en la Pokédex")
        else:
            self.set_status("Pokédex no disponible. Guarda manualmente los datos si lo requieres.")

    # ------------------------------------------------------------------
    # Información
    # ------------------------------------------------------------------
    def search_animal_info_async(self, animal_name: str) -> None:
        thread = threading.Thread(target=self._search_animal_info, args=(animal_name,), daemon=True)
        thread.start()

    def _search_animal_info(self, animal_name: str) -> None:
        self._set_info_text("Buscando información...")
        try:
            info = self.animal_api.get_animal_info(animal_name)
        except Exception as exc:  # pragma: no cover
            logging.error("Error en consulta de información: %s", exc)
            self._set_info_text("No se pudo obtener información externa")
            return

        if not info:
            self._set_info_text("Sin resultados para la especie detectada")
            return

        formatted = self._format_animal_info(info)
        self._set_info_text(formatted)

        if self.pokedex_repo and self.last_entry_id:
            try:
                self.pokedex_repo.update_entry_info(
                    self.last_entry_id,
                    summary=info.get("summary"),
                    habitat=info.get("habitat"),
                    diet=info.get("diet"),
                    characteristics=info.get("characteristics"),
                    conservation_status=info.get("conservation_status"),
                    scientific_name=info.get("scientific_name"),
                    source_url=info.get("source_url"),
                )
                self._refresh_tree()
            except Exception as exc:  # pragma: no cover
                logging.error("No se pudo actualizar la entrada con información externa: %s", exc)

    def _format_animal_info(self, info: dict) -> str:
        blocks = []
        summary = info.get("summary")
        if summary:
            blocks.append(summary.strip())

        for label in ("habitat", "diet", "characteristics", "conservation_status"):
            value = info.get(label)
            if value:
                title = label.replace("_", " ").title()
                blocks.append(f"{title}: {value}")
        return "\n\n".join(blocks)

    def _set_info_text(self, text: str) -> None:
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Pokédex
    # ------------------------------------------------------------------
    def _refresh_tree(self) -> None:
        if not self.pokedex_repo:
            return
        rows = self.pokedex_repo.list_entries(limit=200)
        self._populate_tree(rows)

    def _populate_tree(self, rows) -> None:
        self.entries_tree.delete(*self.entries_tree.get_children())
        filter_captured = self.filter_captured_var.get()
        for entry in rows:
            if filter_captured and entry.captured != 1:
                continue
            date_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.timestamp))
            confidence_text = f"{entry.confidence:.0%}"
            status = "Capturado" if entry.captured else "Visto"
            self.entries_tree.insert(
                "",
                tk.END,
                iid=str(entry.id),
                values=(entry.id, date_text, entry.name, confidence_text, status),
            )

    def on_select_entry(self, _event) -> None:
        if not self.pokedex_repo:
            return
        selection = self.entries_tree.selection()
        if not selection:
            return
        entry_id = int(selection[0])
        entry = self.pokedex_repo.get_entry(entry_id)
        if not entry:
            return

        self.nick_var.set(entry.nickname or "")
        self.notes_var.set(entry.notes or "")
        self.last_entry_id = entry_id
        self.update_snapshot_preview(entry.image_path, entry.bbox)

        lines = [
            f"ID: {entry.id}",
            f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry.timestamp))}",
            f"Especie: {entry.name}",
            f"Confianza: {entry.confidence:.1%}",
            f"Estado: {'Capturado' if entry.captured else 'Visto'}",
        ]
        if entry.nickname:
            lines.append(f"Nombre asignado: {entry.nickname}")
        if entry.scientific_name:
            lines.append(f"Nombre científico: {entry.scientific_name}")
        if entry.habitat:
            lines.append(f"Hábitat: {entry.habitat}")
        if entry.diet:
            lines.append(f"Dieta: {entry.diet}")
        if entry.characteristics:
            lines.append(f"Características: {entry.characteristics}")
        if entry.conservation_status:
            lines.append(f"Estado de conservación: {entry.conservation_status}")
        if entry.dominant_color:
            lines.append(f"Color dominante: {entry.dominant_color} ({entry.dominant_color_rgb})")
        if entry.relative_size is not None:
            lines.append(f"Tamaño relativo: {entry.relative_size:.1%}")
        if entry.bbox:
            lines.append(f"Bounding box: {entry.bbox}")
        if entry.notes:
            lines.append(f"Notas: {entry.notes}")
        if entry.source_url:
            lines.append(f"Fuente: {entry.source_url}")
        if entry.summary:
            lines.append("")
            lines.append(entry.summary)

        detail_text = "\n".join(lines)
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, detail_text)
        self.detail_text.configure(state=tk.DISABLED)

    def list_recent_entries(self) -> None:
        if not self.pokedex_repo:
            return
        rows = self.pokedex_repo.list_entries(limit=50)
        self._populate_tree(rows)

    def search_pokedex(self) -> None:
        if not self.pokedex_repo:
            return
        term = self.search_var.get().strip()
        if not term:
            self._refresh_tree()
            return
        rows = self.pokedex_repo.find_by_name(term, limit=100)
        self._populate_tree(rows)

    def save_current_entry(self) -> None:
        if not self.pokedex_repo:
            self.set_status("Pokédex no disponible")
            return
        target_id: Optional[int] = None
        selection = self.entries_tree.selection()
        if selection:
            target_id = int(selection[0])
        elif self.last_entry_id:
            target_id = self.last_entry_id

        if not target_id:
            self.set_status("No hay entrada seleccionada para guardar")
            return

        updates = {
            "nickname": self.nick_var.get().strip() or None,
            "notes": self.notes_var.get().strip() or None,
            "captured": 1,
        }
        try:
            self.pokedex_repo.update_entry_fields(target_id, **updates)
            self.set_status(f"Entrada {target_id} actualizada")
            self._refresh_tree()
        except Exception as exc:  # pragma: no cover
            logging.error("No se pudo actualizar la entrada: %s", exc)
            self.set_status("Error al guardar la captura")

    def export_selected(self, fmt: str) -> None:
        if not self.pokedex_repo:
            self.set_status("Pokédex no disponible")
            return
        selection = self.entries_tree.selection()
        if not selection:
            self.set_status("Selecciona una entrada para exportar")
            return
        entry_id = int(selection[0])
        entry = self.pokedex_repo.get_entry(entry_id)
        if not entry:
            self.set_status("Entrada no encontrada")
            return

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        if fmt == "json":
            path = EXPORT_DIR / f"entry_{entry_id}.json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.pokedex_repo.to_dict(entry), handle, ensure_ascii=False, indent=2)
            self.set_status(f"Exportado JSON en {path}")
        else:
            path = EXPORT_DIR / f"entry_{entry_id}.md"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.pokedex_repo.to_markdown(entry))
            self.set_status(f"Exportado Markdown en {path}")

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def update_snapshot_preview(self, image_path: Optional[str], bbox: Optional[str]) -> None:
        if not image_path or not os.path.exists(image_path):
            self.snapshot_label.configure(image=None, text="Sin captura disponible")
            self.snapshot_label.image = None
            return

        image = Image.open(image_path).convert("RGB")
        if bbox:
            try:
                coords = [int(float(v)) for v in bbox.split(",")]
                if len(coords) == 4:
                    draw = ImageDraw.Draw(image)
                    draw.rectangle(coords, outline="#3a6ff7", width=4)
            except ValueError:
                pass
        preview = image.copy()
        preview.thumbnail((520, 360), Image.LANCZOS)
        self.snapshot_photo = ImageTk.PhotoImage(preview)
        self.snapshot_label.configure(image=self.snapshot_photo, text="")
        self.snapshot_label.image = self.snapshot_photo

    def set_status(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.status_var.set(f"[{timestamp}] {message}")
        logging.info(message)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)
        label = "Salir de pantalla completa" if self.fullscreen else "Pantalla completa"
        self.fullscreen_button.configure(text=label)

    def disable_fullscreen(self) -> None:
        if self.fullscreen:
            self.fullscreen = False
            self.root.attributes("-fullscreen", False)
            self.fullscreen_button.configure(text="Pantalla completa")

    def quit_app(self) -> None:
        self.stop_camera()
        self.root.quit()
        self.root.destroy()

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.set_status("Aplicación iniciada")
        self.root.mainloop()


def main() -> None:
    print("=" * 70)
    print("POKEDEX ANIMAL - Procesamiento Digital de Imágenes")
    print("=" * 70)

    try:
        app = AnimalPokedexApp()
        app.run()
    except KeyboardInterrupt:
        print("\nAplicación interrumpida por el usuario")
    except Exception as exc:  # pragma: no cover
        print(f"Error crítico: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        print("Ejecución finalizada")


if __name__ == "__main__":
    main()
