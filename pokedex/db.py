"""
Repositorio SQLite para la Pokédex.

Responsabilidades:
- Inicializar el esquema de la base de datos
- Insertar entradas de análisis
- Consultar por id, por nombre y listar
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PokedexEntry:
    id: Optional[int]
    timestamp: float
    name: str
    confidence: float
    summary: Optional[str] = None
    habitat: Optional[str] = None
    diet: Optional[str] = None
    characteristics: Optional[str] = None
    conservation_status: Optional[str] = None
    scientific_name: Optional[str] = None
    source_url: Optional[str] = None
    image_path: Optional[str] = None
    nickname: Optional[str] = None
    captured: int = 0  # 0/1
    notes: Optional[str] = None
    dominant_color: Optional[str] = None  # nombre o hex
    dominant_color_rgb: Optional[str] = None  # "r,g,b"
    relative_size: Optional[float] = None  # 0..1 respecto del frame
    bbox: Optional[str] = None  # "x1,y1,x2,y2"
    features_json: Optional[str] = None  # blob JSON para extensibilidad


class PokedexRepository:
    def __init__(self, db_path: str = "pokedex.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_dir()
        self._init_schema()

    def _ensure_dir(self) -> None:
        d = os.path.dirname(self.db_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    summary TEXT,
                    habitat TEXT,
                    diet TEXT,
                    characteristics TEXT,
                    conservation_status TEXT,
                    scientific_name TEXT,
                    source_url TEXT,
                    image_path TEXT,
                    nickname TEXT,
                    captured INTEGER DEFAULT 0,
                    notes TEXT,
                    dominant_color TEXT,
                    dominant_color_rgb TEXT,
                    relative_size REAL,
                    bbox TEXT,
                    features_json TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entries_name ON entries(name)"
            )

            # Migración de columnas adicionales si no existen
            cur = conn.execute("PRAGMA table_info(entries)")
            cols = {row[1] for row in cur.fetchall()}

            def add_col(name: str, decl: str):
                if name not in cols:
                    conn.execute(f"ALTER TABLE entries ADD COLUMN {name} {decl}")

            add_col("conservation_status", "TEXT")
            add_col("scientific_name", "TEXT")
            add_col("source_url", "TEXT")
            add_col("image_path", "TEXT")
            add_col("nickname", "TEXT")
            add_col("captured", "INTEGER DEFAULT 0")
            add_col("notes", "TEXT")
            add_col("dominant_color", "TEXT")
            add_col("dominant_color_rgb", "TEXT")
            add_col("relative_size", "REAL")
            add_col("bbox", "TEXT")
            add_col("features_json", "TEXT")

    def add_entry(self, entry: PokedexEntry) -> int:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO entries(
                    timestamp,name,confidence,summary,habitat,diet,characteristics,
                    conservation_status,scientific_name,source_url,image_path,
                    nickname,captured,notes,dominant_color,dominant_color_rgb,relative_size,bbox,features_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?)
                """,
                (
                    entry.timestamp,
                    entry.name,
                    entry.confidence,
                    entry.summary,
                    entry.habitat,
                    entry.diet,
                    entry.characteristics,
                    entry.conservation_status,
                    entry.scientific_name,
                    entry.source_url,
                    entry.image_path,
                    entry.nickname,
                    entry.captured,
                    entry.notes,
                    entry.dominant_color,
                    entry.dominant_color_rgb,
                    entry.relative_size,
                    entry.bbox,
                    entry.features_json,
                ),
            )
            return int(cur.lastrowid)

    def list_entries(self, limit: int = 100, offset: int = 0) -> List[PokedexEntry]:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM entries ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [self._row_to_entry(r) for r in cur.fetchall()]

    def get_entry(self, entry_id: int) -> Optional[PokedexEntry]:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute("SELECT * FROM entries WHERE id=?", (entry_id,))
            row = cur.fetchone()
            return self._row_to_entry(row) if row else None

    def update_entry_info(
        self,
        entry_id: int,
        *,
        summary: Optional[str] = None,
        habitat: Optional[str] = None,
        diet: Optional[str] = None,
        characteristics: Optional[str] = None,
        conservation_status: Optional[str] = None,
        scientific_name: Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> None:
        """Actualizar campos de información enriquecida en una entrada existente."""
        fields = []
        values = []
        if summary is not None:
            fields.append("summary=?")
            values.append(summary)
        if habitat is not None:
            fields.append("habitat=?")
            values.append(habitat)
        if diet is not None:
            fields.append("diet=?")
            values.append(diet)
        if characteristics is not None:
            fields.append("characteristics=?")
            values.append(characteristics)
        if conservation_status is not None:
            fields.append("conservation_status=?")
            values.append(conservation_status)
        if scientific_name is not None:
            fields.append("scientific_name=?")
            values.append(scientific_name)
        if source_url is not None:
            fields.append("source_url=?")
            values.append(source_url)

        if not fields:
            return

        values.append(entry_id)
        set_clause = ", ".join(fields)
        with self._lock, self._get_conn() as conn:
            conn.execute(f"UPDATE entries SET {set_clause} WHERE id=?", tuple(values))

    def update_entry_fields(self, entry_id: int, **kwargs: Any) -> None:
        """Actualizar campos arbitrarios por nombre de columna.

        Ejemplo: update_entry_fields(1, nickname="Zorro", captured=1)
        """
        if not kwargs:
            return
        fields = []
        values = []
        for k, v in kwargs.items():
            fields.append(f"{k}=?")
            values.append(v)
        values.append(entry_id)
        set_clause = ", ".join(fields)
        with self._lock, self._get_conn() as conn:
            conn.execute(f"UPDATE entries SET {set_clause} WHERE id=?", tuple(values))

    def find_by_name(self, name: str, limit: int = 50) -> List[PokedexEntry]:
        with self._lock, self._get_conn() as conn:
            cur = conn.execute(
                "SELECT * FROM entries WHERE name LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{name}%", limit),
            )
            return [self._row_to_entry(r) for r in cur.fetchall()]

    def stats(self) -> Dict[str, Any]:
        with self._lock, self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            last = conn.execute(
                "SELECT MAX(timestamp) FROM entries"
            ).fetchone()[0]
            distinct = conn.execute(
                "SELECT COUNT(DISTINCT name) FROM entries"
            ).fetchone()[0]
            return {"total_entries": total, "last_timestamp": last, "distinct_species": distinct}

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> PokedexEntry:
        return PokedexEntry(
            id=row["id"],
            timestamp=row["timestamp"],
            name=row["name"],
            confidence=row["confidence"],
            summary=row["summary"],
            habitat=row["habitat"],
            diet=row["diet"],
            characteristics=row["characteristics"],
            conservation_status=row["conservation_status"],
            scientific_name=row["scientific_name"],
            source_url=row["source_url"],
            image_path=row["image_path"],
            nickname=row["nickname"] if "nickname" in row.keys() else None,
            captured=row["captured"] if "captured" in row.keys() else 0,
            notes=row["notes"] if "notes" in row.keys() else None,
            dominant_color=row["dominant_color"] if "dominant_color" in row.keys() else None,
            dominant_color_rgb=row["dominant_color_rgb"] if "dominant_color_rgb" in row.keys() else None,
            relative_size=row["relative_size"] if "relative_size" in row.keys() else None,
            bbox=row["bbox"] if "bbox" in row.keys() else None,
            features_json=row["features_json"] if "features_json" in row.keys() else None,
        )

    # Exportaciones
    @staticmethod
    def to_dict(entry: PokedexEntry) -> Dict[str, Any]:
        d = asdict(entry)
        # Formatear timestamp a ISO 8601
        try:
            d["timestamp_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(entry.timestamp))
        except Exception:
            pass
        return d

    @staticmethod
    def to_markdown(entry: PokedexEntry) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.timestamp))
        lines = [
            f"# Entrada Pokédex #{entry.id or ''}",
            "",
            f"Fecha: {ts}",
            f"Especie: {entry.name}",
            f"Confianza: {entry.confidence:.1%}",
        ]
        if entry.nickname:
            lines.append(f"Nombre asignado: {entry.nickname}")
        if entry.scientific_name:
            lines.append(f"Nombre científico: {entry.scientific_name}")
        if entry.summary:
            lines += ["", "## Descripción", entry.summary]
        if entry.habitat:
            lines += ["", "## Hábitat", entry.habitat]
        if entry.diet:
            lines += ["", "## Dieta", entry.diet]
        if entry.characteristics:
            lines += ["", "## Características", entry.characteristics]
        if entry.conservation_status:
            lines += ["", "## Estado de conservación", entry.conservation_status]
        # Características visuales
        vis = []
        if entry.dominant_color:
            vis.append(f"Color dominante: {entry.dominant_color} ({entry.dominant_color_rgb or ''})")
        if entry.relative_size is not None:
            vis.append(f"Tamaño relativo: {entry.relative_size:.1%}")
        if entry.bbox:
            vis.append(f"BBox: {entry.bbox}")
        if vis:
            lines += ["", "## Rasgos visuales", *vis]
        if entry.source_url:
            lines += ["", f"Fuente: {entry.source_url}"]
        if entry.image_path:
            lines += ["", f"Imagen: {entry.image_path}"]
        if entry.notes:
            lines += ["", "## Notas", entry.notes]
        return "\n".join(lines)
