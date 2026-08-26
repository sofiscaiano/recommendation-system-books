from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Valores por defecto de la pipeline. `main.py` puede overridear por CLI."""

    k: int = 20
    batch_size: int = 8
    # Si True, el ranking solo considera libros que aparecen en `interacciones`.
    only_interacted_items: bool = True
    # Si True, solo libros cuyo género el lector ya vio en el historial de ranking.
    filter_seen_genres: bool = True
    # Si True, solo libros cuyo autor el lector ya vio en el historial de ranking.
    filter_seen_authors: bool = True
