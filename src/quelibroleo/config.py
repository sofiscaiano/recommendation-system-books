from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Valores por defecto de la pipeline. `main.py` puede overridear por CLI."""

    k: int = 20
    batch_size: int = 8
    # Fracción de lectores del hold-out para eval (1.0 = todos). Solo afecta eval, no submission.
    eval_user_frac: float = 1.0
    eval_seed: int = 42
    # Si True, el ranking solo considera libros que aparecen en `interacciones`.
    only_interacted_items: bool = True
    # Si True, solo libros cuyo género el lector ya vio en el historial de ranking.
    filter_seen_genres: bool = True
    # Si True, solo libros cuyo autor el lector ya vio en el historial de ranking.
    filter_seen_authors: bool = True
