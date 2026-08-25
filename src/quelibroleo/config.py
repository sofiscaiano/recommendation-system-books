from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Valores por defecto de la pipeline. `main.py` puede overridear por CLI."""

    k: int = 20
    batch_size: int = 8
    # Si True, el ranking solo considera libros que aparecen en `interacciones`.
    only_interacted_items: bool = True
