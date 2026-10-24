"""Modelos intercambiables. El corte de datos vive en ``split_data.py``."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .baselines import PopularityRecommender
from .config import PipelineConfig
from .data import ITEM_COL, observed_catalog
from .split_data import SplitData

MODEL_NAMES = ("two_stage", "popularity")


class Recommender(Protocol):
    """Cualquier algoritmo, de una etapa o de dos. El corte no lo arma el modelo.

    ``fit`` entrena con ``split.history``. Si hace falta validación, usa
    ``validation_folds``: el último fold es la validación. ``recommend``
    devuelve ``id_lector``, ``id_libro`` y ``rank`` desde 1, sin libros ya
    leídos. ``two_stage`` busca candidatos y después los ordena; un modelo
    único implementa estos dos métodos y no importa ``two_stage``.
    """

    name: str

    def fit(self, split: SplitData, *, tune: bool = False) -> None:
        """Entrena solo con ``split.history``. ``tune`` es opcional."""
        ...

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        """Ranking con columna ``rank`` desde 1. No incluye libros ya leídos."""
        ...


class PopularityModel:
    """Baseline: el mismo orden global para todos, saltando lo ya leído."""

    name = "popularity"

    def fit(self, split: SplitData, *, tune: bool = False) -> None:
        """Cuenta interacciones del train. Ignora ``tune``: no tiene hiperparámetros."""

        del tune
        catalog = set(observed_catalog(split.history, split.books)[ITEM_COL])
        self._model = PopularityRecommender().fit(split.history, catalog)
        self._history = split.history

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        """Top-20 global, saltando lo que cada lector ya leyó."""

        return self._model.recommend(users, self._history, n=20)


def build_model(name: str, config: PipelineConfig) -> Recommender:
    """Instancia el modelo pedido. Uno nuevo se agrega acá y en ``MODEL_NAMES``.

    No tiene que separar retrieval y ranking. Alcanza con ``fit`` y ``recommend``.
    """

    if name == "two_stage":
        from .pipeline import TwoStage

        return TwoStage(config)
    if name == "popularity":
        return PopularityModel()
    known = ", ".join(MODEL_NAMES)
    raise ValueError(f"Modelo desconocido: {name}. Opciones: {known}")
