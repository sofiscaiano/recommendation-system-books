"""Modelos intercambiables. El corte de datos vive en ``experiment.py``."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from .baselines import PopularityRecommender
from .config import PipelineConfig
from .data import ITEM_COL, observed_catalog
from .experiment import Experiment

MODEL_NAMES = ("two_stage", "popularity")


class Recommender(Protocol):
    """``fit`` ve el train del experimento. ``recommend`` devuelve ``rank`` desde 1."""

    name: str

    def fit(self, experiment: Experiment, *, tune: bool = False) -> None: ...

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame: ...


class PopularityModel:
    """Baseline: el mismo orden global para todos, saltando lo ya leído."""

    name = "popularity"

    def fit(self, experiment: Experiment, *, tune: bool = False) -> None:
        del tune
        catalog = set(observed_catalog(experiment.history, experiment.books)[ITEM_COL])
        self._model = PopularityRecommender().fit(experiment.history, catalog)
        self._history = experiment.history

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        return self._model.recommend(users, self._history, n=20)


def build_model(name: str, config: PipelineConfig) -> Recommender:
    """Instancia el modelo pedido. Uno nuevo se agrega acá y en ``MODEL_NAMES``."""

    if name == "two_stage":
        from .pipeline import TwoStage

        return TwoStage(config)
    if name == "popularity":
        return PopularityModel()
    known = ", ".join(MODEL_NAMES)
    raise ValueError(f"Modelo desconocido: {name}. Opciones: {known}")
