from __future__ import annotations

from typing import Protocol

import pandas as pd

from .baselines import PopularityRecommender
from .config import PipelineConfig
from .data import ITEM_COL, observed_catalog
from .split_data import SplitData

MODEL_NAMES = (
    "two_stage",
    "popularity",
    "ease",
    "two_tower",
    "item_cosine",
    "als",
    "bpr",
    "tfidf",
)


class Recommender(Protocol):
    name: str

    def fit(self, split: SplitData, *, tune: bool = False) -> None: ...

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame: ...


class PopularityModel:
    name = "popularity"

    def fit(self, split: SplitData, *, tune: bool = False) -> None:
        del tune
        catalog = set(observed_catalog(split.history, split.books)[ITEM_COL])
        self._model = PopularityRecommender().fit(split.history, catalog)
        self._history = split.history

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        return self._model.recommend(users, self._history, n=20)


def build_model(name: str, config: PipelineConfig) -> Recommender:
    if name == "two_stage":
        from .pipeline import TwoStage

        return TwoStage(config)
    if name == "popularity":
        return PopularityModel()
    if name == "ease":
        from .ease import EaseModel

        return EaseModel(config)
    if name == "two_tower":
        from .two_tower import TwoTowerModel

        return TwoTowerModel(config)
    if name in {"item_cosine", "als", "bpr", "tfidf"}:
        from .source_models import SourceModel

        return SourceModel(name, config)
    known = ", ".join(MODEL_NAMES)
    raise ValueError(f"Modelo desconocido: {name}. Opciones: {known}")
