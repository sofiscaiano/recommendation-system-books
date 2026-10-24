"""Configuración serializable del recomendador."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SearchRange:
    """Intervalo que Optuna recorre para un hiperparámetro."""

    low: float
    high: float
    log: bool = False
    integer: bool = False

    def suggest(self, trial: Any, name: str) -> float | int:
        if self.integer:
            return trial.suggest_int(name, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(name, self.low, self.high, log=self.log)


def search_ranges(search: Any) -> dict[str, SearchRange]:
    """Campos de un bloque `search`, en el orden de la clase."""

    return {item.name: getattr(search, item.name) for item in fields(search)}


@dataclass(frozen=True)
class SplitConfig:
    """Parámetros del holdout y del backtesting temporal."""

    holdout_k: int = 20
    rolling_folds: int = 3
    rolling_horizon: int = 20
    min_history: int = 5


@dataclass(frozen=True)
class RetrievalSearch:
    """Hiperparámetros de fusión que Optuna recorre."""

    rrf_cutoff: SearchRange = field(
        default_factory=lambda: SearchRange(100, 1000, log=True, integer=True)
    )
    source_weight: SearchRange = field(
        default_factory=lambda: SearchRange(0.1, 3.0, log=True)
    )

    def __post_init__(self) -> None:
        low = int(self.rrf_cutoff.low)
        high = int(self.rrf_cutoff.high)
        if low < 1 or low > high:
            raise ValueError("la búsqueda de rrf_cutoff debe ser un intervalo positivo")
        if self.source_weight.low < 0:
            raise ValueError("la búsqueda de source_weight no admite pesos negativos")


@dataclass(frozen=True)
class RetrievalConfig:
    """Fuentes de candidatos. Sin `--tune` se usan estos valores; Optuna solo toca `search`."""

    rrf_cutoff: int = 200
    rrf_k: float = 60.0
    recency_half_life_days: float = 365.0
    implicit_factors: int = 64
    implicit_iterations: int = 20
    tfidf_max_features: int = 60_000
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "popularidad": 0.5,
            "item_cosine": 1.0,
            "als": 1.0,
            "bpr": 1.0,
            "tfidf": 0.8,
        }
    )
    search: RetrievalSearch = field(default_factory=RetrievalSearch)

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("rrf_k debe ser positivo")
        if self.rrf_cutoff < 1:
            raise ValueError("rrf_cutoff debe ser positivo")


@dataclass(frozen=True)
class RankerSearch:
    """Hiperparámetros de LambdaRank que Optuna recorre."""

    learning_rate: SearchRange = field(
        default_factory=lambda: SearchRange(0.015, 0.12, log=True)
    )
    num_leaves: SearchRange = field(
        default_factory=lambda: SearchRange(15, 127, log=True, integer=True)
    )
    max_depth: SearchRange = field(
        default_factory=lambda: SearchRange(4, 12, integer=True)
    )
    min_child_samples: SearchRange = field(
        default_factory=lambda: SearchRange(10, 100, integer=True)
    )
    subsample: SearchRange = field(default_factory=lambda: SearchRange(0.65, 1.0))
    colsample_bytree: SearchRange = field(
        default_factory=lambda: SearchRange(0.6, 1.0)
    )
    reg_alpha: SearchRange = field(
        default_factory=lambda: SearchRange(1e-8, 10.0, log=True)
    )
    reg_lambda: SearchRange = field(
        default_factory=lambda: SearchRange(1e-8, 20.0, log=True)
    )


LABEL_GAIN_MODES = ("linear", "exponential")


@dataclass(frozen=True)
class RankerConfig:
    """LambdaRank. `n_estimators` y `early_stopping_rounds` no entran a Optuna.

    `label_gain` no entra a la búsqueda. `linear` es la identidad 0..10.
    `exponential` es `2^rating - 1`, la misma ganancia que
    `sklearn.metrics.ndcg_score`. Cambiarlo pide un `--tune` nuevo: el archivo
    guardado se eligió bajo la ganancia anterior.
    """

    n_estimators: int = 200
    learning_rate: float = 0.04
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 30
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30
    label_gain: str = "linear"
    search: RankerSearch = field(default_factory=RankerSearch)

    def __post_init__(self) -> None:
        if self.label_gain not in LABEL_GAIN_MODES:
            raise ValueError("label_gain debe ser 'linear' o 'exponential'")


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración raíz del pipeline."""

    db_path: Path = Path("data.db")
    example_path: Path = Path("ejemplo.csv")
    artifacts_dir: Path = Path("artifacts")
    output_path: Path = Path("submission.csv")
    random_seed: int = 42
    run_note: str = ""
    optuna_trials: int = 30
    split: SplitConfig = field(default_factory=SplitConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)

    def as_dict(self) -> dict[str, Any]:
        """Devuelve una representación apta para JSON."""

        raw = asdict(self)
        for key in ("db_path", "example_path", "artifacts_dir", "output_path"):
            raw[key] = str(raw[key])
        return raw
