"""Configuración serializable del recomendador."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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


@dataclass(frozen=True)
class SplitConfig:
    """Parámetros del holdout y del backtesting temporal."""

    holdout_k: int = 20
    rolling_folds: int = 3
    rolling_horizon: int = 20
    min_history: int = 5


@dataclass(frozen=True)
class RetrievalConfig:
    """Parámetros comunes de las fuentes de candidatos."""

    rrf_window: int = 1000
    rrf_cutoff: int = 200
    rrf_cutoff_search: SearchRange = field(
        default_factory=lambda: SearchRange(100, 1000, log=True, integer=True)
    )
    recency_half_life_days: float = 365.0
    item_knn_mode: tuple[str, ...] = ("occurrence", "adjusted")
    svd_factors: int = 64
    svd_epochs: int = 20
    implicit_factors: int = 64
    implicit_iterations: int = 20
    tfidf_max_features: int = 60_000
    rrf_k: float = 60.0
    fusion_weight: SearchRange = field(
        default_factory=lambda: SearchRange(0.1, 3.0, log=True)
    )
    source_weights: dict[str, float] = field(
        default_factory=lambda: {
            "popularidad": 0.5,
            "item_cosine": 1.0,
            "item_adjusted_cosine": 1.0,
            "svd": 1.0,
            "als": 1.0,
            "bpr": 1.0,
            "tfidf": 0.8,
        }
    )

    def __post_init__(self) -> None:
        low = int(self.rrf_cutoff_search.low)
        high = int(self.rrf_cutoff_search.high)
        if self.rrf_k <= 0:
            raise ValueError("rrf_k debe ser positivo")
        if self.rrf_window < 1:
            raise ValueError("rrf_window debe ser positivo")
        if not 1 <= self.rrf_cutoff <= self.rrf_window:
            raise ValueError("rrf_cutoff debe estar entre 1 y rrf_window")
        if low < 1 or high > self.rrf_window or low > high:
            raise ValueError("rrf_cutoff_search debe caer dentro de 1..rrf_window")


@dataclass(frozen=True)
class RankerConfig:
    """Parámetros de LambdaRank y su búsqueda."""

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
    search_space: dict[str, SearchRange] = field(
        default_factory=lambda: {
            "learning_rate": SearchRange(0.015, 0.12, log=True),
            "num_leaves": SearchRange(15, 127, log=True, integer=True),
            "max_depth": SearchRange(4, 12, integer=True),
            "min_child_samples": SearchRange(10, 100, integer=True),
            "subsample": SearchRange(0.65, 1.0),
            "colsample_bytree": SearchRange(0.6, 1.0),
            "reg_alpha": SearchRange(1e-8, 10.0, log=True),
            "reg_lambda": SearchRange(1e-8, 20.0, log=True),
        }
    )


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
