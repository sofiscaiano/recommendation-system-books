"""Configuración serializable del recomendador."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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

    candidate_budgets: tuple[int, ...] = (100, 200, 500, 1000)
    selected_budget: int = 200
    recency_half_life_days: float = 365.0
    item_knn_mode: tuple[str, ...] = ("occurrence", "adjusted")
    svd_factors: int = 64
    svd_epochs: int = 20
    implicit_factors: int = 64
    implicit_iterations: int = 20
    tfidf_max_features: int = 60_000
    rrf_k: float = 60.0
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
    source_quotas: dict[str, int] = field(
        default_factory=lambda: {
            "popularidad": 25,
            "item_cosine": 60,
            "item_adjusted_cosine": 60,
            "svd": 75,
            "als": 75,
            "bpr": 75,
            "tfidf": 60,
        }
    )


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
    optuna_trials: int = 30


@dataclass(frozen=True)
class PipelineConfig:
    """Configuración raíz del pipeline."""

    db_path: Path = Path("data.db")
    example_path: Path = Path("ejemplo.csv")
    artifacts_dir: Path = Path("artifacts")
    output_path: Path = Path("submission.csv")
    random_seed: int = 42
    run_note: str = ""
    split: SplitConfig = field(default_factory=SplitConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)

    def as_dict(self) -> dict[str, Any]:
        """Devuelve una representación apta para JSON."""

        raw = asdict(self)
        for key in ("db_path", "example_path", "artifacts_dir", "output_path"):
            raw[key] = str(raw[key])
        return raw
