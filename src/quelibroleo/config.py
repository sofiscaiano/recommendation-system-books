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
        """Pide a Optuna un entero o un float dentro de este intervalo."""

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
    """Fuentes de candidatos. Sin `--tune` se usan estos valores; Optuna solo toca `search`.

    `two_tower_source` suma el two-tower como sexta fuente. Está apagado por
    defecto: el serving y el tuneo guardado no lo tienen.
    """

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
            "two_tower": 1.0,
        }
    )
    two_tower_source: bool = False
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
EASE_WEIGHT_MODES = ("binary", "rating", "confidence")


@dataclass(frozen=True)
class EaseSearch:
    """Hiperparámetros de EASE que Optuna recorre. No los usa `two_stage`."""

    reg_lambda: SearchRange = field(
        default_factory=lambda: SearchRange(1.0, 100_000.0, log=True)
    )
    min_interactions: SearchRange = field(
        default_factory=lambda: SearchRange(10, 100, log=True, integer=True)
    )


@dataclass(frozen=True)
class EaseConfig:
    """EASE como modelo único. El peso de la matriz no entra al intervalo numérico.

    `binary` pone unos, `rating` el valor 1–10 y `confidence` la misma
    confianza de ALS: `(1 + rating / 10)` por recencia. Sin `--tune` se usan
    estos tres valores. La búsqueda escribe otro archivo que el de LambdaRank.
    """

    reg_lambda: float = 500.0
    min_interactions: int = 20
    weight_mode: str = "binary"
    search: EaseSearch = field(default_factory=EaseSearch)

    def __post_init__(self) -> None:
        if self.weight_mode not in EASE_WEIGHT_MODES:
            known = ", ".join(EASE_WEIGHT_MODES)
            raise ValueError(f"weight_mode debe ser uno de: {known}")
        if self.reg_lambda <= 0:
            raise ValueError("reg_lambda debe ser positivo")
        if self.min_interactions < 1:
            raise ValueError("min_interactions debe ser positivo")


@dataclass(frozen=True)
class TwoTowerConfig:
    """Two-tower de Covington (2016) con la pérdida de Yi (2019). Sin Optuna.

    Solo la época sale de la validación: NDCG@20 del último fold rolling.
    `recent_reads` suma a la torre de lector el promedio de sus últimas
    lecturas (0 lo apaga). `user_authors` suma el promedio de los autores.
    """

    embedding_dim: int = 64
    recent_reads: int = 20
    user_authors: bool = True
    dropout: float = 0.1
    temperature: float = 0.05
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 3
    min_item_reads: int = 5

    def __post_init__(self) -> None:
        if min(self.embedding_dim, self.batch_size, self.max_epochs, self.patience) < 1:
            raise ValueError("dimensión, batch, épocas y patience deben ser positivos")
        if self.min_item_reads < 1:
            raise ValueError("min_item_reads debe ser positivo")
        if self.recent_reads < 0:
            raise ValueError("recent_reads no puede ser negativo")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout tiene que estar en [0, 1)")
        if self.temperature <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("temperatura y learning rate positivos; weight decay no negativo")


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
    """Configuración raíz del pipeline.

    `tune_on_example_users` limita el NDCG@20 de Optuna y el early stopping
    del último fold a los lectores de `example_path`. El ranker sigue
    entrenando con todos los folds. Apagado, ese fold completo define ambos.
    """

    db_path: Path = Path("data.db")
    example_path: Path = Path("ejemplo.csv")
    artifacts_dir: Path = Path("artifacts")
    output_path: Path = Path("submission.csv")
    random_seed: int = 42
    run_note: str = ""
    model: str = "two_stage"
    optuna_trials: int = 30
    tune_on_example_users: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    ease: EaseConfig = field(default_factory=EaseConfig)
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)

    def as_dict(self) -> dict[str, Any]:
        """Devuelve una representación apta para JSON."""

        raw = asdict(self)
        for key in ("db_path", "example_path", "artifacts_dir", "output_path"):
            raw[key] = str(raw[key])
        return raw
