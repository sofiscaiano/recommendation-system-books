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


def _check_neural_schedule(
    *,
    embedding_dim: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    finetune_epochs: int,
) -> None:
    """Valida el bloque compartido por rating_mlp y two_tower."""

    if embedding_dim < 1:
        raise ValueError("embedding_dim debe ser positivo")
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo")
    if min(batch_size, max_epochs, patience) < 1:
        raise ValueError("batch_size, max_epochs y patience deben ser positivos")
    if finetune_epochs < 0:
        raise ValueError("finetune_epochs no puede ser negativo")


@dataclass(frozen=True)
class RatingMlpConfig:
    """Red única del Colab. Los hiperparámetros no pasan por Optuna.

    Lector, libro, autor y género son embeddings. Se concatenan y una sola red
    predice el rating. `--tune` no abre un study: se usan estos valores.
    """

    embedding_dim: int = 512
    hidden_1: int = 256
    hidden_2: int = 64
    learning_rate: float = 0.001
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5
    finetune_epochs: int = 5

    def __post_init__(self) -> None:
        _check_neural_schedule(
            embedding_dim=self.embedding_dim,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            finetune_epochs=self.finetune_epochs,
        )
        if min(self.hidden_1, self.hidden_2) < 1:
            raise ValueError("las capas ocultas deben ser positivas")


@dataclass(frozen=True)
class TwoTowerConfig:
    """Dos torres. El score es el producto de los vectores, sin tercera red.

    El lector es un embedding pasado por su red. El libro concatena su
    embedding con el de autor y el de género, y pasa por la otra. `--tune`
    no abre un study: se usan estos valores.
    """

    embedding_dim: int = 512
    hidden_dim: int = 512
    output_dim: int = 256
    learning_rate: float = 0.001
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5
    finetune_epochs: int = 5

    def __post_init__(self) -> None:
        _check_neural_schedule(
            embedding_dim=self.embedding_dim,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            finetune_epochs=self.finetune_epochs,
        )
        if min(self.hidden_dim, self.output_dim) < 1:
            raise ValueError("hidden_dim y output_dim deben ser positivos")


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
    model: str = "two_stage"
    optuna_trials: int = 30
    split: SplitConfig = field(default_factory=SplitConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    ease: EaseConfig = field(default_factory=EaseConfig)
    rating_mlp: RatingMlpConfig = field(default_factory=RatingMlpConfig)
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)

    def as_dict(self) -> dict[str, Any]:
        """Devuelve una representación apta para JSON."""

        raw = asdict(self)
        for key in ("db_path", "example_path", "artifacts_dir", "output_path"):
            raw[key] = str(raw[key])
        return raw
