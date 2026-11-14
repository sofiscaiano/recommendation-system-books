from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SearchRange:
    low: float
    high: float
    log: bool = False
    integer: bool = False

    def suggest(self, trial: Any, name: str) -> float | int:
        if self.integer:
            return trial.suggest_int(name, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(name, self.low, self.high, log=self.log)


def search_ranges(search: Any) -> dict[str, SearchRange]:
    return {item.name: getattr(search, item.name) for item in fields(search)}


@dataclass(frozen=True)
class SplitConfig:
    holdout_k: int = 20
    rolling_folds: int = 3
    rolling_horizon: int = 20
    min_history: int = 5


@dataclass(frozen=True)
class RetrievalSearch:
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
    rrf_k: float = 60.0
    recency_half_life_days: float = 365.0
    popularidad_source: bool = True
    item_cosine_source: bool = True
    als_source: bool = True
    bpr_source: bool = True
    tfidf_source: bool = True
    two_tower_source: bool = True
    ease_source: bool = True
    search: RetrievalSearch = field(default_factory=RetrievalSearch)

    def enabled_sources(self) -> tuple[str, ...]:
        flags = (
            ("popularidad", self.popularidad_source),
            ("item_cosine", self.item_cosine_source),
            ("als", self.als_source),
            ("bpr", self.bpr_source),
            ("tfidf", self.tfidf_source),
            ("two_tower", self.two_tower_source),
            ("ease", self.ease_source),
        )
        return tuple(name for name, enabled in flags if enabled)

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("rrf_k debe ser positivo")
        if not self.enabled_sources():
            raise ValueError("al menos una fuente de retrieval tiene que estar activa")


@dataclass(frozen=True)
class RankerSearch:
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
    colsample_bytree: SearchRange = field(default_factory=lambda: SearchRange(0.6, 1.0))
    reg_alpha: SearchRange = field(
        default_factory=lambda: SearchRange(1e-8, 10.0, log=True)
    )
    reg_lambda: SearchRange = field(
        default_factory=lambda: SearchRange(1e-8, 20.0, log=True)
    )


LABEL_GAIN_MODES = ("linear", "exponential")


@dataclass(frozen=True)
class EaseSearch:
    reg_lambda: SearchRange = field(
        default_factory=lambda: SearchRange(100.0, 100_000.0, log=True)
    )
    min_interactions: SearchRange = field(
        default_factory=lambda: SearchRange(10, 100, log=True, integer=True)
    )


@dataclass(frozen=True)
class EaseConfig:
    reg_lambda: float = 15_000.0
    min_interactions: int = 20
    search: EaseSearch = field(default_factory=EaseSearch)

    def __post_init__(self) -> None:
        if self.reg_lambda <= 0:
            raise ValueError("reg_lambda debe ser positivo")
        if self.min_interactions < 1:
            raise ValueError("min_interactions debe ser positivo")


@dataclass(frozen=True)
class ItemCosineSearch:
    half_life_days: SearchRange = field(
        default_factory=lambda: SearchRange(30, 1095, log=True)
    )


@dataclass(frozen=True)
class ItemCosineConfig:
    half_life_days: float = 365.0
    search: ItemCosineSearch = field(default_factory=ItemCosineSearch)

    def __post_init__(self) -> None:
        if self.half_life_days <= 0:
            raise ValueError("half_life_days debe ser positivo")


@dataclass(frozen=True)
class AlsSearch:
    factors: SearchRange = field(
        default_factory=lambda: SearchRange(16, 256, log=True, integer=True)
    )
    iterations: SearchRange = field(
        default_factory=lambda: SearchRange(5, 40, integer=True)
    )
    regularization: SearchRange = field(
        default_factory=lambda: SearchRange(1e-3, 10.0, log=True)
    )
    half_life_days: SearchRange = field(
        default_factory=lambda: SearchRange(30, 1095, log=True)
    )


@dataclass(frozen=True)
class AlsConfig:
    factors: int = 64
    iterations: int = 20
    regularization: float = 0.01
    half_life_days: float = 365.0
    search: AlsSearch = field(default_factory=AlsSearch)

    def __post_init__(self) -> None:
        if min(self.factors, self.iterations) < 1:
            raise ValueError("factors e iterations deben ser positivos")
        if self.regularization <= 0 or self.half_life_days <= 0:
            raise ValueError("regularization y half_life_days deben ser positivos")


@dataclass(frozen=True)
class BprSearch:
    factors: SearchRange = field(
        default_factory=lambda: SearchRange(16, 256, log=True, integer=True)
    )
    iterations: SearchRange = field(
        default_factory=lambda: SearchRange(10, 100, integer=True)
    )
    learning_rate: SearchRange = field(
        default_factory=lambda: SearchRange(1e-3, 0.1, log=True)
    )
    regularization: SearchRange = field(
        default_factory=lambda: SearchRange(1e-4, 1.0, log=True)
    )
    half_life_days: SearchRange = field(
        default_factory=lambda: SearchRange(30, 1095, log=True)
    )


@dataclass(frozen=True)
class BprConfig:
    factors: int = 64
    iterations: int = 20
    learning_rate: float = 0.01
    regularization: float = 0.01
    half_life_days: float = 365.0
    search: BprSearch = field(default_factory=BprSearch)

    def __post_init__(self) -> None:
        if min(self.factors, self.iterations) < 1:
            raise ValueError("factors e iterations deben ser positivos")
        if min(self.learning_rate, self.regularization, self.half_life_days) <= 0:
            raise ValueError(
                "learning_rate, regularization y half_life_days deben ser positivos"
            )


@dataclass(frozen=True)
class TfidfSearch:
    max_features: SearchRange = field(
        default_factory=lambda: SearchRange(5_000, 80_000, log=True, integer=True)
    )
    half_life_days: SearchRange = field(
        default_factory=lambda: SearchRange(30, 1095, log=True)
    )


@dataclass(frozen=True)
class TfidfConfig:
    max_features: int = 60_000
    half_life_days: float = 365.0
    search: TfidfSearch = field(default_factory=TfidfSearch)

    def __post_init__(self) -> None:
        if self.max_features < 1:
            raise ValueError("max_features debe ser positivo")
        if self.half_life_days <= 0:
            raise ValueError("half_life_days debe ser positivo")


@dataclass(frozen=True)
class TwoTowerSearch:
    embedding_dim: SearchRange = field(
        default_factory=lambda: SearchRange(32, 128, log=True, integer=True)
    )
    learning_rate: SearchRange = field(
        default_factory=lambda: SearchRange(1e-4, 3e-3, log=True)
    )
    temperature: SearchRange = field(
        default_factory=lambda: SearchRange(0.02, 0.2, log=True)
    )
    min_item_reads: SearchRange = field(
        default_factory=lambda: SearchRange(2, 20, integer=True)
    )


@dataclass(frozen=True)
class TwoTowerConfig:
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
    search: TwoTowerSearch = field(default_factory=TwoTowerSearch)

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
            raise ValueError(
                "temperatura y learning rate positivos; weight decay no negativo"
            )


@dataclass(frozen=True)
class RankerConfig:
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
    label_gain: str = "exponential"
    search: RankerSearch = field(default_factory=RankerSearch)

    def __post_init__(self) -> None:
        if self.label_gain not in LABEL_GAIN_MODES:
            raise ValueError("label_gain debe ser 'linear' o 'exponential'")


@dataclass(frozen=True)
class PipelineConfig:
    db_path: Path = Path("data.db")
    example_path: Path = Path("ejemplo.csv")
    artifacts_dir: Path = Path("artifacts")
    output_path: Path = Path("submission.csv")
    random_seed: int = 42
    run_note: str = ""
    model: str = "two_stage"
    optuna_trials: int = 30
    tune_on_example_users: bool = True
    split: SplitConfig = field(default_factory=SplitConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    ease: EaseConfig = field(default_factory=EaseConfig)
    two_tower: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    item_cosine: ItemCosineConfig = field(default_factory=ItemCosineConfig)
    als: AlsConfig = field(default_factory=AlsConfig)
    bpr: BprConfig = field(default_factory=BprConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)

    def as_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        for key in ("db_path", "example_path", "artifacts_dir", "output_path"):
            raw[key] = str(raw[key])
        return raw
