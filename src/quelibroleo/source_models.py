"""Fuentes clásicas como modelos de una etapa, cada una con su study.

`item_cosine`, `als`, `bpr` y `tfidf` se tunean solas, con el NDCG@20 del
último fold, y dejan su JSON. `two_stage` lee ese archivo para armar la
fuente y no lo pisa: en la fusión solo elige el peso. El retriever que entra
a la fusión no rellena con popularidad; el modelo de una etapa sí, igual que
EASE, para que el top-20 esté completo.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import PopularityRecommender
from .config import PipelineConfig, search_ranges
from .data import ITEM_COL, USER_COL, observed_catalog
from .metrics import ndcg_at_k
from .split_data import SplitData, example_users, validation_folds
from .two_stage.retrieval import (
    ImplicitMFRetriever,
    ItemCosineRetriever,
    TfidfRetriever,
)

LOGGER = logging.getLogger(__name__)
TOP_K = 20
SOURCE_MODEL_NAMES = ("item_cosine", "als", "bpr", "tfidf")


@dataclass(frozen=True)
class ItemCosineParameters:
    """Semivida del coseno. El perfil no tiene otros hiperparámetros."""

    half_life_days: float
    run_name: str = ""


@dataclass(frozen=True)
class AlsParameters:
    """Factores, pasos, regularización y semivida de ALS."""

    factors: int
    iterations: int
    regularization: float
    half_life_days: float
    run_name: str = ""


@dataclass(frozen=True)
class BprParameters:
    """Factores, pasos, learning rate, regularización y semivida de BPR."""

    factors: int
    iterations: int
    learning_rate: float
    regularization: float
    half_life_days: float
    run_name: str = ""


@dataclass(frozen=True)
class TfidfParameters:
    """Tope del vocabulario y semivida del perfil de texto."""

    max_features: int
    half_life_days: float
    run_name: str = ""


SourceParameters = ItemCosineParameters | AlsParameters | BprParameters | TfidfParameters

_FIELDS: dict[str, tuple[str, ...]] = {
    "item_cosine": ("half_life_days",),
    "als": ("factors", "iterations", "regularization", "half_life_days"),
    "bpr": ("factors", "iterations", "learning_rate", "regularization", "half_life_days"),
    "tfidf": ("max_features", "half_life_days"),
}
_INTEGERS = {
    "item_cosine": frozenset(),
    "als": frozenset({"factors", "iterations"}),
    "bpr": frozenset({"factors", "iterations"}),
    "tfidf": frozenset({"max_features"}),
}
_CLASSES: dict[str, type[SourceParameters]] = {
    "item_cosine": ItemCosineParameters,
    "als": AlsParameters,
    "bpr": BprParameters,
    "tfidf": TfidfParameters,
}


def source_hyperparameters_path(artifacts_dir: Path, name: str) -> Path:
    """Ruta del tuneo de una fuente. No es el JSON de two_stage ni el de otra."""

    _check_name(name)
    return Path(artifacts_dir) / f"{name}_hyperparameters.json"


def has_saved_source_hyperparameters(artifacts_dir: Path, name: str) -> bool:
    """Indica si hay un tuneo de esta fuente para reutilizar."""

    return source_hyperparameters_path(artifacts_dir, name).is_file()


def load_source_hyperparameters(path: Path, name: str) -> SourceParameters:
    """Lee el JSON de una fuente. Falla si el archivo existe pero no es válido."""

    _check_name(name)
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("el archivo no es un objeto")
        values = {
            field: _read_field(payload[field], field, integer=field in _INTEGERS[name])
            for field in _FIELDS[name]
        }
        run_name = payload.get("run_name", "")
        if not isinstance(run_name, str):
            raise TypeError("run_name debe ser texto")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Hiperparámetros de {name} inválidos en {path}") from exc
    return _CLASSES[name](**values, run_name=run_name)


def parameters_from_config(config: PipelineConfig, name: str) -> SourceParameters:
    """Valores fijos, los de una corrida sin archivo guardado."""

    _check_name(name)
    block = getattr(config, name)
    values = {field: getattr(block, field) for field in _FIELDS[name]}
    return _CLASSES[name](**values)


def resolve_source_parameters(
    config: PipelineConfig,
    name: str,
) -> tuple[SourceParameters, str]:
    """Devuelve el tuneo de la fuente, o los valores fijos si ese archivo no existe."""

    path = source_hyperparameters_path(config.artifacts_dir, name)
    if path.is_file():
        return load_source_hyperparameters(path, name), "saved"
    return parameters_from_config(config, name), "default"


def item_cosine_source_parameters(config: PipelineConfig) -> ItemCosineParameters:
    """Semivida del coseno, como modelo único o como fuente de two_stage."""

    parameters, _origin = resolve_source_parameters(config, "item_cosine")
    assert isinstance(parameters, ItemCosineParameters)
    return parameters


def als_source_parameters(config: PipelineConfig) -> AlsParameters:
    """Parámetros de ALS. two_stage no los vuelve a buscar."""

    parameters, _origin = resolve_source_parameters(config, "als")
    assert isinstance(parameters, AlsParameters)
    return parameters


def bpr_source_parameters(config: PipelineConfig) -> BprParameters:
    """Parámetros de BPR. No se comparten con ALS."""

    parameters, _origin = resolve_source_parameters(config, "bpr")
    assert isinstance(parameters, BprParameters)
    return parameters


def tfidf_source_parameters(config: PipelineConfig) -> TfidfParameters:
    """Tope de términos y semivida de TF-IDF."""

    parameters, _origin = resolve_source_parameters(config, "tfidf")
    assert isinstance(parameters, TfidfParameters)
    return parameters


def tune_source(
    name: str,
    history: pd.DataFrame,
    books: pd.DataFrame,
    config: PipelineConfig,
) -> SourceParameters:
    """Elige los hiperparámetros de una fuente maximizando el NDCG@20 del último fold."""

    _check_name(name)
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(f"El tuning de {name} requiere `optuna`") from exc

    folds = validation_folds(history, config)
    if len(folds) < 2:
        raise ValueError(f"Se requieren al menos dos folds para el study de {name}")
    fold = folds[-1]
    train = _in_catalog(fold.train, books)
    users, target = _study_population(config, fold.query_users, fold.target)
    catalog = _catalog(train, books)
    popularity = PopularityRecommender().fit(train, set(catalog[ITEM_COL].astype(str)))
    search = getattr(config, name).search
    vectorizers: dict[int, TfidfRetriever] = {}

    def objective(trial: Any) -> float:
        """NDCG@20 del último fold. El vectorizer de TF-IDF se reusa si solo cambia la semivida."""

        parameters = _from_suggestion(name, search, trial)
        retriever = _fit_for_trial(
            name,
            parameters,
            train,
            catalog,
            random_state=config.random_seed,
            vectorizers=vectorizers,
        )
        ranked = retriever.recommend(users, train, TOP_K)
        recommendations = fill_with_popularity(ranked, popularity, train, users, TOP_K)
        score = ndcg_at_k(recommendations, target, k=TOP_K)
        LOGGER.info(
            "%s trial %s: %s ndcg=%.4f",
            name,
            trial.number,
            _public_dict(parameters),
            score,
        )
        return score

    from .tracking import log_optuna_plots, optuna_callback, study_name, track_study

    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(
        study_name=study_name(name),
        direction="maximize",
        sampler=sampler,
    )
    with track_study(name):
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.set_tag("model", name)
        study.optimize(
            objective,
            n_trials=config.optuna_trials,
            callbacks=[optuna_callback(name)],
            show_progress_bar=True,
        )
        log_optuna_plots(study, name)

    from .tracking import active_run_name

    return _from_suggestion(
        name,
        search,
        study.best_trial,
        run_name=active_run_name(),
    )


def fill_with_popularity(
    ranked: pd.DataFrame,
    popularity: PopularityRecommender,
    history: pd.DataFrame,
    users: list[str],
    n: int,
) -> pd.DataFrame:
    """Top-n de la fuente y, si no alcanza, popularidad sin repetir."""

    columns = [USER_COL, ITEM_COL, "score", "rank"]
    by_user: dict[str, list[tuple[str, float]]] = {}
    if history.empty:
        seen: dict[str, set[str]] = {}
    else:
        seen = {
            str(user): {str(item) for item in items}
            for user, items in history.groupby(USER_COL, sort=False)[ITEM_COL].agg(set).items()
        }
    if not ranked.empty:
        for user, group in ranked.groupby(USER_COL, sort=False):
            already_read = seen.get(str(user), set())
            picked: list[tuple[str, float]] = []
            taken: set[str] = set()
            for item, score in zip(
                group[ITEM_COL].astype(str).tolist(),
                group["score"].astype(float).tolist(),
                strict=True,
            ):
                if item in already_read or item in taken or not math.isfinite(score):
                    continue
                picked.append((item, score))
                taken.add(item)
            by_user[str(user)] = picked
    unique = list(dict.fromkeys(str(user) for user in users))
    requested = [str(user) for user in users]
    fallback = popularity.recommend(unique, history, n=n)
    if fallback.empty:
        fallback_by_user: dict[str, list[str]] = {}
    else:
        fallback_by_user = {
            str(user): [str(item) for item in items]
            for user, items in fallback.groupby(USER_COL)[ITEM_COL].agg(list).items()
        }
    rows: list[dict[str, object]] = []
    for user in requested:
        chosen = list(by_user.get(user, []))
        already = {item for item, _score in chosen}
        for item in fallback_by_user.get(user, []):
            if len(chosen) >= n:
                break
            if item in already:
                continue
            chosen.append((item, float("nan")))
            already.add(item)
        for rank, (item, score) in enumerate(chosen[:n], start=1):
            rows.append(
                {USER_COL: user, ITEM_COL: item, "score": score, "rank": rank}
            )
    return pd.DataFrame.from_records(rows, columns=columns)


def _without_seen_or_padding(ranked: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Saca leídos, duplicados y el relleno de implicit cuando se pide más que el catálogo.

    `recommend` de implicit completa hasta N con el mínimo de float32 en los ítems
    ya vistos y, si no alcanza, repite el ítem 0 con score 0.
    """

    frame = ranked.copy()
    frame[USER_COL] = frame[USER_COL].astype(str)
    frame[ITEM_COL] = frame[ITEM_COL].astype(str)
    scores = pd.to_numeric(frame["score"], errors="coerce")
    # El filtro de implicit marca lo ya visto con el mínimo de float32.
    frame = frame.loc[scores.notna() & (scores > np.finfo(np.float32).min / 2)]
    if not history.empty:
        seen = (
            history.astype({USER_COL: str, ITEM_COL: str})
            .groupby(USER_COL, sort=False)[ITEM_COL]
            .agg(set)
            .to_dict()
        )
        keep = [
            item not in seen.get(user, set())
            for user, item in zip(frame[USER_COL], frame[ITEM_COL], strict=True)
        ]
        frame = frame.loc[keep]
    return frame.drop_duplicates([USER_COL, ITEM_COL], keep="first")


class SourceModel:
    """Recomendador de una etapa. `fit` y `recommend` no pasan por two_stage."""

    def __init__(self, name: str, config: PipelineConfig) -> None:
        _check_name(name)
        self.name = name
        self.config = config
        self.hyperparameters_origin = "default"
        self.hyperparameters_run_name = ""
        self.hyperparameters: dict[str, Any] = {"model": name}
        self._history: pd.DataFrame | None = None
        self._retriever: ItemCosineRetriever | ImplicitMFRetriever | TfidfRetriever | None = None
        self._popularity: PopularityRecommender | None = None
        self._parameters: SourceParameters | None = None

    def fit(self, split: SplitData, *, tune: bool = False) -> None:
        """Ajusta la fuente con `split.history`. El último fold entra si `tune` es verdadero."""

        raw = split.history.copy()
        raw[USER_COL] = raw[USER_COL].astype(str)
        raw[ITEM_COL] = raw[ITEM_COL].astype(str)
        if tune:
            parameters = tune_source(self.name, raw, split.books, self.config)
            self.hyperparameters_origin = "tune"
            self.hyperparameters_run_name = parameters.run_name
            _write_hyperparameters(self.config, self.name, parameters)
        else:
            parameters, origin = resolve_source_parameters(self.config, self.name)
            self.hyperparameters_origin = origin
            self.hyperparameters_run_name = parameters.run_name if origin == "saved" else ""
        history = _in_catalog(raw, split.books)
        catalog = _catalog(history, split.books)
        LOGGER.info("%s entrena con %s", self.name, _public_dict(parameters))
        self._history = history
        self._retriever = _fit_retriever(
            self.name,
            parameters,
            history,
            catalog,
            random_state=self.config.random_seed,
        )
        self._popularity = PopularityRecommender().fit(
            history,
            set(catalog[ITEM_COL].astype(str)),
        )
        self._parameters = parameters
        self.hyperparameters = {"model": self.name, **_public_dict(parameters)}
        _log_selected(self.name, parameters, self.hyperparameters_origin)

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        """Top-20, sin libros ya leídos. Popularidad completa si la fuente no llega."""

        if self._history is None or self._retriever is None or self._popularity is None:
            raise RuntimeError("El modelo debe entrenarse antes de recomendar")
        user_ids = [str(user) for user in users]
        ranked = self._retriever.recommend(user_ids, self._history, TOP_K)
        return fill_with_popularity(
            ranked,
            self._popularity,
            self._history,
            user_ids,
            TOP_K,
        )


def _check_name(name: str) -> None:
    if name not in _CLASSES:
        known = ", ".join(SOURCE_MODEL_NAMES)
        raise ValueError(f"Fuente sin study propio: {name}. Opciones: {known}")


def _read_field(value: Any, name: str, *, integer: bool) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} debe ser un número")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} debe ser finito")
    if integer:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{name} debe ser entero")
        number = int(value)
        if number < 1:
            raise ValueError(f"{name} debe ser positivo")
        return number
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} debe ser positivo")
    return number


def _public_dict(parameters: SourceParameters) -> dict[str, Any]:
    name = _name_of(parameters)
    return {field: getattr(parameters, field) for field in _FIELDS[name]}


def _name_of(parameters: SourceParameters) -> str:
    for name, cls in _CLASSES.items():
        if isinstance(parameters, cls):
            return name
    raise TypeError("parámetros de una fuente desconocida")


def _from_suggestion(
    name: str,
    search: Any,
    trial: Any,
    *,
    run_name: str = "",
) -> SourceParameters:
    """Arma los parámetros con lo que Optuna ya eligió, o pidiéndoselo al trial."""

    chosen: dict[str, Any] = {}
    stored = getattr(trial, "params", None)
    for field, spec in search_ranges(search).items():
        if stored is not None and field in stored:
            raw = stored[field]
        else:
            raw = spec.suggest(trial, field)
        chosen[field] = int(raw) if field in _INTEGERS[name] else float(raw)
    return _CLASSES[name](**chosen, run_name=run_name)


def _fit_retriever(
    name: str,
    parameters: SourceParameters,
    history: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    random_state: int,
) -> ItemCosineRetriever | ImplicitMFRetriever | TfidfRetriever:
    retriever = _new_retriever(name, parameters, random_state=random_state)
    retriever.fit(history, catalog)
    return retriever


def _fit_for_trial(
    name: str,
    parameters: SourceParameters,
    history: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    random_state: int,
    vectorizers: dict[int, TfidfRetriever],
) -> ItemCosineRetriever | ImplicitMFRetriever | TfidfRetriever:
    if name != "tfidf":
        return _fit_retriever(
            name,
            parameters,
            history,
            catalog,
            random_state=random_state,
        )
    assert isinstance(parameters, TfidfParameters)
    cached = vectorizers.get(parameters.max_features)
    if cached is None:
        cached = TfidfRetriever(
            max_features=parameters.max_features,
            half_life_days=parameters.half_life_days,
        )
        cached.fit(history, catalog)
        vectorizers[parameters.max_features] = cached
    cached.half_life_days = parameters.half_life_days
    return cached


def _new_retriever(
    name: str,
    parameters: SourceParameters,
    *,
    random_state: int,
) -> ItemCosineRetriever | ImplicitMFRetriever | TfidfRetriever:
    if isinstance(parameters, ItemCosineParameters):
        return ItemCosineRetriever(half_life_days=parameters.half_life_days)
    if isinstance(parameters, AlsParameters):
        return ImplicitMFRetriever(
            "als",
            factors=parameters.factors,
            iterations=parameters.iterations,
            regularization=parameters.regularization,
            half_life_days=parameters.half_life_days,
            random_state=random_state,
        )
    if isinstance(parameters, BprParameters):
        return ImplicitMFRetriever(
            "bpr",
            factors=parameters.factors,
            iterations=parameters.iterations,
            learning_rate=parameters.learning_rate,
            regularization=parameters.regularization,
            half_life_days=parameters.half_life_days,
            random_state=random_state,
        )
    assert isinstance(parameters, TfidfParameters)
    return TfidfRetriever(
        max_features=parameters.max_features,
        half_life_days=parameters.half_life_days,
    )


def _study_population(
    config: PipelineConfig,
    query_users: tuple[str, ...],
    target: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    """Lectores del objetivo. Con el flag, solo los de ejemplo.csv."""

    users = [str(user) for user in query_users]
    if not config.tune_on_example_users:
        return users, target
    allowed = example_users(config)
    present = [user for user in users if user in allowed]
    if not present:
        raise ValueError(
            "Ningún lector de ejemplo.csv entra al último fold; "
            "no se puede tunear solo sobre la entrega."
        )
    truth = target.loc[target[USER_COL].astype(str).isin(allowed)].reset_index(drop=True)
    LOGGER.info(
        "El study puntúa el último fold en %s lectores de %s",
        len(present),
        config.example_path,
    )
    return present, truth


def _in_catalog(frame: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """Deja solo libros del catálogo. El corte temporal no los saca."""

    catalog = set(books[ITEM_COL].astype(str))
    kept = frame[frame[ITEM_COL].astype(str).isin(catalog)].copy()
    kept[USER_COL] = kept[USER_COL].astype(str)
    kept[ITEM_COL] = kept[ITEM_COL].astype(str)
    return kept


def _catalog(history: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    listed = books.copy()
    listed[ITEM_COL] = listed[ITEM_COL].astype(str)
    return observed_catalog(history, listed)


def _write_hyperparameters(
    config: PipelineConfig,
    name: str,
    parameters: SourceParameters,
) -> None:
    """Guarda el study de la fuente sin tocar el JSON de two_stage ni el de otra."""

    path = source_hyperparameters_path(config.artifacts_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    from .tracking import with_run_name

    path.write_text(
        json.dumps(with_run_name(_public_dict(parameters)), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Hiperparámetros de %s guardados en %s", name, path)


def _log_selected(name: str, parameters: SourceParameters, origin: str) -> None:
    """Deja en la corrida activa qué fuente se usó."""

    import mlflow

    if mlflow.active_run() is None:
        return
    payload = {f"{name}_selected.{key}": value for key, value in _public_dict(parameters).items()}
    payload[f"{name}_selected.origin"] = origin
    mlflow.log_params(payload)


TWO_TOWER_HYPERPARAMETERS_FILE = "two_tower_hyperparameters.json"
_TWO_TOWER_FIELDS = ("embedding_dim", "learning_rate", "temperature", "min_item_reads")
_TWO_TOWER_INTEGERS = frozenset({"embedding_dim", "min_item_reads"})


@dataclass(frozen=True)
class TwoTowerParameters:
    """Lo que elige el study del two-tower. El resto queda en ``TwoTowerConfig``.

    Vive acá, y no en ``two_tower.py``, para que ``two_stage`` arme la fuente
    sin importar torch en el mismo proceso que LightGBM.
    """

    embedding_dim: int
    learning_rate: float
    temperature: float
    min_item_reads: int
    run_name: str = ""


def two_tower_hyperparameters_path(artifacts_dir: Path) -> Path:
    """Ruta del tuneo del two-tower. No es el JSON de two_stage."""

    return Path(artifacts_dir) / TWO_TOWER_HYPERPARAMETERS_FILE


def has_saved_two_tower_hyperparameters(artifacts_dir: Path) -> bool:
    """Indica si hay un tuneo del two-tower para reutilizar."""

    return two_tower_hyperparameters_path(artifacts_dir).is_file()


def load_two_tower_hyperparameters(path: Path) -> TwoTowerParameters:
    """Lee el JSON del two-tower. Falla si el archivo existe pero no es válido."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("el archivo no es un objeto")
        values = {
            field: _read_field(
                payload[field],
                field,
                integer=field in _TWO_TOWER_INTEGERS,
            )
            for field in _TWO_TOWER_FIELDS
        }
        run_name = payload.get("run_name", "")
        if not isinstance(run_name, str):
            raise TypeError("run_name debe ser texto")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Hiperparámetros de two_tower inválidos en {path}") from exc
    return TwoTowerParameters(**values, run_name=run_name)


def apply_two_tower_parameters(
    config: PipelineConfig,
    parameters: TwoTowerParameters,
) -> PipelineConfig:
    """Devuelve la config con los cuatro valores del study ya puestos."""

    block = replace(
        config.two_tower,
        embedding_dim=int(parameters.embedding_dim),
        learning_rate=float(parameters.learning_rate),
        temperature=float(parameters.temperature),
        min_item_reads=int(parameters.min_item_reads),
    )
    return replace(config, two_tower=block)


def resolved_two_tower_config(config: PipelineConfig) -> tuple[PipelineConfig, str, str]:
    """Config con el JSON ya aplicado. Sin archivo, devuelve la config tal cual.

    El proceso hijo recibe esta config y no vuelve a abrir el JSON.
    """

    path = two_tower_hyperparameters_path(config.artifacts_dir)
    if not path.is_file():
        return config, "default", ""
    saved = load_two_tower_hyperparameters(path)
    return apply_two_tower_parameters(config, saved), "saved", saved.run_name


def two_tower_resolved_values(config: PipelineConfig) -> dict[str, Any]:
    """Valores con los que se entrena la torre, sin el rango de búsqueda."""

    resolved, _origin, _run_name = resolved_two_tower_config(config)
    values = asdict(resolved.two_tower)
    values.pop("search", None)
    return values


def write_two_tower_hyperparameters(
    config: PipelineConfig,
    parameters: TwoTowerParameters,
) -> None:
    """Guarda el study del two-tower sin tocar el JSON de two_stage."""

    path = two_tower_hyperparameters_path(config.artifacts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    from .tracking import with_run_name

    payload = {field: getattr(parameters, field) for field in _TWO_TOWER_FIELDS}
    path.write_text(
        json.dumps(with_run_name(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Hiperparámetros de two_tower guardados en %s", path)
