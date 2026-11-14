from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .baselines import PopularityRecommender
from .config import EaseConfig, PipelineConfig
from .data import DATE_COL, ITEM_COL, RATING_COL, USER_COL, object_ids, observed_catalog
from .metrics import ndcg_at_k
from .split_data import SplitData, example_users, validation_folds

LOGGER = logging.getLogger(__name__)
EASE_HYPERPARAMETERS_FILE = "ease_hyperparameters.json"
TOP_K = 20

MAX_ITEMS = 12_000
_SCORE_BATCH = 512


@dataclass(frozen=True)
class EaseParameters:
    reg_lambda: float
    min_interactions: int
    run_name: str = ""


@dataclass(frozen=True)
class EaseSpectrum:
    item_ids: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


def ease_hyperparameters_path(artifacts_dir: Path) -> Path:
    return Path(artifacts_dir) / EASE_HYPERPARAMETERS_FILE


def has_saved_ease_hyperparameters(artifacts_dir: Path) -> bool:
    return ease_hyperparameters_path(artifacts_dir).is_file()


def load_ease_hyperparameters(path: Path) -> EaseParameters:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("el archivo no es un objeto")
        reg_lambda = float(payload["reg_lambda"])
        min_interactions = int(payload["min_interactions"])
        run_name = payload.get("run_name", "")
        if not isinstance(run_name, str):
            raise TypeError("run_name debe ser texto")
        if reg_lambda <= 0 or min_interactions < 1:
            raise ValueError("reg_lambda y min_interactions deben ser positivos")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Hiperparámetros de EASE inválidos en {path}") from exc
    if "weight_mode" in payload:
        LOGGER.info(
            "Se ignora weight_mode=%s en %s: la matriz usa el rating y la consulta, la recencia",
            payload["weight_mode"],
            path,
        )
    return EaseParameters(
        reg_lambda=reg_lambda,
        min_interactions=min_interactions,
        run_name=run_name,
    )


def parameters_from_config(config: EaseConfig) -> EaseParameters:
    return EaseParameters(
        reg_lambda=float(config.reg_lambda),
        min_interactions=int(config.min_interactions),
    )


def ease_source_parameters(config: PipelineConfig) -> EaseParameters:
    path = ease_hyperparameters_path(config.artifacts_dir)
    if path.is_file():
        return load_ease_hyperparameters(path)
    return parameters_from_config(config.ease)


def resolve_ease_parameters(config: PipelineConfig) -> tuple[EaseParameters, str]:
    path = ease_hyperparameters_path(config.artifacts_dir)
    parameters = ease_source_parameters(config)
    if path.is_file():
        LOGGER.info(
            "Reusando EASE de %s (corrida %s, λ=%s, mínimo=%s)",
            path,
            parameters.run_name or "sin nombre",
            parameters.reg_lambda,
            parameters.min_interactions,
        )
        return parameters, "saved"
    LOGGER.info("Sin %s; EASE usa los hiperparámetros fijos", path)
    return parameters, "default"


def _profile_weights(frame: pd.DataFrame, half_life_days: float) -> np.ndarray:
    rating = frame[RATING_COL].to_numpy(dtype=np.float64)
    if half_life_days <= 0 or len(frame) == 0:
        return rating
    max_dates = frame.groupby(USER_COL, sort=False)[DATE_COL].transform("max")
    age_days = (max_dates - frame[DATE_COL]).dt.days.clip(lower=0).to_numpy()
    return rating * np.exp2(-age_days / half_life_days)


def interaction_matrix(
    frame: pd.DataFrame,
    *,
    min_interactions: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    counts = frame.groupby(ITEM_COL, sort=False).size()
    item_ids = np.array(
        sorted(counts[counts >= min_interactions].index.astype(str)),
        dtype=object,
    )
    if len(item_ids) > MAX_ITEMS:
        raise ValueError(
            f"EASE armaría una matriz densa de {len(item_ids)} libros. "
            f"Subí min_interactions (ahora {min_interactions}); el tope es {MAX_ITEMS}."
        )
    if len(item_ids) == 0:
        empty = sparse.csr_matrix((0, 0), dtype=np.float64)
        return empty, item_ids
    item_index = {item: index for index, item in enumerate(item_ids)}
    columns = np.fromiter(
        (item_index.get(item, -1) for item in frame[ITEM_COL].astype(str)),
        dtype=np.int64,
        count=len(frame),
    )
    mask = columns >= 0
    users = np.array(
        sorted(frame.loc[mask, USER_COL].astype(str).unique()),
        dtype=object,
    )
    user_index = {user: index for index, user in enumerate(users)}
    rows = frame.loc[mask, USER_COL].astype(str).map(user_index).to_numpy()
    weights = frame.loc[mask, RATING_COL].to_numpy(dtype=np.float64)
    matrix = sparse.csr_matrix(
        (weights, (rows, columns[mask])),
        shape=(len(users), len(item_ids)),
        dtype=np.float64,
    )
    return matrix, item_ids


def spectrum_from_matrix(
    matrix: sparse.csr_matrix,
    item_ids: np.ndarray,
) -> EaseSpectrum:
    if matrix.shape[1] == 0:
        return EaseSpectrum(
            item_ids=item_ids,
            eigenvalues=np.zeros(0, dtype=np.float64),
            eigenvectors=np.zeros((0, 0), dtype=np.float64),
        )
    gram = (matrix.T @ matrix).toarray()
    gram = (gram + gram.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    np.maximum(eigenvalues, 0.0, out=eigenvalues)
    return EaseSpectrum(
        item_ids=item_ids,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def score_users(
    matrix: sparse.csr_matrix, spectrum: EaseSpectrum, reg_lambda: float
) -> np.ndarray:
    n_users, n_items = matrix.shape
    if n_items == 0 or n_users == 0:
        return np.zeros((n_users, n_items), dtype=np.float64)
    if reg_lambda <= 0:
        raise ValueError("reg_lambda debe ser positivo")
    inverse = 1.0 / (spectrum.eigenvalues + reg_lambda)
    diagonal = np.square(spectrum.eigenvectors) @ inverse
    projected = matrix @ spectrum.eigenvectors
    reconstructed = (projected * inverse) @ spectrum.eigenvectors.T
    return matrix.toarray() - reconstructed / diagonal


def _matrix_for_users(
    frame: pd.DataFrame,
    users: list[str],
    item_ids: np.ndarray,
    *,
    half_life_days: float,
) -> sparse.csr_matrix:
    item_index = {item: index for index, item in enumerate(item_ids)}
    if not item_index:
        return sparse.csr_matrix((len(users), 0), dtype=np.float64)
    user_index = {user: index for index, user in enumerate(users)}
    columns = object_ids(frame[ITEM_COL]).map(item_index)
    rows = object_ids(frame[USER_COL]).map(user_index)
    mask = columns.notna() & rows.notna()
    weights = _profile_weights(frame, half_life_days)
    return sparse.csr_matrix(
        (
            weights[mask.to_numpy()],
            (
                rows[mask].to_numpy(dtype=np.int64),
                columns[mask].to_numpy(dtype=np.int64),
            ),
        ),
        shape=(len(users), len(item_ids)),
        dtype=np.float64,
    )


def _top_unseen(
    item_ids: np.ndarray,
    scores: np.ndarray,
    seen: set[str],
    n: int,
) -> list[tuple[str, float]]:
    if n <= 0 or len(item_ids) == 0:
        return []
    ranked_scores = scores.copy()
    for index, item in enumerate(item_ids):
        if item in seen:
            ranked_scores[index] = -np.inf
    order = np.lexsort((item_ids, -ranked_scores))
    chosen: list[tuple[str, float]] = []
    for index in order:
        if not np.isfinite(ranked_scores[index]):
            continue
        chosen.append((str(item_ids[index]), float(ranked_scores[index])))
        if len(chosen) == n:
            break
    return chosen


class _SpectrumCache:
    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self.spectrum: EaseSpectrum | None = None

    def get(
        self,
        frame: pd.DataFrame,
        *,
        min_interactions: int,
    ) -> EaseSpectrum:
        key = (id(frame), min_interactions)
        if key == self._key and self.spectrum is not None:
            return self.spectrum
        matrix, item_ids = interaction_matrix(
            frame,
            min_interactions=min_interactions,
        )
        LOGGER.info(
            "EASE factoriza %s libros (mínimo %s)",
            len(item_ids),
            min_interactions,
        )
        spectrum = spectrum_from_matrix(matrix, item_ids)
        self._key = key
        self.spectrum = spectrum
        return spectrum


def ease_ranking(
    history: pd.DataFrame,
    users: list[str],
    spectrum: EaseSpectrum,
    parameters: EaseParameters,
    *,
    half_life_days: float,
    n: int,
) -> pd.DataFrame:
    columns = [USER_COL, ITEM_COL, "score", "rank"]
    if not users or n <= 0 or len(spectrum.item_ids) == 0:
        return pd.DataFrame(columns=columns)
    unique = list(dict.fromkeys(users))
    matrix = _matrix_for_users(
        history,
        unique,
        spectrum.item_ids,
        half_life_days=half_life_days,
    )
    scored = np.zeros((len(unique), len(spectrum.item_ids)), dtype=np.float64)
    for start in range(0, len(unique), _SCORE_BATCH):
        stop = start + _SCORE_BATCH
        scored[start:stop] = score_users(
            matrix[start:stop],
            spectrum,
            parameters.reg_lambda,
        )
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    if history.empty:
        seen: dict[object, set[str]] = {}
    else:
        seen = history.groupby(USER_COL, sort=False)[ITEM_COL].agg(set).to_dict()
    rows: list[dict[str, object]] = []
    for user, row, total in zip(unique, scored, totals, strict=True):
        if total <= 0:
            continue
        chosen = _top_unseen(
            spectrum.item_ids,
            row,
            set(seen.get(user, set())),
            n,
        )
        for rank, (item, score) in enumerate(chosen, start=1):
            rows.append(
                {
                    USER_COL: user,
                    ITEM_COL: item,
                    "score": score,
                    "rank": rank,
                }
            )
    return pd.DataFrame.from_records(rows, columns=columns)


def _recommend_frame(
    history: pd.DataFrame,
    users: list[str],
    spectrum: EaseSpectrum,
    parameters: EaseParameters,
    popularity: PopularityRecommender,
    *,
    half_life_days: float,
    n: int,
) -> pd.DataFrame:
    columns = [USER_COL, ITEM_COL, "score", "rank"]
    if not users:
        return pd.DataFrame(columns=columns)
    ranked = ease_ranking(
        history,
        users,
        spectrum,
        parameters,
        half_life_days=half_life_days,
        n=n,
    )
    by_user: dict[object, list[tuple[str, float]]] = {}
    if not ranked.empty:
        for user, group in ranked.groupby(USER_COL, sort=False):
            by_user[user] = list(
                zip(group[ITEM_COL].tolist(), group["score"].tolist(), strict=True)
            )
    unique = list(dict.fromkeys(users))
    fallback = popularity.recommend(unique, history, n=n)
    if fallback.empty:
        fallback_by_user: dict[object, list[str]] = {}
    else:
        fallback_by_user = fallback.groupby(USER_COL)[ITEM_COL].agg(list).to_dict()
    if history.empty:
        seen: dict[object, set[object]] = {}
    else:
        seen = history.groupby(USER_COL, sort=False)[ITEM_COL].agg(set).to_dict()
    rows: list[dict[str, object]] = []
    for user in users:
        chosen = list(by_user.get(user, []))
        already = {item for item, _score in chosen}
        for item in fallback_by_user.get(user, []):
            if len(chosen) >= n:
                break
            if item in already or item in seen.get(user, set()):
                continue
            chosen.append((item, float("nan")))
            already.add(item)
        for rank, (item, score) in enumerate(chosen, start=1):
            rows.append(
                {
                    USER_COL: user,
                    ITEM_COL: item,
                    "score": score,
                    "rank": rank,
                }
            )
    return pd.DataFrame.from_records(rows, columns=columns)


def _study_population(
    config: PipelineConfig,
    query_users: tuple[str, ...],
    target: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
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
    truth = target.loc[target[USER_COL].astype(str).isin(allowed)].reset_index(
        drop=True
    )
    LOGGER.info(
        "El study puntúa el último fold en %s lectores de %s",
        len(present),
        config.example_path,
    )
    return present, truth


def _in_catalog(frame: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    catalog = set(books[ITEM_COL].astype(str))
    kept = frame[frame[ITEM_COL].astype(str).isin(catalog)].copy()
    kept[USER_COL] = kept[USER_COL].astype(str)
    kept[ITEM_COL] = kept[ITEM_COL].astype(str)
    return kept


def tune_ease(
    history: pd.DataFrame,
    books: pd.DataFrame,
    config: PipelineConfig,
) -> EaseParameters:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("El tuning de EASE requiere `optuna`") from exc

    folds = validation_folds(history, config)
    if len(folds) < 2:
        raise ValueError("Se requieren al menos dos folds para el study de EASE")
    train = _in_catalog(folds[-1].train, books)
    query_users, target = _study_population(
        config,
        folds[-1].query_users,
        folds[-1].target,
    )
    half_life = config.retrieval.recency_half_life_days
    cache = _SpectrumCache()
    search = config.ease.search

    def objective(trial: Any) -> float:
        reg_lambda = float(search.reg_lambda.suggest(trial, "reg_lambda"))
        min_interactions = int(
            search.min_interactions.suggest(trial, "min_interactions")
        )
        spectrum = cache.get(train, min_interactions=min_interactions)
        parameters = EaseParameters(reg_lambda, min_interactions)
        popularity = PopularityRecommender().fit(
            train,
            set(train[ITEM_COL].astype(str)),
        )
        recommendations = _recommend_frame(
            train,
            query_users,
            spectrum,
            parameters,
            popularity,
            half_life_days=half_life,
            n=TOP_K,
        )
        score = ndcg_at_k(recommendations, target, k=TOP_K)
        LOGGER.info(
            "EASE trial %s: λ=%.4g mínimo=%s ndcg=%.4f",
            trial.number,
            reg_lambda,
            min_interactions,
            score,
        )
        return score

    from .tracking import log_optuna_plots, optuna_callback, study_name, track_study

    stage = "ease"
    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(
        study_name=study_name(stage),
        direction="maximize",
        sampler=sampler,
    )
    with track_study(stage):
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.set_tag("model", "ease")
        study.optimize(
            objective,
            n_trials=config.optuna_trials,
            callbacks=[optuna_callback(stage)],
            show_progress_bar=True,
        )
        log_optuna_plots(study, stage)

    best = study.best_trial
    from .tracking import active_run_name

    return EaseParameters(
        reg_lambda=float(best.params["reg_lambda"]),
        min_interactions=int(best.params["min_interactions"]),
        run_name=active_run_name(),
    )


def _write_ease_hyperparameters(
    config: PipelineConfig, parameters: EaseParameters
) -> None:
    path = ease_hyperparameters_path(config.artifacts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    from .tracking import with_run_name

    path.write_text(
        json.dumps(
            with_run_name(
                {
                    "reg_lambda": parameters.reg_lambda,
                    "min_interactions": parameters.min_interactions,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Hiperparámetros de EASE guardados en %s", path)


def _log_selected(parameters: EaseParameters, n_items: int, origin: str) -> None:
    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.log_params(
        {
            "ease_selected.reg_lambda": parameters.reg_lambda,
            "ease_selected.min_interactions": parameters.min_interactions,
            "ease_selected.origin": origin,
            "ease_selected.n_items": n_items,
        }
    )
    mlflow.log_metric("ease_items", float(n_items))


class EaseModel:
    name = "ease"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.hyperparameters_origin = "default"
        self.hyperparameters_run_name = ""
        self.hyperparameters: dict[str, Any] = {"model": self.name}
        self._history: pd.DataFrame | None = None
        self._spectrum: EaseSpectrum | None = None
        self._parameters: EaseParameters | None = None
        self._popularity: PopularityRecommender | None = None

    def fit(self, split: SplitData, *, tune: bool = False) -> None:
        raw = split.history.copy()
        raw[USER_COL] = raw[USER_COL].astype(str)
        raw[ITEM_COL] = raw[ITEM_COL].astype(str)
        if tune:
            parameters = tune_ease(raw, split.books, self.config)
            self.hyperparameters_origin = "tune"
            self.hyperparameters_run_name = parameters.run_name
            _write_ease_hyperparameters(self.config, parameters)
        else:
            parameters, origin = resolve_ease_parameters(self.config)
            self.hyperparameters_origin = origin
            self.hyperparameters_run_name = (
                parameters.run_name if origin == "saved" else ""
            )
        history = _in_catalog(raw, split.books)
        matrix, item_ids = interaction_matrix(
            history,
            min_interactions=parameters.min_interactions,
        )
        LOGGER.info(
            "EASE entrena %s libros (λ=%s, mínimo=%s)",
            len(item_ids),
            parameters.reg_lambda,
            parameters.min_interactions,
        )
        books = split.books.copy()
        books[ITEM_COL] = books[ITEM_COL].astype(str)
        catalog = set(observed_catalog(history, books)[ITEM_COL].astype(str))
        self._history = history
        self._spectrum = spectrum_from_matrix(matrix, item_ids)
        self._parameters = parameters
        self._popularity = PopularityRecommender().fit(history, catalog)
        self.hyperparameters = {
            "model": self.name,
            "reg_lambda": parameters.reg_lambda,
            "min_interactions": parameters.min_interactions,
            "n_items": len(item_ids),
        }
        _log_selected(parameters, len(item_ids), self.hyperparameters_origin)

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        if (
            self._history is None
            or self._spectrum is None
            or self._parameters is None
            or self._popularity is None
        ):
            raise RuntimeError("El modelo debe entrenarse antes de recomendar")
        return _recommend_frame(
            self._history,
            [str(user) for user in users],
            self._spectrum,
            self._parameters,
            self._popularity,
            half_life_days=self.config.retrieval.recency_half_life_days,
            n=TOP_K,
        )
