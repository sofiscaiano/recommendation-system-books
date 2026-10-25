"""Two-tower colaborativo, de una sola etapa.

La torre de ítem es un embedding. La torre de usuario proyecta el promedio de
los embeddings de los libros leídos, pesado por rating, y deja afuera el libro
que se está prediciendo. El score es el coseno. El estudio de Optuna escribe
su propio JSON y no toca el de two_stage ni el de EASE.
"""

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
from .config import PipelineConfig, TwoTowerConfig
from .data import ITEM_COL, RATING_COL, USER_COL, observed_catalog
from .metrics import ndcg_at_k
from .split_data import SplitData, validation_folds

LOGGER = logging.getLogger(__name__)
TWO_TOWER_HYPERPARAMETERS_FILE = "two_tower_hyperparameters.json"
TOP_K = 20
_ADAM_BETA1 = 0.9
_ADAM_BETA2 = 0.999
_ADAM_EPS = 1e-8


@dataclass(frozen=True)
class TwoTowerParameters:
    """Hiperparámetros con los que se entrena esta corrida."""

    embedding_dim: int
    learning_rate: float
    n_negatives: int
    temperature: float
    epochs: int
    run_name: str = ""


@dataclass
class _Tower:
    """Embeddings de ítem y proyección de usuario, con el estado de Adam."""

    item_ids: np.ndarray
    user_ids: np.ndarray
    embeddings: np.ndarray
    projection: np.ndarray
    adam_m: dict[str, np.ndarray]
    adam_v: dict[str, np.ndarray]
    adam_step: int = 0


@dataclass
class _Interactions:
    """Matriz usuario × ítem y las lecturas en el mismo orden de índices."""

    user_item: sparse.csr_matrix
    degrees: np.ndarray
    users: np.ndarray
    items: np.ndarray
    weights: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray


def two_tower_hyperparameters_path(artifacts_dir: Path) -> Path:
    """Ruta del tuneo del two-tower."""

    return Path(artifacts_dir) / TWO_TOWER_HYPERPARAMETERS_FILE


def has_saved_two_tower_hyperparameters(artifacts_dir: Path) -> bool:
    """Indica si hay un tuneo de two-tower para reutilizar."""

    return two_tower_hyperparameters_path(artifacts_dir).is_file()


def load_two_tower_hyperparameters(path: Path) -> TwoTowerParameters:
    """Lee el JSON del two-tower. Falla si el archivo existe pero no es válido."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("el archivo no es un objeto")
        embedding_dim = int(payload["embedding_dim"])
        learning_rate = float(payload["learning_rate"])
        n_negatives = int(payload["n_negatives"])
        temperature = float(payload["temperature"])
        epochs = int(payload["epochs"])
        run_name = payload.get("run_name", "")
        if not isinstance(run_name, str):
            raise TypeError("run_name debe ser texto")
        if min(embedding_dim, n_negatives, epochs) < 1:
            raise ValueError("dimensión, negativos y épocas deben ser positivos")
        if learning_rate <= 0 or temperature <= 0:
            raise ValueError("learning_rate y temperature deben ser positivos")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Hiperparámetros de two-tower inválidos en {path}") from exc
    return TwoTowerParameters(
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
        n_negatives=n_negatives,
        temperature=temperature,
        epochs=epochs,
        run_name=run_name,
    )


def parameters_from_config(config: TwoTowerConfig) -> TwoTowerParameters:
    """Valores fijos, los de una corrida sin archivo guardado."""

    return TwoTowerParameters(
        embedding_dim=config.embedding_dim,
        learning_rate=config.learning_rate,
        n_negatives=config.n_negatives,
        temperature=config.temperature,
        epochs=config.max_epochs,
    )


def resolve_two_tower_parameters(
    config: PipelineConfig,
) -> tuple[TwoTowerParameters, str]:
    """Devuelve el tuneo guardado, o los valores fijos si ese archivo no existe."""

    path = two_tower_hyperparameters_path(config.artifacts_dir)
    if path.is_file():
        loaded = load_two_tower_hyperparameters(path)
        LOGGER.info(
            "Reusando two-tower de %s (corrida %s, dim=%s, lr=%s, negativos=%s)",
            path,
            loaded.run_name or "sin nombre",
            loaded.embedding_dim,
            loaded.learning_rate,
            loaded.n_negatives,
        )
        return loaded, "saved"
    LOGGER.info("Sin %s; two-tower usa los hiperparámetros fijos", path)
    return parameters_from_config(config.two_tower), "default"


def _in_catalog(frame: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """Deja solo libros del catálogo. El corte temporal no los saca."""

    catalog = set(books[ITEM_COL].astype(str))
    kept = frame[frame[ITEM_COL].astype(str).isin(catalog)].copy()
    kept[USER_COL] = kept[USER_COL].astype(str)
    kept[ITEM_COL] = kept[ITEM_COL].astype(str)
    return kept


def _interactions(frame: pd.DataFrame) -> _Interactions:
    """Arma la matriz dispersa. El peso de cada lectura es el rating."""

    user_ids = np.array(sorted(frame[USER_COL].astype(str).unique()), dtype=object)
    item_ids = np.array(sorted(frame[ITEM_COL].astype(str).unique()), dtype=object)
    user_index = {user: index for index, user in enumerate(user_ids)}
    item_index = {item: index for index, item in enumerate(item_ids)}
    users = frame[USER_COL].astype(str).map(user_index).to_numpy(dtype=np.int64)
    items = frame[ITEM_COL].astype(str).map(item_index).to_numpy(dtype=np.int64)
    weights = frame[RATING_COL].to_numpy(dtype=np.float64)
    matrix = sparse.csr_matrix(
        (weights, (users, items)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float64,
    )
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    return _Interactions(
        matrix,
        degrees,
        users,
        items,
        weights,
        user_ids,
        item_ids,
    )


def _normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normaliza por la última dimensión y devuelve también la norma."""

    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-8), norm


def _normalize_backward(
    normalized: np.ndarray,
    gradient: np.ndarray,
    norm: np.ndarray,
) -> np.ndarray:
    """Propaga el gradiente a través de la normalización L2."""

    dot = np.sum(gradient * normalized, axis=-1, keepdims=True)
    raw = (gradient - normalized * dot) / np.maximum(norm, 1e-8)
    raw[np.squeeze(norm, axis=-1) < 1e-8] = 0.0
    return raw


def _new_tower(
    item_ids: np.ndarray,
    user_ids: np.ndarray,
    embedding_dim: int,
    rng: np.random.Generator,
) -> _Tower:
    """Inicializa embeddings chicos y la proyección como identidad."""

    scale = 0.1
    embeddings = rng.normal(0.0, scale, size=(len(item_ids), embedding_dim))
    projection = np.eye(embedding_dim, dtype=np.float64)
    return _Tower(
        item_ids=item_ids,
        user_ids=user_ids,
        embeddings=embeddings,
        projection=projection,
        adam_m={},
        adam_v={},
    )


def _adam(tower: _Tower, name: str, gradient: np.ndarray, learning_rate: float, decay: float) -> np.ndarray:
    """Un paso de AdamW para embeddings o para la proyección."""

    moment = tower.adam_m.get(name)
    velocity = tower.adam_v.get(name)
    if moment is None or velocity is None:
        moment = np.zeros_like(gradient)
        velocity = np.zeros_like(gradient)
    moment = _ADAM_BETA1 * moment + (1.0 - _ADAM_BETA1) * gradient
    velocity = _ADAM_BETA2 * velocity + (1.0 - _ADAM_BETA2) * np.square(gradient)
    tower.adam_m[name] = moment
    tower.adam_v[name] = velocity
    step = tower.adam_step
    corrected_m = moment / (1.0 - _ADAM_BETA1**step)
    corrected_v = velocity / (1.0 - _ADAM_BETA2**step)
    parameter = tower.embeddings if name == "embeddings" else tower.projection
    adapted = corrected_m / (np.sqrt(corrected_v) + _ADAM_EPS)
    return parameter - learning_rate * (adapted + decay * parameter)


def _training_step(
    tower: _Tower,
    interactions: _Interactions,
    batch: np.ndarray,
    *,
    n_negatives: int,
    temperature: float,
    learning_rate: float,
    weight_decay: float,
    rng: np.random.Generator,
) -> float:
    """Un batch de softmax con negativos aleatorios. Devuelve la loss media."""

    users = interactions.users[batch]
    positives = interactions.items[batch]
    positive_weight = interactions.weights[batch]
    unique, inverse = np.unique(users, return_inverse=True)
    pooled = np.asarray(interactions.user_item[unique] @ tower.embeddings)
    history = pooled[inverse]
    degree = interactions.degrees[users]
    numerator = history - positive_weight[:, None] * tower.embeddings[positives]
    denominator = degree - positive_weight
    valid = denominator > 1e-8
    if not np.any(valid):
        return 0.0
    hidden = np.zeros_like(numerator)
    hidden[valid] = numerator[valid] / denominator[valid, None]
    raw_user, user_norm = _normalize(hidden @ tower.projection)
    raw_positive = tower.embeddings[positives]
    positive_vector, positive_norm = _normalize(raw_positive)
    negatives = rng.integers(0, tower.embeddings.shape[0], size=(len(batch), n_negatives))
    negative_vector, negative_norm = _normalize(tower.embeddings[negatives])

    positive_score = np.sum(raw_user * positive_vector, axis=1) / temperature
    negative_score = np.einsum("bd,bkd->bk", raw_user, negative_vector) / temperature
    logits = np.concatenate([positive_score[:, None], negative_score], axis=1)
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    residual = probabilities.copy()
    residual[:, 0] -= 1.0
    residual[~valid] = 0.0
    residual /= temperature

    user_gradient = residual[:, :1] * positive_vector + np.einsum(
        "bk,bkd->bd",
        residual[:, 1:],
        negative_vector,
    )
    positive_gradient = residual[:, :1] * raw_user
    negative_gradient = residual[:, 1:, None] * raw_user[:, None, :]
    embedding_gradient = np.zeros_like(tower.embeddings)
    np.add.at(
        embedding_gradient,
        positives,
        _normalize_backward(positive_vector, positive_gradient, positive_norm),
    )
    np.add.at(
        embedding_gradient,
        negatives.reshape(-1),
        _normalize_backward(negative_vector, negative_gradient, negative_norm).reshape(
            -1,
            tower.embeddings.shape[1],
        ),
    )

    hidden_gradient = _normalize_backward(raw_user, user_gradient, user_norm) @ tower.projection.T
    pool_gradient = np.zeros_like(hidden_gradient)
    pool_gradient[valid] = hidden_gradient[valid] / denominator[valid, None]
    unique_gradient = np.zeros((len(unique), tower.embeddings.shape[1]))
    np.add.at(unique_gradient, inverse, pool_gradient)
    embedding_gradient += np.asarray(interactions.user_item[unique].T @ unique_gradient)
    np.add.at(embedding_gradient, positives, -positive_weight[:, None] * pool_gradient)
    projection_gradient = hidden.T @ _normalize_backward(raw_user, user_gradient, user_norm)

    tower.adam_step += 1
    tower.embeddings = _adam(
        tower,
        "embeddings",
        embedding_gradient,
        learning_rate,
        weight_decay,
    )
    tower.projection = _adam(
        tower,
        "projection",
        projection_gradient,
        learning_rate,
        weight_decay,
    )
    picked = probabilities[valid, 0]
    return float(-np.mean(np.log(np.maximum(picked, 1e-12)))) if len(picked) else 0.0


def _user_vectors(tower: _Tower, interactions: _Interactions, users: np.ndarray) -> np.ndarray:
    """Torre de usuario con toda la historia, sin dejar un libro afuera."""

    pooled = np.asarray(interactions.user_item[users] @ tower.embeddings)
    degree = interactions.degrees[users]
    hidden = np.zeros_like(pooled)
    active = degree > 1e-8
    hidden[active] = pooled[active] / degree[active, None]
    vectors, _ = _normalize(hidden @ tower.projection)
    vectors[~active] = 0.0
    return vectors


def _top_scores(
    item_ids: np.ndarray,
    item_index: dict[str, int],
    scores: np.ndarray,
    seen: set[str],
    n: int,
) -> list[tuple[str, float]]:
    """Top-n con desempate por id_libro. Los ya leídos quedan afuera."""

    ranked = scores.copy()
    for item in seen:
        index = item_index.get(item)
        if index is not None:
            ranked[index] = -np.inf
    finite = np.isfinite(ranked)
    available = int(finite.sum())
    if n <= 0 or available == 0:
        return []
    take = min(n, available)
    chosen = np.argpartition(ranked, -take)[-take:]
    chosen = chosen[np.lexsort((item_ids[chosen], -ranked[chosen]))]
    return [
        (str(item_ids[index]), float(ranked[index]))
        for index in chosen
        if np.isfinite(ranked[index])
    ]


def _recommend_from_tower(
    tower: _Tower,
    interactions: _Interactions,
    history: pd.DataFrame,
    users: list[str],
    popularity: PopularityRecommender,
    *,
    n: int,
) -> pd.DataFrame:
    """Top-n del coseno y, si no alcanza, popularidad."""

    if not users:
        return pd.DataFrame(columns=[USER_COL, ITEM_COL, "score", "rank"])
    user_index = {user: index for index, user in enumerate(tower.user_ids)}
    known = np.array([user_index.get(user, -1) for user in users], dtype=np.int64)
    vectors = np.zeros((len(users), tower.embeddings.shape[1]))
    present = known >= 0
    if np.any(present):
        vectors[present] = _user_vectors(tower, interactions, known[present])
    item_vectors, _ = _normalize(tower.embeddings)
    item_index = {item: index for index, item in enumerate(tower.item_ids)}
    seen = history.groupby(USER_COL, sort=False)[ITEM_COL].agg(set).to_dict()
    fallback = popularity.recommend(users, history, n=n)
    fallback_by_user = (
        {}
        if fallback.empty
        else fallback.groupby(USER_COL)[ITEM_COL].agg(list).to_dict()
    )
    rows: list[dict[str, object]] = []
    batch = 256
    for start in range(0, len(users), batch):
        stop = min(start + batch, len(users))
        scores = vectors[start:stop] @ item_vectors.T
        for offset, user in enumerate(users[start:stop]):
            chosen = _top_scores(
                tower.item_ids,
                item_index,
                scores[offset],
                set(seen.get(user, set())),
                n,
            )
            already = {item for item, _ in chosen}
            for item in fallback_by_user.get(user, []):
                if len(chosen) >= n:
                    break
                if item in already or item in seen.get(user, set()):
                    continue
                chosen.append((item, float("nan")))
                already.add(item)
            for rank, (item, score) in enumerate(chosen, start=1):
                rows.append({USER_COL: user, ITEM_COL: item, "score": score, "rank": rank})
    return pd.DataFrame.from_records(rows, columns=[USER_COL, ITEM_COL, "score", "rank"])


def _fit_tower(
    frame: pd.DataFrame,
    parameters: TwoTowerParameters,
    config: TwoTowerConfig,
    rng: np.random.Generator,
    *,
    validation: tuple[list[str], pd.DataFrame] | None = None,
) -> tuple[_Tower, _Interactions, int, float]:
    """Entrena hasta `parameters.epochs`. Con validación, corta por paciencia."""

    interactions = _interactions(frame)
    tower = _new_tower(
        interactions.item_ids,
        interactions.user_ids,
        parameters.embedding_dim,
        rng,
    )
    best_epochs = 1
    best_score = -1.0
    stalled = 0
    order = np.arange(len(interactions.users))
    for epoch in range(1, parameters.epochs + 1):
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            losses.append(
                _training_step(
                    tower,
                    interactions,
                    batch,
                    n_negatives=parameters.n_negatives,
                    temperature=parameters.temperature,
                    learning_rate=parameters.learning_rate,
                    weight_decay=config.weight_decay,
                    rng=rng,
                )
            )
        score = 0.0
        if validation is not None:
            query_users, target = validation
            popularity = PopularityRecommender().fit(frame, set(frame[ITEM_COL].astype(str)))
            recommendations = _recommend_from_tower(
                tower,
                interactions,
                frame,
                query_users,
                popularity,
                n=TOP_K,
            )
            score = ndcg_at_k(recommendations, target, k=TOP_K)
            LOGGER.info(
                "Two-tower época %s: loss=%.4f ndcg=%.4f",
                epoch,
                float(np.mean(losses)) if losses else 0.0,
                score,
            )
            if score > best_score + 1e-4:
                best_score = score
                best_epochs = epoch
                stalled = 0
            else:
                stalled += 1
                if stalled >= config.patience:
                    break
        else:
            best_epochs = epoch
            LOGGER.info(
                "Two-tower época %s: loss=%.4f",
                epoch,
                float(np.mean(losses)) if losses else 0.0,
            )
    return tower, interactions, best_epochs, best_score


def tune_two_tower(
    history: pd.DataFrame,
    books: pd.DataFrame,
    config: PipelineConfig,
) -> TwoTowerParameters:
    """Elige dimensión, learning rate, negativos y temperatura en el último fold."""

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("El tuning de two-tower requiere `optuna`") from exc

    folds = validation_folds(history, config)
    if len(folds) < 2:
        raise ValueError("Se requieren al menos dos folds para el study de two-tower")
    train = _in_catalog(folds[-1].train, books)
    target = folds[-1].target
    query_users = list(folds[-1].query_users)
    search = config.two_tower.search

    def objective(trial: Any) -> float:
        """NDCG@20 del último fold. Las épocas se cortan por paciencia."""

        parameters = TwoTowerParameters(
            embedding_dim=int(search.embedding_dim.suggest(trial, "embedding_dim")),
            learning_rate=float(search.learning_rate.suggest(trial, "learning_rate")),
            n_negatives=int(search.n_negatives.suggest(trial, "n_negatives")),
            temperature=float(search.temperature.suggest(trial, "temperature")),
            epochs=config.two_tower.max_epochs,
        )
        rng = np.random.default_rng(config.random_seed + trial.number)
        _, _, epochs, score = _fit_tower(
            train,
            parameters,
            config.two_tower,
            rng,
            validation=(query_users, target),
        )
        trial.set_user_attr("epochs", epochs)
        LOGGER.info(
            "Two-tower trial %s: dim=%s lr=%.4g negativos=%s temperatura=%.3g épocas=%s ndcg=%.4f",
            trial.number,
            parameters.embedding_dim,
            parameters.learning_rate,
            parameters.n_negatives,
            parameters.temperature,
            epochs,
            score,
        )
        return score

    from .tracking import (
        active_run_name,
        log_optuna_plots,
        optuna_callback,
        study_name,
        track_study,
    )

    stage = "two_tower"
    study = optuna.create_study(
        study_name=study_name(stage),
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.random_seed),
    )
    with track_study(stage):
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.set_tag("model", "two_tower")
        study.optimize(
            objective,
            n_trials=config.optuna_trials,
            callbacks=[optuna_callback(stage)],
            show_progress_bar=True,
        )
        log_optuna_plots(study, stage)

    best = study.best_trial
    return TwoTowerParameters(
        embedding_dim=int(best.params["embedding_dim"]),
        learning_rate=float(best.params["learning_rate"]),
        n_negatives=int(best.params["n_negatives"]),
        temperature=float(best.params["temperature"]),
        epochs=int(best.user_attrs["epochs"]),
        run_name=active_run_name(),
    )


def _write_two_tower_hyperparameters(
    config: PipelineConfig,
    parameters: TwoTowerParameters,
) -> None:
    """Guarda el study sin tocar los JSON de two_stage ni de EASE."""

    path = two_tower_hyperparameters_path(config.artifacts_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    from .tracking import with_run_name

    path.write_text(
        json.dumps(
            with_run_name(
                {
                    "embedding_dim": parameters.embedding_dim,
                    "learning_rate": parameters.learning_rate,
                    "n_negatives": parameters.n_negatives,
                    "temperature": parameters.temperature,
                    "epochs": parameters.epochs,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Hiperparámetros de two-tower guardados en %s", path)


def _log_selected(parameters: TwoTowerParameters, n_items: int, origin: str) -> None:
    """Deja en la corrida activa qué two-tower se entrenó."""

    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.log_params(
        {
            "two_tower_selected.embedding_dim": parameters.embedding_dim,
            "two_tower_selected.learning_rate": parameters.learning_rate,
            "two_tower_selected.n_negatives": parameters.n_negatives,
            "two_tower_selected.temperature": parameters.temperature,
            "two_tower_selected.epochs": parameters.epochs,
            "two_tower_selected.origin": origin,
            "two_tower_selected.n_items": n_items,
        }
    )
    mlflow.log_metric("two_tower_items", float(n_items))


class TwoTowerModel:
    """Recomendador de una etapa. No usa el retrieval ni el LambdaRank."""

    name = "two_tower"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.hyperparameters_origin = "default"
        self.hyperparameters_run_name = ""
        self.hyperparameters: dict[str, Any] = {"model": self.name}
        self._history: pd.DataFrame | None = None
        self._tower: _Tower | None = None
        self._interactions: _Interactions | None = None
        self._popularity: PopularityRecommender | None = None

    def fit(self, split: SplitData, *, tune: bool = False) -> None:
        """Entrena con `split.history`. El último fold solo se usa si `tune` es verdadero."""

        raw = split.history.copy()
        raw[USER_COL] = raw[USER_COL].astype(str)
        raw[ITEM_COL] = raw[ITEM_COL].astype(str)
        if tune:
            parameters = tune_two_tower(raw, split.books, self.config)
            self.hyperparameters_origin = "tune"
            self.hyperparameters_run_name = parameters.run_name
            _write_two_tower_hyperparameters(self.config, parameters)
        else:
            parameters, origin = resolve_two_tower_parameters(self.config)
            self.hyperparameters_origin = origin
            self.hyperparameters_run_name = parameters.run_name if origin == "saved" else ""
        history = _in_catalog(raw, split.books)
        LOGGER.info(
            "Two-tower entrena dim=%s lr=%s negativos=%s temperatura=%s épocas=%s",
            parameters.embedding_dim,
            parameters.learning_rate,
            parameters.n_negatives,
            parameters.temperature,
            parameters.epochs,
        )
        rng = np.random.default_rng(self.config.random_seed)
        tower, interactions, _, _ = _fit_tower(
            history,
            parameters,
            self.config.two_tower,
            rng,
        )
        books = split.books.copy()
        books[ITEM_COL] = books[ITEM_COL].astype(str)
        catalog = set(observed_catalog(history, books)[ITEM_COL].astype(str))
        self._history = history
        self._tower = tower
        self._interactions = interactions
        self._popularity = PopularityRecommender().fit(history, catalog)
        self.hyperparameters = {
            "model": self.name,
            "embedding_dim": parameters.embedding_dim,
            "learning_rate": parameters.learning_rate,
            "n_negatives": parameters.n_negatives,
            "temperature": parameters.temperature,
            "epochs": parameters.epochs,
            "n_items": len(tower.item_ids),
        }
        _log_selected(parameters, len(tower.item_ids), self.hyperparameters_origin)

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        """Top-20 por coseno, sin libros ya leídos. Popularidad completa si falta."""

        if (
            self._history is None
            or self._tower is None
            or self._interactions is None
            or self._popularity is None
        ):
            raise RuntimeError("El modelo debe entrenarse antes de recomendar")
        return _recommend_from_tower(
            self._tower,
            self._interactions,
            self._history,
            [str(user) for user in users],
            self._popularity,
            n=TOP_K,
        )
