"""Catálogo, pérdida y entrenamiento compartidos por rating_mlp y two_tower.

El usuario es un embedding de id_lector. Cada lectura real es una fila: el
modelo predice su rating y el error es cuadrático contra (rating - 1) / 9.
No hay libros negativos. El último fold interno elige los pesos; después se
sigue entrenando, sin reiniciar, sobre toda la historia previa al hold-out.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baselines import PopularityRecommender
from .config import PipelineConfig
from .data import ITEM_COL, RATING_COL, USER_COL
from .metrics import ndcg_at_k
from .split_data import SplitData, validation_folds

LOGGER = logging.getLogger(__name__)
TOP_K = 20
_RATING_SPAN = 9.0
_IMPROVEMENT = 1e-4


@dataclass(frozen=True)
class TrainSchedule:
    """Hiperparámetros de optimización que comparten los dos modelos."""

    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    finetune_epochs: int


@dataclass(frozen=True)
class InteractionCatalog:
    """Índices de lector, libro, autor y género de una historia."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    user_index: dict[str, int]
    item_index: dict[str, int]
    item_author: np.ndarray
    item_genre: np.ndarray
    n_authors: int
    n_genres: int


def pick_device() -> torch.device:
    """MPS si está disponible, y CPU en caso contrario."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_catalog(frame: pd.DataFrame, books: pd.DataFrame) -> InteractionCatalog:
    """Arma los índices. El autor o el género vacío ocupa el índice 0."""

    history = _in_catalog(frame, books)
    if history.empty:
        raise ValueError("No hay lecturas dentro del catálogo para entrenar")
    metadata = _metadata(books)
    user_ids = np.array(sorted(history[USER_COL].unique()), dtype=object)
    item_ids = np.array(sorted(history[ITEM_COL].unique()), dtype=object)
    user_index = {user: index for index, user in enumerate(user_ids)}
    item_index = {item: index for index, item in enumerate(item_ids)}
    by_item = metadata.set_index(ITEM_COL)
    authors = [
        str(by_item.at[item, "autor"]) if item in by_item.index else ""
        for item in item_ids
    ]
    genres = [
        str(by_item.at[item, "genero"]) if item in by_item.index else ""
        for item in item_ids
    ]
    author_index = _vocabulary(authors)
    genre_index = _vocabulary(genres)
    return InteractionCatalog(
        user_ids=user_ids,
        item_ids=item_ids,
        user_index=user_index,
        item_index=item_index,
        item_author=np.array([author_index[value] for value in authors], dtype=np.int64),
        item_genre=np.array([genre_index[value] for value in genres], dtype=np.int64),
        n_authors=len(author_index),
        n_genres=len(genre_index),
    )


def training_rows(
    frame: pd.DataFrame,
    catalog: InteractionCatalog,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lecturas del catálogo: índice de lector, índice de libro y rating escalado."""

    history = frame.copy()
    history[USER_COL] = history[USER_COL].astype(str)
    history[ITEM_COL] = history[ITEM_COL].astype(str)
    kept = history[
        history[USER_COL].isin(catalog.user_index)
        & history[ITEM_COL].isin(catalog.item_index)
    ]
    users = kept[USER_COL].map(catalog.user_index).to_numpy(dtype=np.int64)
    items = kept[ITEM_COL].map(catalog.item_index).to_numpy(dtype=np.int64)
    ratings = kept[RATING_COL].to_numpy(dtype=np.float32)
    targets = np.clip((ratings - 1.0) / _RATING_SPAN, 0.0, 1.0)
    return users, items, targets


def init_embeddings(module: nn.Module) -> None:
    """Embeddings chicos y capas lineales con Xavier. Evita un primer paso enorme."""

    for layer in module.modules():
        if isinstance(layer, nn.Embedding):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def fit_neural(
    net: nn.Module,
    catalog: InteractionCatalog,
    history: pd.DataFrame,
    config: PipelineConfig,
    schedule: TrainSchedule,
    *,
    model_name: str,
    recommend: Callable[[pd.DataFrame, list[str]], pd.DataFrame],
) -> None:
    """Entrena el último fold y sigue desde esos pesos sobre toda la historia.

    Si la historia no alcanza para un fold, entrena directo sobre ella.
    """

    device = pick_device()
    net.to(device)
    LOGGER.info("%s entrena en %s", model_name, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=schedule.learning_rate)
    rng = np.random.default_rng(config.random_seed)
    folds = _folds(history, config)
    if not folds:
        LOGGER.info("%s no tiene fold interno; entrena sobre toda la historia", model_name)
        _run_epochs(
            net,
            training_rows(history, catalog),
            optimizer,
            schedule,
            rng,
            device,
            epochs=schedule.max_epochs,
            model_name=model_name,
            phase="época",
        )
        return

    fold = folds[-1]
    best_score = -1.0
    best_state = _clone_state(net)
    stalled = 0
    rows = training_rows(fold.train, catalog)
    for epoch in range(1, schedule.max_epochs + 1):
        loss = _epoch(net, rows, optimizer, schedule.batch_size, rng, device)
        recommendations = recommend(fold.train, list(fold.query_users))
        score = ndcg_at_k(recommendations, fold.target, k=TOP_K)
        LOGGER.info(
            "%s época %s: loss=%.4f ndcg=%.4f",
            model_name,
            epoch,
            loss,
            score,
        )
        if score > best_score + _IMPROVEMENT:
            best_score = score
            best_state = _clone_state(net)
            stalled = 0
        else:
            stalled += 1
            if stalled >= schedule.patience:
                break
    net.load_state_dict(best_state)
    if schedule.finetune_epochs < 1:
        return
    optimizer = torch.optim.Adam(net.parameters(), lr=schedule.learning_rate)
    LOGGER.info(
        "%s sigue desde el fold interno durante %s épocas",
        model_name,
        schedule.finetune_epochs,
    )
    _run_epochs(
        net,
        training_rows(history, catalog),
        optimizer,
        schedule,
        rng,
        device,
        epochs=schedule.finetune_epochs,
        model_name=model_name,
        phase="continuación",
    )


def rank_users(
    catalog: InteractionCatalog,
    history: pd.DataFrame,
    users: list[str],
    scores: np.ndarray | None,
    known: np.ndarray,
) -> pd.DataFrame:
    """Top-20 por score, sin libros ya leídos. Popularidad completa si falta."""

    if not users:
        return pd.DataFrame(columns=[USER_COL, ITEM_COL, "score", "rank"])
    seen = history.groupby(USER_COL, sort=False)[ITEM_COL].agg(set).to_dict()
    popularity = PopularityRecommender().fit(history, set(catalog.item_ids.tolist()))
    fallback = popularity.recommend(users, history, n=TOP_K)
    fallback_by_user = (
        {}
        if fallback.empty
        else fallback.groupby(USER_COL)[ITEM_COL].agg(list).to_dict()
    )
    rows: list[dict[str, object]] = []
    known_cursor = 0
    for user, is_known in zip(users, known, strict=True):
        user_seen = set(seen.get(user, set()))
        if is_known and scores is not None:
            user_scores = scores[known_cursor].copy()
            known_cursor += 1
            _mask_seen(user_scores, catalog.item_index, user_seen)
            chosen = _top_n(catalog.item_ids, user_scores, TOP_K)
        else:
            chosen = []
        already = {item for item, _ in chosen}
        for item in fallback_by_user.get(user, []):
            if len(chosen) >= TOP_K:
                break
            if item in already or item in user_seen:
                continue
            chosen.append((item, float("nan")))
            already.add(item)
        for rank, (item, score) in enumerate(chosen, start=1):
            rows.append({USER_COL: user, ITEM_COL: item, "score": score, "rank": rank})
    return pd.DataFrame.from_records(rows, columns=[USER_COL, ITEM_COL, "score", "rank"])


class NeuralRanker:
    """fit y recommend comunes. Cada modelo arma su red y sus scores."""

    name = ""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.hyperparameters_origin = "config"
        self.hyperparameters_run_name = ""
        self.hyperparameters: dict[str, object] = {"model": self.name}
        self._net: nn.Module | None = None
        self._catalog: InteractionCatalog | None = None
        self._history: pd.DataFrame | None = None

    def fit(self, split: SplitData, *, tune: bool = False) -> None:
        """Entrena con ``split.history``. ``tune`` no abre Optuna."""

        if tune:
            LOGGER.info(
                "%s no usa Optuna; entrena con los hiperparámetros del config",
                self.name,
            )
        torch.manual_seed(self.config.random_seed)
        history = _in_catalog(split.history, split.books)
        catalog = build_catalog(history, split.books)
        net = self.build_net(catalog)
        init_embeddings(net)
        self._net = net
        self._catalog = catalog

        def recommend(frame: pd.DataFrame, users: list[str]) -> pd.DataFrame:
            return self._recommend(frame, users)

        fit_neural(
            net,
            catalog,
            history,
            self.config,
            self.schedule(),
            model_name=self.name,
            recommend=recommend,
        )
        self._history = history
        self.hyperparameters = self.snapshot(catalog)
        log_network(self.name, self.hyperparameters)

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        """Top-20. Quien no está en la matriz sigue la popularidad."""

        if self._history is None or self._net is None or self._catalog is None:
            raise RuntimeError("El modelo debe entrenarse antes de recomendar")
        return self._recommend(self._history, [str(user) for user in users])

    def build_net(self, catalog: InteractionCatalog) -> nn.Module:
        """Red de este modelo, todavía sin mover de dispositivo."""

        raise NotImplementedError

    def schedule(self) -> TrainSchedule:
        """Hiperparámetros de Adam y de las épocas."""

        raise NotImplementedError

    def score_users(self, user_indices: np.ndarray) -> np.ndarray:
        """Una fila de scores por lector, una columna por libro del catálogo."""

        raise NotImplementedError

    def snapshot(self, catalog: InteractionCatalog) -> dict[str, object]:
        """Hiperparámetros que quedan en la corrida y en la entrega."""

        raise NotImplementedError

    def _recommend(self, history: pd.DataFrame, users: list[str]) -> pd.DataFrame:
        if self._net is None or self._catalog is None:
            raise RuntimeError("El modelo debe entrenarse antes de recomendar")
        known_mask = np.array(
            [user in self._catalog.user_index for user in users],
            dtype=bool,
        )
        scores = None
        if np.any(known_mask):
            indices = np.array(
                [
                    self._catalog.user_index[user]
                    for user, known in zip(users, known_mask, strict=True)
                    if known
                ],
                dtype=np.int64,
            )
            self._net.eval()
            with torch.no_grad():
                scores = self.score_users(indices)
        return rank_users(self._catalog, history, users, scores, known_mask)


def log_network(model_name: str, values: dict[str, object]) -> None:
    """Deja en la corrida activa los hiperparámetros con los que se entrenó."""

    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.set_tag("model", model_name)
    mlflow.log_params({f"{model_name}.{key}": value for key, value in values.items()})


def _in_catalog(frame: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    """Deja solo libros del catálogo. El corte temporal no los saca."""

    catalog = set(books[ITEM_COL].astype(str))
    kept = frame[frame[ITEM_COL].astype(str).isin(catalog)].copy()
    kept[USER_COL] = kept[USER_COL].astype(str)
    kept[ITEM_COL] = kept[ITEM_COL].astype(str)
    return kept


def _metadata(books: pd.DataFrame) -> pd.DataFrame:
    """Autor y género por libro. Si la columna no existe, queda vacía."""

    frame = books.copy()
    frame[ITEM_COL] = frame[ITEM_COL].astype(str)
    for column in ("autor", "genero"):
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    return frame[[ITEM_COL, "autor", "genero"]].drop_duplicates(ITEM_COL)


def _vocabulary(values: list[str]) -> dict[str, int]:
    """Índice 0 reservado al valor vacío. El resto sigue el orden alfabético."""

    names = sorted({value for value in values if value})
    return {"": 0, **{name: index + 1 for index, name in enumerate(names)}}


def _folds(history: pd.DataFrame, config: PipelineConfig) -> list:
    """Último fold usable, o ninguno si la historia es demasiado corta."""

    try:
        return validation_folds(history, config)
    except ValueError as exc:
        if "suficientes" not in str(exc):
            raise
        return []


def _run_epochs(
    net: nn.Module,
    rows: tuple[np.ndarray, np.ndarray, np.ndarray],
    optimizer: torch.optim.Optimizer,
    schedule: TrainSchedule,
    rng: np.random.Generator,
    device: torch.device,
    *,
    epochs: int,
    model_name: str,
    phase: str,
) -> None:
    """Varias épocas sin mirar el NDCG. Sirve para la continuación y el atajo."""

    for epoch in range(1, epochs + 1):
        loss = _epoch(net, rows, optimizer, schedule.batch_size, rng, device)
        LOGGER.info("%s %s %s: loss=%.4f", model_name, phase, epoch, loss)


def _epoch(
    net: nn.Module,
    rows: tuple[np.ndarray, np.ndarray, np.ndarray],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    """Una pasada por las lecturas. Devuelve el error cuadrático medio."""

    users, items, targets = rows
    if len(users) == 0:
        return 0.0
    net.train()
    order = rng.permutation(len(users))
    losses: list[float] = []
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        user_batch = torch.as_tensor(users[batch], dtype=torch.long, device=device)
        item_batch = torch.as_tensor(items[batch], dtype=torch.long, device=device)
        target_batch = torch.as_tensor(targets[batch], dtype=torch.float32, device=device)
        optimizer.zero_grad()
        prediction = net(user_batch, item_batch)
        loss = nn.functional.mse_loss(prediction, target_batch)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _clone_state(net: nn.Module) -> dict[str, torch.Tensor]:
    """Copia de los pesos en CPU, para volver a la mejor época."""

    return {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}


def _mask_seen(scores: np.ndarray, item_index: dict[str, int], seen: set[str]) -> None:
    """Tapa los libros ya leídos para que no vuelvan al top."""

    for item in seen:
        index = item_index.get(item)
        if index is not None:
            scores[index] = -np.inf


def _top_n(item_ids: np.ndarray, scores: np.ndarray, n: int) -> list[tuple[str, float]]:
    """Top-n con desempate por id_libro."""

    available = int(np.isfinite(scores).sum())
    take = min(n, available)
    if take <= 0:
        return []
    chosen = np.argpartition(scores, -take)[-take:]
    labels = item_ids[chosen].astype(str)
    chosen = chosen[np.lexsort((labels, -scores[chosen]))]
    return [
        (str(item_ids[index]), float(scores[index]))
        for index in chosen
        if np.isfinite(scores[index])
    ]
