"""Catálogo, pérdida y entrenamiento compartidos por rating_mlp y two_tower.

El usuario es un embedding de id_lector. Cada lectura real es una fila: el
modelo predice su rating y el error es cuadrático contra (rating - 1) / 9.
No hay libros negativos. En la evaluación se entrena con la historia previa
al hold-out. De ese prefijo, las últimas ``validation_k`` lecturas de cada
lector con más que esas quedan afuera del gradiente y eligen la época: al
terminar se restauran los pesos de la validación más baja. En el submit se
leen esos pesos y se suman algunas épocas en las que el hold-out ya entra.
Si el archivo no está, el submit rehace las épocas previas y desde ahí suma
el hold-out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baselines import PopularityRecommender
from .config import PipelineConfig
from .data import ITEM_COL, RATING_COL, USER_COL
from .split_data import SplitData
from .splits import leave_last_k

LOGGER = logging.getLogger(__name__)
TOP_K = 20
_RATING_SPAN = 9.0


@dataclass(frozen=True)
class TrainSchedule:
    """Hiperparámetros de optimización que comparten los dos modelos."""

    learning_rate: float
    batch_size: int
    max_epochs: int
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
    author_labels: tuple[str, ...]
    genre_labels: tuple[str, ...]


def place_on_device(net: nn.Module, device: torch.device) -> None:
    """Pasa la red al dispositivo y deja los índices enteros en CPU.

    Copiar un índice int64 desde MPS dispara un fallo de OpenMP en macOS.
    """

    net.to(device)
    for name, buffer in list(net.named_buffers()):
        if buffer.dtype != torch.long or buffer.device.type == "cpu":
            continue
        parent, _, attr = name.rpartition(".")
        module = net.get_submodule(parent) if parent else net
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            module.register_buffer(attr, buffer.detach().to("cpu"))
        finally:
            torch.set_num_threads(threads)


def lookup_rows(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Filas de una tabla de índices, aunque esa tabla esté en CPU."""

    if table.device == index.device:
        return table[index]
    gathered = table.index_select(0, index.detach().to(device=table.device))
    return gathered.to(device=index.device)


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
        author_labels=_labels(author_index),
        genre_labels=_labels(genre_index),
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


def preholdout(history: pd.DataFrame, holdout_k: int) -> pd.DataFrame | None:
    """Lecturas anteriores a las últimas ``holdout_k`` de cada lector.

    Devuelve None si nadie tiene esas últimas lecturas para apartar.
    """

    outer = leave_last_k(history, k=holdout_k)
    if outer.test.empty:
        return None
    return outer.train


def last_reads_for_validation(
    frame: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Últimas ``k`` lecturas de quien tiene más de ``k``. El resto queda en train.

    Quien no llega a ese largo aporta todas sus filas al entrenamiento. Si
    nadie llega, la validación vuelve vacía y no elige época.
    """

    if k < 1:
        raise ValueError("k debe ser positivo")
    empty = frame.iloc[0:0].copy()
    if frame.empty:
        return frame, empty
    outer = leave_last_k(frame, k=k)
    if outer.test.empty:
        return frame, empty
    return outer.train, outer.test


def fit_neural(
    net: nn.Module,
    catalog: InteractionCatalog,
    history: pd.DataFrame,
    config: PipelineConfig,
    schedule: TrainSchedule,
    *,
    model_name: str,
    incorporate_holdout: bool,
) -> None:
    """Entrena ``max_epochs`` sin el hold-out y se queda con la mejor época.

    Dentro de esa historia, las últimas ``validation_k`` lecturas de cada
    lector que tiene más no entran al gradiente. Cada época mide el RMSE de
    ese error y, al cerrar las épocas, la red vuelve a la de menor RMSE. En
    el submit, si la evaluación guardó los pesos, solo corre las
    épocas que suman el hold-out, sin ese corte. Si ese archivo no está,
    rehace las ``max_epochs`` sin el hold-out y desde ahí suma las otras. La
    evaluación no hace la segunda pasada: el NDCG del hold-out se mide
    afuera, y al terminar guarda la red ya restaurada.
    """

    device = pick_device()
    place_on_device(net, device)
    LOGGER.info("%s entrena en %s", model_name, device)
    optimizer = torch.optim.Adam(net.parameters(), lr=schedule.learning_rate)
    rng = np.random.default_rng(config.random_seed)
    holdout_k = config.split.holdout_k
    if incorporate_holdout and _resume(net, catalog, config, model_name):
        if schedule.finetune_epochs < 1:
            LOGGER.info("%s queda en los pesos de la evaluación", model_name)
            return
        LOGGER.info(
            "%s continúa %s épocas con el hold-out incluido",
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
        return
    earlier = preholdout(history, holdout_k) if incorporate_holdout else None
    first = history if earlier is None else earlier
    train_history, val_history = last_reads_for_validation(
        first,
        config.split.validation_k,
    )
    val_rows = training_rows(val_history, catalog) if not val_history.empty else None
    if earlier is None:
        LOGGER.info(
            "%s entrena %s épocas con la historia previa al hold-out",
            model_name,
            schedule.max_epochs,
        )
    else:
        LOGGER.info(
            "%s no encontró pesos de la evaluación; entrena %s épocas sin las últimas %s",
            model_name,
            schedule.max_epochs,
            holdout_k,
        )
    if val_rows is None:
        LOGGER.info(
            "%s no aparta validación: ningún lector supera %s lecturas",
            model_name,
            config.split.validation_k,
        )
    else:
        LOGGER.info(
            "%s aparta las últimas %s lecturas de cada lector (%s filas) para elegir la época",
            model_name,
            config.split.validation_k,
            len(val_history),
        )
    restored = _run_epochs(
        net,
        training_rows(train_history, catalog),
        optimizer,
        schedule,
        rng,
        device,
        epochs=schedule.max_epochs,
        model_name=model_name,
        phase="época",
        val_rows=val_rows,
    )
    if not incorporate_holdout:
        _save_checkpoint(net, catalog, config, model_name)
    if earlier is None or schedule.finetune_epochs < 1:
        return
    if restored:
        optimizer = torch.optim.Adam(net.parameters(), lr=schedule.learning_rate)
    LOGGER.info(
        "%s incorpora el hold-out durante %s épocas",
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


def checkpoint_path(artifacts_dir: Path, model_name: str) -> Path:
    """Archivo de pesos que la evaluación deja para el submit."""

    return Path(artifacts_dir) / f"{model_name}.pt"


_ROW_SOURCES = {
    "user.weight": "users",
    "user_embedding.weight": "users",
    "book.weight": "items",
    "author.weight": "authors",
    "genre.weight": "genres",
}


def _save_checkpoint(
    net: nn.Module,
    catalog: InteractionCatalog,
    config: PipelineConfig,
    model_name: str,
) -> None:
    """Guarda pesos e índices para que el submit no rehaga las épocas."""

    path = checkpoint_path(config.artifacts_dir, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "model": model_name,
        "user_ids": [str(user) for user in catalog.user_ids],
        "item_ids": [str(item) for item in catalog.item_ids],
        "authors": list(catalog.author_labels),
        "genres": list(catalog.genre_labels),
        "state": _cpu_state(net),
    }
    torch.save(payload, path)
    LOGGER.info("%s guardó los pesos en %s", model_name, path)


def _cpu_state(net: nn.Module) -> dict[str, torch.Tensor]:
    """Copia los pesos a CPU de a uno. El copiado paralelo de int64 se cae en MPS."""

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        return {
            key: value.detach().to("cpu").contiguous().clone()
            for key, value in net.state_dict().items()
        }
    finally:
        torch.set_num_threads(threads)


def _resume(
    net: nn.Module,
    catalog: InteractionCatalog,
    config: PipelineConfig,
    model_name: str,
) -> bool:
    """Copia los pesos guardados. Las filas nuevas se quedan en la inicialización."""

    path = checkpoint_path(config.artifacts_dir, model_name)
    if not path.is_file():
        LOGGER.info("%s no encontró %s", model_name, path)
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("el archivo no es un objeto")
        state = payload["state"]
        if payload.get("model") != model_name or payload.get("version") != 1:
            raise ValueError("el archivo es de otro modelo")
        if not isinstance(state, dict) or not _checkpoint_fits(state, net):
            raise ValueError("la forma de la red no coincide")
        users = [str(user) for user in payload["user_ids"]]
        items = [str(item) for item in payload["item_ids"]]
        authors = [str(author) for author in payload["authors"]]
        genres = [str(genre) for genre in payload["genres"]]
    except (OSError, KeyError, ValueError, RuntimeError, TypeError) as exc:
        LOGGER.info("%s ignora %s (%s) y entrena de cero", model_name, path, exc)
        return False
    _transplant(
        net,
        state,
        users=users,
        items=items,
        authors=authors,
        genres=genres,
        catalog=catalog,
    )
    LOGGER.info("%s retoma los pesos de %s", model_name, path)
    return True


def _checkpoint_fits(state: dict, net: nn.Module) -> bool:
    """Las capas densas coinciden. Los embeddings coinciden en la dimensión."""

    current = net.state_dict()
    for key, value in current.items():
        if key in {"item_author", "item_genre"}:
            continue
        previous = state.get(key)
        if not isinstance(previous, torch.Tensor):
            return False
        if key in _ROW_SOURCES:
            if previous.ndim != 2 or value.ndim != 2 or previous.shape[1] != value.shape[1]:
                return False
            continue
        if previous.shape != value.shape:
            return False
    return True


def _transplant(
    net: nn.Module,
    state: dict,
    *,
    users: list[str],
    items: list[str],
    authors: list[str],
    genres: list[str],
    catalog: InteractionCatalog,
) -> None:
    """Copia cada fila al índice nuevo del mismo id. El resto no se toca."""

    saved = {
        "users": users,
        "items": items,
        "authors": authors,
        "genres": genres,
    }
    current_labels = {
        "users": [str(user) for user in catalog.user_ids],
        "items": [str(item) for item in catalog.item_ids],
        "authors": list(catalog.author_labels),
        "genres": list(catalog.genre_labels),
    }
    own = net.state_dict()
    for key, value in own.items():
        if key in {"item_author", "item_genre"}:
            continue
        previous = state[key]
        kind = _ROW_SOURCES.get(key)
        if kind is None:
            value.copy_(previous)
            continue
        _copy_rows(value, previous, current_labels[kind], saved[kind])
    net.load_state_dict(own)


def _copy_rows(
    destination: torch.Tensor,
    source: torch.Tensor,
    destination_labels: list[str],
    source_labels: list[str],
) -> None:
    """Escribe en cada fila nueva el vector de la fila vieja con el mismo nombre."""

    source_index = {label: index for index, label in enumerate(source_labels)}
    for index, label in enumerate(destination_labels):
        previous = source_index.get(label)
        if previous is not None:
            destination[index].copy_(source[previous])


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
        fit_neural(
            net,
            catalog,
            history,
            self.config,
            self.schedule(),
            model_name=self.name,
            incorporate_holdout=not split.holdout,
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


def _labels(vocabulary: dict[str, int]) -> tuple[str, ...]:
    """Nombres en el orden de las filas del embedding."""

    ordered = [""] * len(vocabulary)
    for name, index in vocabulary.items():
        ordered[index] = name
    return tuple(ordered)


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
    val_rows: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Varias épocas sobre las lecturas indicadas.

    Con validación, cada época mide el RMSE del mismo error cuadrático y al
    final la red vuelve a la época de menor RMSE. Devuelve si restauró esos
    pesos. El optimizador de esa pasada queda en la última época; quien sigue
    entrenando arma uno nuevo.
    """

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        loss = _epoch(net, rows, optimizer, schedule.batch_size, rng, device)
        if val_rows is None:
            LOGGER.info("%s %s %s: rmse=%.4f", model_name, phase, epoch, loss)
            continue
        val_loss = _validation_loss(net, val_rows, schedule.batch_size, device)
        LOGGER.info(
            "%s %s %s: rmse=%.4f val_rmse=%.4f",
            model_name,
            phase,
            epoch,
            loss,
            val_loss,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = _cpu_state(net)
    if best_state is None:
        return False
    net.load_state_dict(best_state)
    place_on_device(net, device)
    LOGGER.info(
        "%s se queda con la %s %s (val_rmse=%.4f)",
        model_name,
        phase,
        best_epoch,
        best_loss,
    )
    return True


def _epoch(
    net: nn.Module,
    rows: tuple[np.ndarray, np.ndarray, np.ndarray],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    """Una pasada por las lecturas. El gradiente es el error cuadrático.

    Devuelve el RMSE, la raíz de ese error medio.
    """

    users, items, targets = rows
    if len(users) == 0:
        return 0.0
    net.train()
    order = rng.permutation(len(users))
    total = 0.0
    seen = 0
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        user_batch = torch.as_tensor(users[batch], dtype=torch.long, device=device)
        item_batch = torch.as_tensor(items[batch], dtype=torch.long, device=device)
        target_batch = torch.as_tensor(targets[batch], dtype=torch.float32, device=device)
        optimizer.zero_grad()
        prediction = net(user_batch, item_batch)
        squared = (prediction - target_batch).square()
        squared.mean().backward()
        optimizer.step()
        total += float(squared.detach().sum().cpu())
        seen += int(batch.shape[0])
    return _rmse(total, seen)


def _validation_loss(
    net: nn.Module,
    rows: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> float:
    """RMSE de las filas de validación, sin gradiente."""

    users, items, targets = rows
    if len(users) == 0:
        return 0.0
    net.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            stop = min(start + batch_size, len(users))
            user_batch = torch.tensor(users[start:stop], dtype=torch.long, device=device)
            item_batch = torch.tensor(items[start:stop], dtype=torch.long, device=device)
            target_batch = torch.tensor(targets[start:stop], dtype=torch.float32, device=device)
            prediction = net(user_batch, item_batch)
            total += float((prediction - target_batch).square().sum().detach().cpu())
            seen += stop - start
    return _rmse(total, seen)


def _rmse(sum_squared: float, count: int) -> float:
    """Raíz del error cuadrático medio."""

    if count <= 0:
        return 0.0
    return float(np.sqrt(sum_squared / count))


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
