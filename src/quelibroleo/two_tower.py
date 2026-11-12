"""Two-tower: un vector por lector, otro por libro, y el ranking es su producto.

- Torre de lector: promedio de los embeddings de los libros que leyó, sin
  vector propio por lector (Covington et al., RecSys 2016).
- Torre de libro: embeddings de id, autor y género. La tabla de id es la
  misma que promedia la torre de lector.
- Pérdida: softmax contra los otros libros del batch, con corrección logQ
  por popularidad y vectores normalizados con temperatura (Yi et al.,
  RecSys 2019).
- Ejemplos: la historia anterior a un corte predice cualquiera de las 20
  lecturas siguientes, como el hold-out (ventana futura de PinnerFormer,
  KDD 2022). Cada uno pesa su rating, la ganancia de ``ndcg_score``.

La cantidad de épocas sale del NDCG@20 del último fold rolling. Con ella se
reentrena de cero sobre toda la historia recibida.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .baselines import PopularityRecommender
from .config import PipelineConfig, TwoTowerConfig
from .data import DATE_COL, ITEM_COL, RATING_COL, USER_COL
from .metrics import ndcg_at_k
from .source_models import (
    TwoTowerParameters,
    apply_two_tower_parameters,
    resolved_two_tower_config,
    two_tower_resolved_values,
    write_two_tower_hyperparameters,
)
from .split_data import SplitData, validation_folds

LOGGER = logging.getLogger(__name__)
TOP_K = 20

# Con el OpenMP de LightGBM ya activo en el proceso, torch multihilo se cuelga en macOS.
torch.set_num_threads(1)


@dataclass(frozen=True)
class Vocabulary:
    """Libros con embedding propio. La posición en ``item_ids`` es la fila."""

    item_ids: np.ndarray
    item_index: dict[str, int]
    item_author: np.ndarray
    item_genre: np.ndarray
    n_authors: int
    n_genres: int
    log_q: np.ndarray


@dataclass(frozen=True)
class Sequences:
    """Lecturas de cada lector en orden temporal, concatenadas en ``items``."""

    position: dict[str, int]
    starts: np.ndarray
    lengths: np.ndarray
    items: np.ndarray
    ratings: np.ndarray


def _codes(values: pd.Series) -> tuple[np.ndarray, int]:
    """Un índice por valor distinto. El valor vacío ocupa el 0."""

    clean = values.fillna("").astype(str).str.strip()
    codes, uniques = pd.factorize(clean.where(clean != ""))
    return (codes + 1).astype(np.int64), len(uniques) + 1


def build_vocabulary(history: pd.DataFrame, books: pd.DataFrame, min_reads: int) -> Vocabulary:
    """Libros con al menos ``min_reads`` lecturas. Solo ellos son candidatos."""

    counts = history[ITEM_COL].value_counts()
    item_ids = np.array(sorted(counts.index[counts >= min_reads]), dtype=object)
    metadata = (
        books.astype({ITEM_COL: str})
        .drop_duplicates(ITEM_COL)
        .set_index(ITEM_COL)
        .reindex(item_ids)
    )
    item_author, n_authors = _codes(metadata["autor"])
    item_genre, n_genres = _codes(metadata["genero"])
    frequency = counts.reindex(item_ids).to_numpy(dtype=np.float64)
    return Vocabulary(
        item_ids=item_ids,
        item_index={item: index for index, item in enumerate(item_ids)},
        item_author=item_author,
        item_genre=item_genre,
        n_authors=n_authors,
        n_genres=n_genres,
        log_q=np.log(frequency / frequency.sum()),
    )


def build_sequences(history: pd.DataFrame, vocabulary: Vocabulary) -> Sequences:
    """Historia de cada lector, por fecha e id_libro, restringida al vocabulario."""

    kept = history[history[ITEM_COL].isin(vocabulary.item_index)].sort_values(
        [USER_COL, DATE_COL, ITEM_COL], kind="mergesort"
    )
    users, starts, lengths = np.unique(
        kept[USER_COL].to_numpy(dtype=str), return_index=True, return_counts=True
    )
    return Sequences(
        position={user: index for index, user in enumerate(users)},
        starts=starts,
        lengths=lengths,
        items=kept[ITEM_COL].map(vocabulary.item_index).to_numpy(dtype=np.int64),
        ratings=kept[RATING_COL].to_numpy(dtype=np.float32),
    )


def sample_examples(
    sequences: Sequences, window: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Una fila por lectura con historia previa: objetivo, inicio y largo de esa historia.

    El corte se sortea en cada época para que el objetivo quede entre las
    ``window`` lecturas siguientes. La historia es todo lo anterior al corte.
    """

    starts = np.repeat(sequences.starts, sequences.lengths)
    offset = np.arange(len(sequences.items)) - starts
    has_past = offset >= 1
    offset = offset[has_past]
    cut = rng.integers(np.maximum(1, offset - window + 1), offset + 1)
    return np.flatnonzero(has_past), starts[has_past], cut


def _bags(items: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Índices y offsets para ``embedding_bag``: un tramo de ``items`` por fila."""

    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    flat = np.repeat(starts - offsets, lengths) + np.arange(lengths.sum())
    return torch.as_tensor(items[flat]), torch.as_tensor(offsets)


def history_bags(
    items: np.ndarray, starts: np.ndarray, lengths: np.ndarray, recent: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """La historia completa y, si ``recent`` es positivo, sus últimas ``recent`` lecturas."""

    bags = [_bags(items, starts, lengths)]
    if recent > 0:
        short = np.minimum(lengths, recent)
        bags.append(_bags(items, starts + lengths - short, short))
    return bags


def _mlp(inputs: int, dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim))


class TwoTowerNet(nn.Module):
    """Las dos torres. Cada una termina en un vector de norma 1."""

    def __init__(self, vocabulary: Vocabulary, config: TwoTowerConfig) -> None:
        super().__init__()
        dim = config.embedding_dim
        self.item = nn.Embedding(len(vocabulary.item_ids), dim)
        self.author = nn.Embedding(vocabulary.n_authors, dim)
        self.genre = nn.Embedding(vocabulary.n_genres, dim)
        self.user_authors = config.user_authors
        self.recent_reads = config.recent_reads
        n_bags = 2 if config.recent_reads > 0 else 1
        per_bag = 2 if config.user_authors else 1
        self.user_tower = _mlp(n_bags * per_bag * dim, dim, config.dropout)
        self.item_tower = _mlp(3 * dim, dim, config.dropout)
        self.register_buffer("item_author", torch.as_tensor(vocabulary.item_author))
        self.register_buffer("item_genre", torch.as_tensor(vocabulary.item_genre))
        self.register_buffer("log_q", torch.as_tensor(vocabulary.log_q, dtype=torch.float32))

    def users(self, bags: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Promedio de libros (y de sus autores) de cada bolsa, concatenados."""

        pooled = []
        for indices, offsets in bags:
            pooled.append(F.embedding_bag(indices, self.item.weight, offsets, mode="mean"))
            if self.user_authors:
                authors = self.item_author[indices]
                pooled.append(F.embedding_bag(authors, self.author.weight, offsets, mode="mean"))
        return F.normalize(self.user_tower(torch.cat(pooled, dim=-1)), dim=-1)

    def items(self, items: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                self.item(items),
                self.author(self.item_author[items]),
                self.genre(self.item_genre[items]),
            ],
            dim=-1,
        )
        return F.normalize(self.item_tower(features), dim=-1)


def in_batch_loss(
    net: TwoTowerNet,
    bags: list[tuple[torch.Tensor, torch.Tensor]],
    targets: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Softmax sobre los libros del batch, menos log q del negativo, ponderada por rating.

    Si un libro se repite en el batch, sus copias no cuentan como negativo.
    """

    logits = net.users(bags) @ net.items(targets).T / temperature - net.log_q[targets]
    duplicate = targets[None, :] == targets[:, None]
    duplicate.fill_diagonal_(False)
    logits = logits.masked_fill(duplicate, float("-inf"))
    loss = F.cross_entropy(logits, torch.arange(len(targets)), reduction="none")
    return (loss * weights).sum() / weights.sum()


def _train_epoch(
    net: TwoTowerNet,
    optimizer: torch.optim.Optimizer,
    sequences: Sequences,
    config: TwoTowerConfig,
    window: int,
    rng: np.random.Generator,
) -> float:
    net.train()
    targets, starts, lengths = sample_examples(sequences, window, rng)
    order = rng.permutation(len(targets))
    total = 0.0
    for begin in range(0, len(order), config.batch_size):
        batch = order[begin : begin + config.batch_size]
        loss = in_batch_loss(
            net,
            history_bags(sequences.items, starts[batch], lengths[batch], net.recent_reads),
            torch.as_tensor(sequences.items[targets[batch]]),
            torch.as_tensor(sequences.ratings[targets[batch]]),
            config.temperature,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(batch)
    return total / max(len(order), 1)


@torch.no_grad()
def _top_items(
    net: TwoTowerNet, sequences: Sequences, users: list[str], k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray]:
    """Top-k del vocabulario y su score para cada lector, sin lo que ya leyó. -1 si no alcanza."""

    net.eval()
    rows = np.array([sequences.position[user] for user in users], dtype=np.int64)
    lengths = sequences.lengths[rows]
    bags = history_bags(sequences.items, sequences.starts[rows], lengths, net.recent_reads)
    seen = bags[0][0]
    scores = net.users(bags) @ net.items(torch.arange(net.item.num_embeddings)).T
    scores[torch.as_tensor(np.repeat(np.arange(len(rows)), lengths)), seen] = float("-inf")
    top = torch.topk(scores, k=min(k, scores.shape[1]), dim=1)
    values = top.values.numpy()
    return np.where(np.isfinite(values), top.indices.numpy(), -1), values


def recommend_frame(
    net: TwoTowerNet,
    vocabulary: Vocabulary,
    sequences: Sequences,
    popularity: PopularityRecommender,
    history: pd.DataFrame,
    users: list[str],
) -> pd.DataFrame:
    """Top-20 del two-tower. Sin historia en el vocabulario, o si no llega a 20, popularidad."""

    known = [user for user in users if user in sequences.position]
    picks = dict(zip(known, _top_items(net, sequences, known)[0], strict=True)) if known else {}
    fallback = popularity.recommend(users, history, n=TOP_K).groupby(USER_COL)[ITEM_COL].agg(list)
    rows: list[dict[str, object]] = []
    for user in users:
        chosen = [str(vocabulary.item_ids[index]) for index in picks.get(user, []) if index >= 0]
        chosen += [item for item in fallback.get(user, []) if item not in chosen]
        rows.extend(
            {USER_COL: user, ITEM_COL: item, "rank": rank}
            for rank, item in enumerate(chosen[:TOP_K], start=1)
        )
    return pd.DataFrame.from_records(rows, columns=[USER_COL, ITEM_COL, "rank"])


@dataclass
class Trained:
    net: TwoTowerNet
    vocabulary: Vocabulary
    sequences: Sequences
    popularity: PopularityRecommender
    history: pd.DataFrame


def train(
    history: pd.DataFrame,
    books: pd.DataFrame,
    config: PipelineConfig,
    *,
    epochs: int,
    validation: tuple[list[str], pd.DataFrame] | None = None,
) -> tuple[Trained, int, float]:
    """Entrena desde cero. Con validación corta por patience y devuelve la mejor época."""

    block = config.two_tower
    torch.manual_seed(config.random_seed)
    rng = np.random.default_rng(config.random_seed)
    vocabulary = build_vocabulary(history, books, block.min_item_reads)
    sequences = build_sequences(history, vocabulary)
    net = TwoTowerNet(vocabulary, block)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=block.learning_rate, weight_decay=block.weight_decay
    )
    trained = Trained(
        net=net,
        vocabulary=vocabulary,
        sequences=sequences,
        popularity=PopularityRecommender().fit(history, set(history[ITEM_COL])),
        history=history,
    )
    LOGGER.info("two_tower entrena con %s libros en el vocabulario", len(vocabulary.item_ids))
    best_epoch, best_ndcg, stale = epochs, float("-inf"), 0
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(net, optimizer, sequences, block, config.split.holdout_k, rng)
        if validation is None:
            LOGGER.info("two_tower época %s: loss=%.4f", epoch, loss)
            continue
        users, target = validation
        ndcg = ndcg_at_k(recommend_frame(net, vocabulary, sequences, trained.popularity, history, users), target)
        LOGGER.info("two_tower época %s: loss=%.4f val_ndcg@20=%.4f", epoch, loss, ndcg)
        if ndcg > best_ndcg:
            best_epoch, best_ndcg, stale = epoch, ndcg, 0
            continue
        stale += 1
        if stale >= block.patience:
            break
    return trained, best_epoch, best_ndcg


def fit_history(
    history: pd.DataFrame, books: pd.DataFrame, config: PipelineConfig
) -> tuple[Trained, int, float]:
    """Elige las épocas en el último fold de ``history`` y reentrena con toda ella."""

    history = history.astype({USER_COL: str, ITEM_COL: str})
    history = history[history[ITEM_COL].isin(set(books[ITEM_COL].astype(str)))]
    fold = validation_folds(history, config)[-1]
    _, epochs, val_ndcg = train(
        fold.train,
        books,
        config,
        epochs=config.two_tower.max_epochs,
        validation=(list(fold.query_users), fold.target),
    )
    LOGGER.info("two_tower reentrena %s épocas con toda la historia", epochs)
    trained, _, _ = train(history, books, config, epochs=epochs)
    return trained, epochs, val_ndcg


def top_candidates(
    history: pd.DataFrame,
    books: pd.DataFrame,
    config: PipelineConfig,
    users: list[str],
    n: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Lectores con historia, índices y scores de su top-n, e ids del vocabulario.

    Arrays en vez de un DataFrame: viajan baratos desde el proceso hijo.
    """

    trained, _, _ = fit_history(history, books, config)
    known = [str(user) for user in users if str(user) in trained.sequences.position]
    if not known:
        empty = np.empty((0, 0))
        return [], empty.astype(np.int64), empty.astype(np.float32), trained.vocabulary.item_ids
    indices, scores = _top_items(trained.net, trained.sequences, known, k=n)
    return known, indices, scores, trained.vocabulary.item_ids


def tune_two_tower(
    history: pd.DataFrame,
    books: pd.DataFrame,
    config: PipelineConfig,
) -> TwoTowerParameters:
    """Elige dimensión, learning rate, temperatura y el mínimo de lecturas.

    Cada trial corta las épocas con patience sobre el último fold y reentrena
    con toda la historia. El NDCG@20 es el de esa validación, no el del
    reentrenamiento. Con ``tune_on_example_users`` la validación mira solo
    los lectores de ``ejemplo.csv``.
    """

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("El tuning de two_tower requiere `optuna`") from exc

    prepared = _books_only(history, books)
    folds = validation_folds(prepared, config)
    if len(folds) < 2:
        raise ValueError("Se requieren al menos dos folds para el study de two_tower")
    search = config.two_tower.search

    def objective(trial: Any) -> float:
        """NDCG@20 del último fold, con las épocas que elige patience."""

        parameters = TwoTowerParameters(
            embedding_dim=int(search.embedding_dim.suggest(trial, "embedding_dim")),
            learning_rate=float(search.learning_rate.suggest(trial, "learning_rate")),
            temperature=float(search.temperature.suggest(trial, "temperature")),
            min_item_reads=int(search.min_item_reads.suggest(trial, "min_item_reads")),
        )
        _trained, _epochs, score = _fit_once(
            prepared,
            books,
            apply_two_tower_parameters(config, parameters),
        )
        LOGGER.info(
            "two_tower trial %s: dim=%s lr=%.4g temperatura=%.4g mínimo=%s ndcg=%.4f",
            trial.number,
            parameters.embedding_dim,
            parameters.learning_rate,
            parameters.temperature,
            parameters.min_item_reads,
            score,
        )
        return score

    from .tracking import log_optuna_plots, optuna_callback, study_name, track_study

    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(
        study_name=study_name("two_tower"),
        direction="maximize",
        sampler=sampler,
    )
    with track_study("two_tower"):
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.set_tag("model", "two_tower")
        study.optimize(
            objective,
            n_trials=config.optuna_trials,
            callbacks=[optuna_callback("two_tower")],
            show_progress_bar=True,
        )
        log_optuna_plots(study, "two_tower")

    best = study.best_trial
    from .tracking import active_run_name

    return TwoTowerParameters(
        embedding_dim=int(best.params["embedding_dim"]),
        learning_rate=float(best.params["learning_rate"]),
        temperature=float(best.params["temperature"]),
        min_item_reads=int(best.params["min_item_reads"]),
        run_name=active_run_name(),
    )


def _fit_once(
    history: pd.DataFrame,
    books: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[Trained, int, float]:
    """Elige las épocas en el último fold y reentrena. El NDCG es el de la validación."""

    fold = validation_folds(history, config)[-1]
    users, target = _study_population(config, fold.query_users, fold.target)
    _, epochs, val_ndcg = train(
        fold.train,
        books,
        config,
        epochs=config.two_tower.max_epochs,
        validation=(users, target),
    )
    trained, _, _ = train(history, books, config, epochs=epochs)
    return trained, epochs, val_ndcg


def _books_only(history: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    prepared = history.astype({USER_COL: str, ITEM_COL: str})
    return prepared[prepared[ITEM_COL].isin(set(books[ITEM_COL].astype(str)))].copy()


def _study_population(
    config: PipelineConfig,
    query_users: tuple[str, ...],
    target: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    """Lectores del objetivo. Con el flag, solo los de ejemplo.csv."""

    from .split_data import example_users

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


class TwoTowerModel:
    """Recomendador de una etapa. No usa el retrieval ni LambdaRank de two_stage."""

    name = "two_tower"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.hyperparameters_origin = "default"
        self.hyperparameters_run_name = ""
        self.hyperparameters: dict[str, object] = {"model": self.name}
        self._trained: Trained | None = None

    def fit(self, split: SplitData, *, tune: bool = False) -> None:
        """Elige las épocas en el último fold y reentrena con toda ``split.history``.

        Con ``tune``, antes busca dimensión, learning rate, temperatura y el
        mínimo de lecturas. Sin ``tune``, usa el JSON si existe.
        """

        if tune:
            parameters = tune_two_tower(split.history, split.books, self.config)
            self.hyperparameters_origin = "tune"
            self.hyperparameters_run_name = parameters.run_name
            write_two_tower_hyperparameters(self.config, parameters)
            self.config = apply_two_tower_parameters(self.config, parameters)
        else:
            self.config, origin, run_name = resolved_two_tower_config(self.config)
            self.hyperparameters_origin = origin
            self.hyperparameters_run_name = run_name if origin == "saved" else ""
        self._trained, epochs, val_ndcg = fit_history(
            split.history,
            split.books,
            self.config,
        )
        self.hyperparameters = {
            "model": self.name,
            **two_tower_resolved_values(self.config),
            "epochs": epochs,
            "validation_ndcg_at_20": val_ndcg,
            "n_items": len(self._trained.vocabulary.item_ids),
        }
        _log_to_mlflow(self.hyperparameters)

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        """Top-20 sin libros ya leídos."""

        if self._trained is None:
            raise RuntimeError("El modelo debe entrenarse antes de recomendar")
        trained = self._trained
        return recommend_frame(
            trained.net,
            trained.vocabulary,
            trained.sequences,
            trained.popularity,
            trained.history,
            [str(user) for user in users],
        )


def _log_to_mlflow(values: dict[str, object]) -> None:
    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.log_params({f"two_tower.{key}": value for key, value in values.items()})
