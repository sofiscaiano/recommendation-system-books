from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from multiprocessing.queues import Queue
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..baselines import PopularityRecommender
from ..config import PipelineConfig
from ..data import DATE_COL, ITEM_COL, RATING_COL, USER_COL, object_ids
from ..ease import (
    EaseParameters,
    EaseSpectrum,
    ease_ranking,
    ease_source_parameters,
    interaction_matrix,
    spectrum_from_matrix,
)

LOGGER = logging.getLogger(__name__)

CANDIDATE_COLUMNS = [USER_COL, ITEM_COL, "source", "score", "rank"]


class Retriever(Protocol):
    source_name: str

    def fit(self, interactions: pd.DataFrame, catalog: pd.DataFrame) -> Retriever: ...

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame: ...


def _recency_weights(
    frame: pd.DataFrame,
    half_life_days: float,
) -> np.ndarray:
    if half_life_days <= 0:
        return np.ones(len(frame), dtype=np.float32)
    max_dates = frame.groupby(USER_COL)[DATE_COL].transform("max")
    age_days = (max_dates - frame[DATE_COL]).dt.days.clip(lower=0).to_numpy()
    return np.exp2(-age_days / half_life_days).astype(np.float32)


def _implicit_confidence(frame: pd.DataFrame, half_life_days: float) -> np.ndarray:
    return (
        1.0 + frame[RATING_COL].to_numpy(dtype=np.float32) / 10.0
    ) * _recency_weights(frame, half_life_days)


def _rank_scores(
    user: str,
    items: np.ndarray,
    scores: np.ndarray,
    source: str,
    n: int,
    seen: set[str],
) -> list[dict[str, object]]:
    if n <= 0 or len(items) == 0:
        return []
    allowed = np.fromiter((item not in seen for item in items), dtype=bool)
    valid_indices = np.flatnonzero(allowed & np.isfinite(scores))
    if len(valid_indices) == 0:
        return []
    take = min(n, len(valid_indices))
    local_scores = scores[valid_indices]
    if take < len(valid_indices):
        top_local = np.argpartition(local_scores, -take)[-take:]
        selected = valid_indices[top_local]
    else:
        selected = valid_indices
    selected = selected[np.lexsort((items[selected].astype(str), -scores[selected]))][
        :take
    ]
    return [
        {
            USER_COL: user,
            ITEM_COL: str(items[index]),
            "source": source,
            "score": float(scores[index]),
            "rank": rank,
        }
        for rank, index in enumerate(selected, start=1)
    ]


class ItemCosineRetriever:
    def __init__(self, *, half_life_days: float = 365.0) -> None:
        self.source_name = "item_cosine"
        self.half_life_days = half_life_days
        self.item_ids_: np.ndarray | None = None
        self.item_index_: dict[str, int] = {}
        self.item_user_: sparse.csr_matrix | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        catalog: pd.DataFrame,
    ) -> ItemCosineRetriever:
        eligible = set(catalog[ITEM_COL].astype(str))
        fit_data = interactions[interactions[ITEM_COL].isin(eligible)].copy()
        self.item_ids_ = np.array(sorted(eligible), dtype=object)
        user_ids = np.array(
            sorted(fit_data[USER_COL].astype(str).unique()), dtype=object
        )
        self.item_index_ = {item: index for index, item in enumerate(self.item_ids_)}
        user_index = {user: index for index, user in enumerate(user_ids)}

        values = (
            fit_data[RATING_COL].to_numpy(dtype=np.float32)
            / 10.0
            * _recency_weights(fit_data, self.half_life_days)
        )

        rows = fit_data[ITEM_COL].map(self.item_index_).to_numpy()
        cols = fit_data[USER_COL].map(user_index).to_numpy()
        matrix = sparse.csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.item_ids_), len(user_ids)),
            dtype=np.float32,
        )
        matrix.eliminate_zeros()
        self.item_user_ = normalize(matrix, norm="l2", axis=1, copy=False)
        return self

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        if self.item_ids_ is None or self.item_user_ is None:
            raise RuntimeError("El retriever debe entrenarse antes de recomendar")
        rows: list[dict[str, object]] = []
        user_ids = [str(user) for user in users]
        user_position = {user: index for index, user in enumerate(user_ids)}
        history_users = object_ids(history[USER_COL])
        history_items = object_ids(history[ITEM_COL])
        usable = history.loc[
            history_users.isin(user_position) & history_items.isin(self.item_index_)
        ].copy()
        usable[USER_COL] = object_ids(usable[USER_COL])
        usable[ITEM_COL] = object_ids(usable[ITEM_COL])
        if usable.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        query_weights = usable[RATING_COL].to_numpy(dtype=np.float32) / 10.0
        query = sparse.csr_matrix(
            (
                query_weights,
                (
                    usable[USER_COL].map(user_position).to_numpy(),
                    usable[ITEM_COL].map(self.item_index_).to_numpy(),
                ),
            ),
            shape=(len(user_ids), len(self.item_ids_)),
            dtype=np.float32,
        )
        seen_by_user = usable.groupby(USER_COL)[ITEM_COL].agg(set).to_dict()
        active_rows = np.flatnonzero(query.getnnz(axis=1))
        for start in range(0, len(active_rows), 64):
            batch_rows = active_rows[start : start + 64]
            user_space = query[batch_rows] @ self.item_user_
            batch_scores = (user_space @ self.item_user_.T).toarray()
            for row_index, query_row in enumerate(batch_rows):
                user = user_ids[query_row]
                rows.extend(
                    _rank_scores(
                        user,
                        self.item_ids_,
                        batch_scores[row_index],
                        self.source_name,
                        n,
                        set(seen_by_user.get(user, set())),
                    )
                )
        return pd.DataFrame.from_records(rows, columns=CANDIDATE_COLUMNS)


class ImplicitMFRetriever:
    def __init__(
        self,
        algorithm: str = "als",
        *,
        factors: int = 64,
        iterations: int = 20,
        regularization: float = 0.01,
        learning_rate: float = 0.01,
        half_life_days: float = 365.0,
        random_state: int = 42,
    ) -> None:
        if algorithm not in {"als", "bpr"}:
            raise ValueError("algorithm debe ser 'als' o 'bpr'")
        self.algorithm = algorithm
        self.source_name = algorithm
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.half_life_days = half_life_days
        self.random_state = random_state
        self.model_: object | None = None
        self.user_item_: sparse.csr_matrix | None = None
        self.user_index_: dict[str, int] = {}
        self.item_ids_: np.ndarray | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        catalog: pd.DataFrame,
    ) -> ImplicitMFRetriever:
        try:
            from implicit.als import AlternatingLeastSquares
            from implicit.bpr import BayesianPersonalizedRanking
        except ImportError as exc:
            raise ImportError("ImplicitMFRetriever requiere `implicit>=0.7.3`") from exc

        eligible = set(catalog[ITEM_COL].astype(str))
        fit_data = interactions[interactions[ITEM_COL].isin(eligible)].copy()
        user_ids = np.array(
            sorted(fit_data[USER_COL].astype(str).unique()), dtype=object
        )
        self.item_ids_ = np.array(sorted(eligible), dtype=object)
        self.user_index_ = {user: index for index, user in enumerate(user_ids)}
        item_index = {item: index for index, item in enumerate(self.item_ids_)}
        confidence = _implicit_confidence(fit_data, self.half_life_days)
        user_item = sparse.csr_matrix(
            (
                confidence,
                (
                    fit_data[USER_COL].map(self.user_index_).to_numpy(),
                    fit_data[ITEM_COL].map(item_index).to_numpy(),
                ),
            ),
            shape=(len(user_ids), len(self.item_ids_)),
            dtype=np.float32,
        )
        if self.algorithm == "als":
            model = AlternatingLeastSquares(
                factors=self.factors,
                iterations=self.iterations,
                regularization=self.regularization,
                random_state=self.random_state,
            )
        else:
            model = BayesianPersonalizedRanking(
                factors=self.factors,
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                regularization=self.regularization,
                random_state=self.random_state,
                verify_negative_samples=True,
            )
        model.fit(user_item, show_progress=False)
        self.model_ = model
        self.user_item_ = user_item
        return self

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        if self.model_ is None or self.user_item_ is None or self.item_ids_ is None:
            raise RuntimeError("El retriever debe entrenarse antes de recomendar")
        rows: list[dict[str, object]] = []
        for user in users:
            user_index = self.user_index_.get(user)
            if user_index is None:
                continue
            item_indices, scores = self.model_.recommend(
                user_index,
                self.user_item_[user_index],
                N=n,
                filter_already_liked_items=True,
            )
            rows.extend(
                {
                    USER_COL: user,
                    ITEM_COL: str(self.item_ids_[item_index]),
                    "source": self.source_name,
                    "score": float(score),
                    "rank": rank,
                }
                for rank, (item_index, score) in enumerate(
                    zip(item_indices, scores, strict=True), start=1
                )
            )
        return pd.DataFrame.from_records(rows, columns=CANDIDATE_COLUMNS)

    def recommend_fold_in(
        self,
        user_id: str,
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        if self.algorithm != "als":
            raise RuntimeError("El fold-in en caliente solo está definido para ALS")
        if self.model_ is None or self.item_ids_ is None:
            raise RuntimeError("El retriever debe entrenarse antes de recomendar")
        item_index = {str(item): index for index, item in enumerate(self.item_ids_)}
        if history.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        history_users = object_ids(history[USER_COL])
        history_items = object_ids(history[ITEM_COL])
        usable = history.loc[
            (history_users == user_id) & history_items.isin(item_index)
        ].copy()
        usable[ITEM_COL] = object_ids(usable[ITEM_COL])
        if usable.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)

        confidence = _implicit_confidence(usable, self.half_life_days)
        indices = usable[ITEM_COL].map(item_index).to_numpy(dtype=int)
        row = sparse.csr_matrix(
            (
                confidence.astype(np.float32),
                (np.zeros(len(indices), dtype=int), indices),
            ),
            shape=(1, len(self.item_ids_)),
            dtype=np.float32,
        )
        item_indices, scores = self.model_.recommend(
            0,
            row,
            N=n,
            filter_already_liked_items=True,
            recalculate_user=True,
        )
        return pd.DataFrame(
            {
                USER_COL: user_id,
                ITEM_COL: [str(self.item_ids_[index]) for index in item_indices],
                "source": self.source_name,
                "score": [float(score) for score in scores],
                "rank": list(range(1, len(item_indices) + 1)),
            }
        )

    def recommend_with_history(
        self,
        user_id: str,
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        if self.algorithm != "bpr":
            raise RuntimeError(
                "El ajuste con historial nuevo solo está definido para BPR"
            )
        if self.model_ is None or self.item_ids_ is None or self.user_item_ is None:
            raise RuntimeError("El retriever debe entrenarse antes de recomendar")
        user_id = str(user_id)
        requested = self._confidence_by_item(user_id, history)
        stored_index = self.user_index_.get(user_id)
        stored = (
            {}
            if stored_index is None
            else _row_confidence(self.user_item_[stored_index])
        )
        if stored_index is not None and _same_confidence(requested, stored):
            return self.recommend([user_id], history, n)
        if not requested and stored_index is None:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)

        item_factors = np.asarray(self.model_.item_factors, dtype=np.float32)
        if stored_index is None:
            user_factor = np.zeros(item_factors.shape[1], dtype=np.float32)
        else:
            user_factor = np.array(
                self.model_.user_factors[stored_index], dtype=np.float32, copy=True
            )
        pull = np.zeros(item_factors.shape[1], dtype=np.float32)
        weight = 0.0
        for item_index in set(requested) | set(stored):
            delta = requested.get(item_index, 0.0) - stored.get(item_index, 0.0)
            if abs(delta) < 1e-5:
                continue
            pull += np.float32(delta) * item_factors[item_index]
            weight += abs(delta)
        if weight < 1e-5:
            if stored_index is None:
                return pd.DataFrame(columns=CANDIDATE_COLUMNS)
            return self.recommend([user_id], history, n)
        if stored_index is None:
            user_factor = pull / np.float32(weight)
        else:
            user_factor = (user_factor + pull) / np.float32(1.0 + weight)
        scores = item_factors @ user_factor
        seen = {str(self.item_ids_[item_index]) for item_index in requested}
        rows = _rank_scores(user_id, self.item_ids_, scores, self.source_name, n, seen)
        return pd.DataFrame.from_records(rows, columns=CANDIDATE_COLUMNS)

    def _confidence_by_item(
        self, user_id: str, history: pd.DataFrame
    ) -> dict[int, float]:
        if self.item_ids_ is None or history.empty:
            return {}
        item_index = {str(item): index for index, item in enumerate(self.item_ids_)}
        history_users = object_ids(history[USER_COL])
        history_items = object_ids(history[ITEM_COL])
        usable = history.loc[
            (history_users == user_id) & history_items.isin(item_index)
        ].copy()
        usable[ITEM_COL] = object_ids(usable[ITEM_COL])
        if usable.empty:
            return {}
        usable = usable.sort_values(DATE_COL, kind="mergesort").drop_duplicates(
            ITEM_COL, keep="last"
        )
        confidence = _implicit_confidence(usable, self.half_life_days)
        indices = usable[ITEM_COL].map(item_index).to_numpy(dtype=int)
        return {
            int(index): float(value)
            for index, value in zip(indices, confidence, strict=True)
        }


def _row_confidence(row: sparse.spmatrix) -> dict[int, float]:
    return {
        int(index): float(value)
        for index, value in zip(row.indices, row.data, strict=True)
    }


def _same_confidence(left: dict[int, float], right: dict[int, float]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(abs(left[item] - right[item]) <= 1e-4 for item in left)


class TfidfRetriever:
    source_name = "tfidf"

    def __init__(
        self,
        *,
        max_features: int = 60_000,
        half_life_days: float = 365.0,
    ) -> None:
        self.max_features = max_features
        self.half_life_days = half_life_days
        self.vectorizer_: TfidfVectorizer | None = None
        self.item_matrix_: sparse.csr_matrix | None = None
        self.item_ids_: np.ndarray | None = None
        self.item_index_: dict[str, int] = {}

    @staticmethod
    def _text(catalog: pd.DataFrame) -> pd.Series:
        fields = ["titulo", "autor", "genero", "editorial", "resumen"]
        available = [field for field in fields if field in catalog]
        text = catalog[available].fillna("").astype(str).agg(" ".join, axis=1)
        fallback = catalog[ITEM_COL].astype(str).str.replace("-", " ", regex=False)
        return (text + " " + fallback).str.strip()

    def fit(
        self,
        interactions: pd.DataFrame,
        catalog: pd.DataFrame,
    ) -> TfidfRetriever:
        del interactions
        fit_catalog = catalog.drop_duplicates(ITEM_COL).sort_values(ITEM_COL)
        self.item_ids_ = fit_catalog[ITEM_COL].astype(str).to_numpy()
        self.item_index_ = {item: index for index, item in enumerate(self.item_ids_)}
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        text = self._text(fit_catalog)
        try:
            self.item_matrix_ = vectorizer.fit_transform(text).tocsr()
        except ValueError:
            vectorizer.set_params(min_df=1, max_df=1.0)
            self.item_matrix_ = vectorizer.fit_transform(text).tocsr()
        self.vectorizer_ = vectorizer
        return self

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        if self.item_matrix_ is None or self.item_ids_ is None:
            raise RuntimeError("El retriever debe entrenarse antes de recomendar")
        rows: list[dict[str, object]] = []
        user_ids = [str(user) for user in users]
        user_position = {user: index for index, user in enumerate(user_ids)}
        history_users = object_ids(history[USER_COL])
        history_items = object_ids(history[ITEM_COL])
        usable = history.loc[
            history_users.isin(user_position) & history_items.isin(self.item_index_)
        ].copy()
        usable[USER_COL] = object_ids(usable[USER_COL])
        usable[ITEM_COL] = object_ids(usable[ITEM_COL])
        if usable.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        weights = (
            usable[RATING_COL].to_numpy(dtype=np.float32)
            / 10.0
            * _recency_weights(usable, self.half_life_days)
        )
        history_matrix = sparse.csr_matrix(
            (
                weights,
                (
                    usable[USER_COL].map(user_position).to_numpy(),
                    usable[ITEM_COL].map(self.item_index_).to_numpy(),
                ),
            ),
            shape=(len(user_ids), len(self.item_ids_)),
            dtype=np.float32,
        )
        profiles = normalize(
            history_matrix @ self.item_matrix_,
            norm="l2",
            axis=1,
        )
        seen_by_user = usable.groupby(USER_COL)[ITEM_COL].agg(set).to_dict()
        active_rows = np.flatnonzero(history_matrix.getnnz(axis=1))
        for start in range(0, len(active_rows), 64):
            batch_rows = active_rows[start : start + 64]
            batch_scores = (profiles[batch_rows] @ self.item_matrix_.T).toarray()
            for row_index, query_row in enumerate(batch_rows):
                user = user_ids[query_row]
                rows.extend(
                    _rank_scores(
                        user,
                        self.item_ids_,
                        batch_scores[row_index],
                        self.source_name,
                        n,
                        set(seen_by_user.get(user, set())),
                    )
                )
        return pd.DataFrame.from_records(rows, columns=CANDIDATE_COLUMNS)


def _train_two_tower(
    history: pd.DataFrame, books: pd.DataFrame, config: PipelineConfig
):
    from ..two_tower import train_weights

    return train_weights(history, books, config)


def _serve_two_tower(weights: object, inbox: Queue, outbox: Queue) -> None:
    try:
        from ..two_tower import load_tower

        tower = load_tower(weights)
    except Exception as exc:  # noqa: BLE001
        outbox.put(("err", RuntimeError(f"{type(exc).__name__}: {exc}")))
        return
    outbox.put(("ready", None))
    while True:
        message = inbox.get()
        if message is None:
            break
        users, history, n = message
        try:
            outbox.put(("ok", tower.score(users, n, history)))
        except Exception as exc:  # noqa: BLE001
            outbox.put(("err", RuntimeError(f"{type(exc).__name__}: {exc}")))


class TwoTowerRetriever:
    source_name = "two_tower"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._weights: object | None = None
        self._process: Any = None
        self._inbox: Queue | None = None
        self._outbox: Queue | None = None
        self._lock = threading.Lock()

    def fit(
        self, interactions: pd.DataFrame, catalog: pd.DataFrame
    ) -> TwoTowerRetriever:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        self._stop_worker()
        with ProcessPoolExecutor(
            max_workers=1, mp_context=get_context("spawn")
        ) as pool:
            self._weights = pool.submit(
                _train_two_tower, interactions, catalog, self.config
            ).result()
        LOGGER.info(
            "two_tower entrenado: %s libros en el vocabulario",
            len(self._weights.item_ids),
        )
        return self

    def warm(self) -> None:
        if self._weights is None:
            return
        with self._lock:
            self._ensure_worker()

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        if self._weights is None:
            raise RuntimeError("El recomendador debe entrenarse antes de recomendar")
        known, indices, scores, item_ids = self._ask(
            [str(user) for user in users],
            history,
            n,
        )
        if not known:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        valid = indices >= 0
        return pd.DataFrame(
            {
                USER_COL: np.repeat(np.array(known, dtype=object), valid.sum(axis=1)),
                ITEM_COL: item_ids[indices[valid]],
                "source": self.source_name,
                "score": scores[valid].astype(float),
                "rank": np.nonzero(valid)[1] + 1,
            }
        )

    def _ask(self, users: list[str], history: pd.DataFrame, n: int) -> tuple:
        with self._lock:
            self._ensure_worker()
            assert self._inbox is not None
            assert self._outbox is not None
            self._inbox.put((users, history, n))
            try:
                status, payload = self._outbox.get(timeout=180)
            except Exception as exc:
                self._stop_worker()
                raise RuntimeError("El proceso del two-tower no respondió") from exc
        if status != "ok":
            raise payload
        return payload

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        from multiprocessing import get_context

        context = get_context("spawn")
        self._inbox = context.Queue()
        self._outbox = context.Queue()
        self._process = context.Process(
            target=_serve_two_tower,
            args=(self._weights, self._inbox, self._outbox),
        )
        self._process.start()
        try:
            status, payload = self._outbox.get(timeout=180)
        except Exception as exc:
            self._stop_worker()
            raise RuntimeError(
                "El proceso del two-tower no llegó a cargar la red"
            ) from exc
        if status != "ready":
            self._stop_worker()
            raise payload

    def _stop_worker(self) -> None:
        process = self._process
        inbox = self._inbox
        self._process = None
        self._inbox = None
        self._outbox = None
        if process is None:
            return
        if process.is_alive() and inbox is not None:
            try:
                inbox.put(None)
                process.join(timeout=2)
            except (OSError, ValueError, RuntimeError):
                process.terminate()
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_process"] = None
        state["_inbox"] = None
        state["_outbox"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._process = None
        self._inbox = None
        self._outbox = None
        self._lock = threading.Lock()

    def __del__(self) -> None:
        try:
            self._stop_worker()
        except (OSError, ValueError, RuntimeError):
            return


class EaseRetriever:
    source_name = "ease"

    def __init__(self, parameters: EaseParameters, *, half_life_days: float) -> None:
        self.parameters = parameters
        self.half_life_days = half_life_days
        self._spectrum: EaseSpectrum | None = None

    def fit(self, interactions: pd.DataFrame, catalog: pd.DataFrame) -> EaseRetriever:
        eligible = set(catalog[ITEM_COL].astype(str))
        fit_data = interactions.loc[
            interactions[ITEM_COL].astype(str).isin(eligible)
        ].copy()
        fit_data[USER_COL] = fit_data[USER_COL].astype(str)
        fit_data[ITEM_COL] = fit_data[ITEM_COL].astype(str)
        matrix, item_ids = interaction_matrix(
            fit_data,
            min_interactions=self.parameters.min_interactions,
        )
        LOGGER.info(
            "EASE como fuente: %s libros (λ=%s, mínimo=%s)",
            len(item_ids),
            self.parameters.reg_lambda,
            self.parameters.min_interactions,
        )
        self._spectrum = spectrum_from_matrix(matrix, item_ids)
        return self

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        if self._spectrum is None:
            raise RuntimeError("El retriever debe entrenarse antes de recomendar")
        ranked = ease_ranking(
            history,
            list(users),
            self._spectrum,
            self.parameters,
            half_life_days=self.half_life_days,
            n=n,
        )
        if ranked.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        ranked = ranked.copy()
        ranked["source"] = self.source_name
        return ranked[CANDIDATE_COLUMNS]


def build_default_retrievers(
    *,
    random_state: int = 42,
    half_life_days: float = 365.0,
    item_cosine: Any | None = None,
    als: Any | None = None,
    bpr: Any | None = None,
    tfidf: Any | None = None,
    enabled: tuple[str, ...] | None = None,
    two_tower: PipelineConfig | None = None,
    ease: EaseParameters | None = None,
) -> list[Retriever]:
    wanted = None if enabled is None else set(enabled)
    known = {"popularidad", "item_cosine", "als", "bpr", "tfidf", "two_tower", "ease"}
    if wanted is not None:
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError("Fuentes de retrieval desconocidas: " + ", ".join(unknown))

    def include(name: str) -> bool:
        return wanted is None or name in wanted

    cosine_half_life = (
        half_life_days if item_cosine is None else item_cosine.half_life_days
    )
    als_factors = 64 if als is None else als.factors
    als_iterations = 20 if als is None else als.iterations
    als_regularization = 0.01 if als is None else als.regularization
    als_half_life = half_life_days if als is None else als.half_life_days
    bpr_factors = 64 if bpr is None else bpr.factors
    bpr_iterations = 20 if bpr is None else bpr.iterations
    bpr_learning_rate = 0.01 if bpr is None else bpr.learning_rate
    bpr_regularization = 0.01 if bpr is None else bpr.regularization
    bpr_half_life = half_life_days if bpr is None else bpr.half_life_days
    tfidf_features = 60_000 if tfidf is None else tfidf.max_features
    tfidf_half_life = half_life_days if tfidf is None else tfidf.half_life_days

    sources: list[Retriever] = []
    if include("popularidad"):
        sources.append(PopularityRecommender())
    if include("item_cosine"):
        sources.append(ItemCosineRetriever(half_life_days=cosine_half_life))
    if include("als"):
        sources.append(
            ImplicitMFRetriever(
                "als",
                factors=als_factors,
                iterations=als_iterations,
                regularization=als_regularization,
                half_life_days=als_half_life,
                random_state=random_state,
            )
        )
    if include("bpr"):
        sources.append(
            ImplicitMFRetriever(
                "bpr",
                factors=bpr_factors,
                iterations=bpr_iterations,
                learning_rate=bpr_learning_rate,
                regularization=bpr_regularization,
                half_life_days=bpr_half_life,
                random_state=random_state,
            )
        )
    if include("tfidf"):
        sources.append(
            TfidfRetriever(
                max_features=tfidf_features,
                half_life_days=tfidf_half_life,
            )
        )
    if two_tower is not None and include("two_tower"):
        sources.append(TwoTowerRetriever(two_tower))
    if ease is not None and include("ease"):
        sources.append(EaseRetriever(ease, half_life_days=half_life_days))
    if not sources:
        raise ValueError("al menos una fuente de retrieval tiene que estar activa")
    return sources


def retrievers_from_config(config: PipelineConfig) -> list[Retriever]:
    from ..source_models import (
        als_source_parameters,
        bpr_source_parameters,
        item_cosine_source_parameters,
        resolved_two_tower_config,
        tfidf_source_parameters,
    )

    retrieval = config.retrieval
    two_tower = None
    if retrieval.two_tower_source:
        two_tower, _origin, _run_name = resolved_two_tower_config(config)
    return build_default_retrievers(
        random_state=config.random_seed,
        half_life_days=retrieval.recency_half_life_days,
        item_cosine=(
            item_cosine_source_parameters(config)
            if retrieval.item_cosine_source
            else None
        ),
        als=als_source_parameters(config) if retrieval.als_source else None,
        bpr=bpr_source_parameters(config) if retrieval.bpr_source else None,
        tfidf=tfidf_source_parameters(config) if retrieval.tfidf_source else None,
        enabled=retrieval.enabled_sources(),
        two_tower=two_tower,
        ease=ease_source_parameters(config) if retrieval.ease_source else None,
    )


def fit_retrievers(
    retrievers: list[Retriever],
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> list[Retriever]:
    for retriever in retrievers:
        if isinstance(retriever, PopularityRecommender):
            retriever.fit(interactions, set(catalog[ITEM_COL]))
        else:
            retriever.fit(interactions, catalog)
    return retrievers


def retrieve_all(
    retrievers: list[Retriever],
    users: list[str] | tuple[str, ...],
    history: pd.DataFrame,
    per_source: int,
) -> pd.DataFrame:
    frames = [
        retriever.recommend(users, history, per_source) for retriever in retrievers
    ]
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    result = pd.concat(non_empty, ignore_index=True)
    return result.sort_values(
        [USER_COL, "source", "rank", ITEM_COL], kind="mergesort"
    ).reset_index(drop=True)


def per_source_depth(cutoff: int, *, rrf_k: float, n_sources: int) -> int:
    if cutoff < 1:
        raise ValueError("cutoff debe ser positivo")
    if rrf_k <= 0:
        raise ValueError("rrf_k debe ser positivo")
    if n_sources < 1:
        raise ValueError("n_sources debe ser positivo")
    weight = 1.0 / n_sources
    return int(np.ceil((cutoff + rrf_k) / weight - rrf_k))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        raise ValueError("Se requiere al menos un peso")
    cleaned: dict[str, float] = {}
    for source, weight in weights.items():
        value = float(weight)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"El peso de {source} debe ser finito y no negativo")
        cleaned[str(source)] = value
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError("Al menos un peso debe ser positivo")
    return {source: value / total for source, value in cleaned.items()}


@dataclass(frozen=True)
class FusionResult:
    candidates: pd.DataFrame
    source_candidates: pd.DataFrame


class CandidateFusion:
    def __init__(
        self,
        *,
        weights: dict[str, float],
        rrf_k: float = 60.0,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k debe ser positivo")
        self.weights = _normalize_weights(weights)
        self.rrf_k = rrf_k

    def fuse(self, candidates: pd.DataFrame, cutoff: int) -> FusionResult:
        if cutoff < 1:
            raise ValueError("cutoff debe ser positivo")
        empty = pd.DataFrame(
            columns=[USER_COL, ITEM_COL, "rrf_score", "retrieval_rank"]
        )
        if candidates.empty:
            return FusionResult(empty, candidates.copy())

        required = set(CANDIDATE_COLUMNS)
        missing = required - set(candidates.columns)
        if missing:
            raise ValueError(f"Faltan columnas de candidatos: {sorted(missing)}")

        long = (
            candidates.sort_values(
                [USER_COL, "source", "rank", ITEM_COL], kind="mergesort"
            )
            .drop_duplicates([USER_COL, ITEM_COL, "source"], keep="first")
            .copy()
        )
        long["weight"] = long["source"].map(self.weights).fillna(0.0)
        long = long.loc[long["weight"] > 0].copy()
        if long.empty:
            return FusionResult(empty, long.drop(columns="weight"))

        long["rrf_component"] = long["weight"] / (
            self.rrf_k + long["rank"].astype(float)
        )
        aggregate = (
            long.groupby([USER_COL, ITEM_COL], as_index=False)["rrf_component"]
            .sum()
            .rename(columns={"rrf_component": "rrf_score"})
            .sort_values(
                [USER_COL, "rrf_score", ITEM_COL],
                ascending=[True, False, True],
                kind="mergesort",
            )
        )
        aggregate["retrieval_rank"] = (
            aggregate.groupby(USER_COL, sort=False).cumcount() + 1
        )
        fused = aggregate.loc[aggregate["retrieval_rank"] <= cutoff]
        return FusionResult(
            candidates=fused[
                [USER_COL, ITEM_COL, "rrf_score", "retrieval_rank"]
            ].reset_index(drop=True),
            source_candidates=long.drop(columns=["weight", "rrf_component"]),
        )
