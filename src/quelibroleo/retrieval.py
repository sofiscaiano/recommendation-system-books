"""Fuentes de candidatos y fusión por Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .baselines import PopularityRecommender
from .data import DATE_COL, ITEM_COL, RATING_COL, USER_COL

CANDIDATE_COLUMNS = [USER_COL, ITEM_COL, "source", "score", "rank"]


class Retriever(Protocol):
    """Contrato mínimo de una fuente de candidatos."""

    source_name: str

    def fit(
        self, interactions: pd.DataFrame, catalog: pd.DataFrame
    ) -> Retriever: ...

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
    selected = selected[
        np.lexsort((items[selected].astype(str), -scores[selected]))
    ][:take]
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
    """Item-item sin materializar la matriz cuadrada de similitud."""

    def __init__(
        self,
        mode: str = "occurrence",
        *,
        half_life_days: float = 365.0,
    ) -> None:
        if mode not in {"occurrence", "adjusted"}:
            raise ValueError("mode debe ser 'occurrence' o 'adjusted'")
        self.mode = mode
        self.source_name = (
            "item_cosine" if mode == "occurrence" else "item_adjusted_cosine"
        )
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
        user_ids = np.array(sorted(fit_data[USER_COL].astype(str).unique()), dtype=object)
        self.item_index_ = {
            item: index for index, item in enumerate(self.item_ids_)
        }
        user_index = {user: index for index, user in enumerate(user_ids)}

        if self.mode == "occurrence":
            values = (
                fit_data[RATING_COL].to_numpy(dtype=np.float32)
                / 10.0
                * _recency_weights(fit_data, self.half_life_days)
            )
        else:
            user_means = fit_data.groupby(USER_COL)[RATING_COL].transform("mean")
            values = (
                fit_data[RATING_COL].to_numpy(dtype=np.float32)
                - user_means.to_numpy(dtype=np.float32)
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
        usable = history[
            history[USER_COL].isin(user_position)
            & history[ITEM_COL].isin(self.item_index_)
        ].copy()
        if usable.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        if self.mode == "occurrence":
            query_weights = usable[RATING_COL].to_numpy(dtype=np.float32) / 10.0
        else:
            means = usable.groupby(USER_COL)[RATING_COL].transform("mean")
            query_weights = (
                usable[RATING_COL].to_numpy(dtype=np.float32)
                - means.to_numpy(dtype=np.float32)
            )
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


class SurpriseSVDRetriever:
    """Factorización explícita de ratings mediante Surprise SVD."""

    source_name = "svd"

    def __init__(
        self,
        *,
        factors: int = 64,
        epochs: int = 20,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.epochs = epochs
        self.random_state = random_state
        self.model_: object | None = None
        self.trainset_: object | None = None
        self.item_ids_: np.ndarray | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        catalog: pd.DataFrame,
    ) -> SurpriseSVDRetriever:
        try:
            from surprise import SVD, Dataset, Reader
        except ImportError as exc:
            raise ImportError(
                "SurpriseSVDRetriever requiere `scikit-surprise>=1.1.5`"
            ) from exc

        eligible = set(catalog[ITEM_COL].astype(str))
        fit_data = interactions[interactions[ITEM_COL].isin(eligible)]
        reader = Reader(rating_scale=(1, 10))
        dataset = Dataset.load_from_df(
            fit_data[[USER_COL, ITEM_COL, RATING_COL]],
            reader,
        )
        trainset = dataset.build_full_trainset()
        model = SVD(
            n_factors=self.factors,
            n_epochs=self.epochs,
            random_state=self.random_state,
        )
        model.fit(trainset)
        self.model_ = model
        self.trainset_ = trainset
        self.item_ids_ = np.array(sorted(eligible), dtype=object)
        return self

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        if self.model_ is None or self.trainset_ is None or self.item_ids_ is None:
            raise RuntimeError("El retriever debe entrenarse antes de recomendar")
        model = self.model_
        trainset = self.trainset_
        known_raw_items = [
            item
            for item in self.item_ids_
            if item in trainset._raw2inner_id_items
        ]
        inner_items = np.array(
            [trainset.to_inner_iid(item) for item in known_raw_items], dtype=int
        )
        raw_items = np.array(known_raw_items, dtype=object)
        seen_by_user = history.groupby(USER_COL)[ITEM_COL].agg(set).to_dict()
        rows: list[dict[str, object]] = []
        known_users: list[tuple[str, int]] = []
        for user in users:
            try:
                inner_user = trainset.to_inner_uid(user)
            except ValueError:
                continue
            known_users.append((str(user), inner_user))

        item_factors = model.qi[inner_items]
        item_bias = model.bi[inner_items]
        for start in range(0, len(known_users), 128):
            batch = known_users[start : start + 128]
            batch_indices = np.array([inner for _, inner in batch], dtype=int)
            batch_scores = (
                model.pu[batch_indices] @ item_factors.T
                + model.bu[batch_indices, None]
                + item_bias[None, :]
                + trainset.global_mean
            )
            for row_index, (user, _) in enumerate(batch):
                rows.extend(
                    _rank_scores(
                        user,
                        raw_items,
                        batch_scores[row_index],
                        self.source_name,
                        n,
                        set(seen_by_user.get(user, set())),
                    )
                )
        return pd.DataFrame.from_records(rows, columns=CANDIDATE_COLUMNS)


class ImplicitMFRetriever:
    """ALS o BPR sobre ocurrencia ponderada por rating y recencia."""

    def __init__(
        self,
        algorithm: str = "als",
        *,
        factors: int = 64,
        iterations: int = 20,
        half_life_days: float = 365.0,
        random_state: int = 42,
    ) -> None:
        if algorithm not in {"als", "bpr"}:
            raise ValueError("algorithm debe ser 'als' o 'bpr'")
        self.algorithm = algorithm
        self.source_name = algorithm
        self.factors = factors
        self.iterations = iterations
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
            raise ImportError(
                "ImplicitMFRetriever requiere `implicit>=0.7.3`"
            ) from exc

        eligible = set(catalog[ITEM_COL].astype(str))
        fit_data = interactions[interactions[ITEM_COL].isin(eligible)].copy()
        user_ids = np.array(sorted(fit_data[USER_COL].astype(str).unique()), dtype=object)
        self.item_ids_ = np.array(sorted(eligible), dtype=object)
        self.user_index_ = {user: index for index, user in enumerate(user_ids)}
        item_index = {item: index for index, item in enumerate(self.item_ids_)}
        confidence = (
            1.0
            + fit_data[RATING_COL].to_numpy(dtype=np.float32) / 10.0
        ) * _recency_weights(fit_data, self.half_life_days)
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
                random_state=self.random_state,
            )
        else:
            model = BayesianPersonalizedRanking(
                factors=self.factors,
                iterations=self.iterations,
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


class TfidfRetriever:
    """Perfil de contenido del lector sobre metadata textual de libros."""

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
        self.item_index_ = {
            item: index for index, item in enumerate(self.item_ids_)
        }
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
        usable = history[
            history[USER_COL].isin(user_position)
            & history[ITEM_COL].isin(self.item_index_)
        ].copy()
        if usable.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        weights = (
            usable[RATING_COL].to_numpy(dtype=np.float32) / 10.0
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


def build_default_retrievers(
    *,
    random_state: int = 42,
    svd_factors: int = 64,
    svd_epochs: int = 20,
    implicit_factors: int = 64,
    implicit_iterations: int = 20,
    half_life_days: float = 365.0,
    tfidf_max_features: int = 60_000,
) -> list[Retriever]:
    """Crea las fuentes acordadas con parámetros homogéneos."""

    return [
        PopularityRecommender(),
        ItemCosineRetriever("occurrence", half_life_days=half_life_days),
        ItemCosineRetriever("adjusted", half_life_days=half_life_days),
        SurpriseSVDRetriever(
            factors=svd_factors,
            epochs=svd_epochs,
            random_state=random_state,
        ),
        ImplicitMFRetriever(
            "als",
            factors=implicit_factors,
            iterations=implicit_iterations,
            half_life_days=half_life_days,
            random_state=random_state,
        ),
        ImplicitMFRetriever(
            "bpr",
            factors=implicit_factors,
            iterations=implicit_iterations,
            half_life_days=half_life_days,
            random_state=random_state,
        ),
        TfidfRetriever(
            max_features=tfidf_max_features,
            half_life_days=half_life_days,
        ),
    ]


def fit_retrievers(
    retrievers: list[Retriever],
    interactions: pd.DataFrame,
    catalog: pd.DataFrame,
) -> list[Retriever]:
    """Entrena todas las fuentes."""

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
    """Concatena rankings largos de todas las fuentes."""

    frames = [
        retriever.recommend(users, history, per_source)
        for retriever in retrievers
    ]
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    result = pd.concat(non_empty, ignore_index=True)
    return result.sort_values(
        [USER_COL, "source", "rank", ITEM_COL], kind="mergesort"
    ).reset_index(drop=True)


@dataclass(frozen=True)
class FusionResult:
    """Candidatos fusionados y rankings fuente conservados."""

    candidates: pd.DataFrame
    source_candidates: pd.DataFrame


class CandidateFusion:
    """Cuotas mínimas por fuente y relleno con RRF ponderado."""

    def __init__(
        self,
        *,
        weights: dict[str, float],
        quotas: dict[str, int],
        rrf_k: float = 60.0,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k debe ser positivo")
        self.weights = weights
        self.quotas = quotas
        self.rrf_k = rrf_k

    def _scaled_quotas(self, budget: int) -> dict[str, int]:
        active = {
            source: max(0, int(quota))
            for source, quota in self.quotas.items()
            if quota > 0
        }
        total = sum(active.values())
        if total <= budget:
            return active
        scaled = {
            source: int(np.floor(quota * budget / total))
            for source, quota in active.items()
        }
        remainder = budget - sum(scaled.values())
        priority = sorted(
            active,
            key=lambda source: (
                -(active[source] * budget / total - scaled[source]),
                source,
            ),
        )
        for source in priority[:remainder]:
            scaled[source] += 1
        return scaled

    def fuse(self, candidates: pd.DataFrame, budget: int) -> FusionResult:
        if budget < 1:
            raise ValueError("budget debe ser positivo")
        if candidates.empty:
            empty = pd.DataFrame(
                columns=[USER_COL, ITEM_COL, "rrf_score", "retrieval_rank"]
            )
            return FusionResult(empty, candidates.copy())

        required = set(CANDIDATE_COLUMNS)
        missing = required - set(candidates.columns)
        if missing:
            raise ValueError(f"Faltan columnas de candidatos: {sorted(missing)}")

        long = candidates.drop_duplicates(
            [USER_COL, ITEM_COL, "source"], keep="first"
        ).copy()
        source_weights = long["source"].map(self.weights).fillna(1.0)
        long["rrf_component"] = source_weights / (
            self.rrf_k + long["rank"].astype(float)
        )
        aggregate = (
            long.groupby([USER_COL, ITEM_COL], as_index=False)["rrf_component"]
            .sum()
            .rename(columns={"rrf_component": "rrf_score"})
        )
        quotas = self._scaled_quotas(budget)
        output: list[pd.DataFrame] = []

        for user, user_long in long.groupby(USER_COL, sort=False):
            selected: set[str] = set()
            for source in sorted(quotas):
                quota = quotas[source]
                source_items = user_long[user_long["source"] == source].nsmallest(
                    quota, "rank"
                )
                for item in source_items[ITEM_COL]:
                    if len(selected) == budget:
                        break
                    selected.add(item)

            user_aggregate = aggregate[aggregate[USER_COL] == user].sort_values(
                ["rrf_score", ITEM_COL],
                ascending=[False, True],
                kind="mergesort",
            )
            for item in user_aggregate[ITEM_COL]:
                if len(selected) == budget:
                    break
                selected.add(item)

            fused = user_aggregate[user_aggregate[ITEM_COL].isin(selected)].copy()
            fused = fused.sort_values(
                ["rrf_score", ITEM_COL],
                ascending=[False, True],
                kind="mergesort",
            ).head(budget)
            fused["retrieval_rank"] = np.arange(1, len(fused) + 1)
            output.append(fused)

        fused_candidates = pd.concat(output, ignore_index=True)
        return FusionResult(
            candidates=fused_candidates[
                [USER_COL, ITEM_COL, "rrf_score", "retrieval_rank"]
            ],
            source_candidates=long.drop(columns="rrf_component"),
        )
