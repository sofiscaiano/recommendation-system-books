"""Fuentes de candidatos y fusión por Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from ..baselines import PopularityRecommender
from ..config import PipelineConfig
from ..data import DATE_COL, ITEM_COL, RATING_COL, USER_COL
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
    """Contrato mínimo de una fuente de candidatos."""

    source_name: str

    def fit(
        self, interactions: pd.DataFrame, catalog: pd.DataFrame
    ) -> Retriever:
        """Ajusta la fuente con el historial y el catálogo de ese corte."""
        ...

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """Ranking de la fuente, con columnas source, score y rank."""
        ...


def _recency_weights(
    frame: pd.DataFrame,
    half_life_days: float,
) -> np.ndarray:
    """Peso exponencial según la edad respecto de la última lectura del lector."""

    if half_life_days <= 0:
        return np.ones(len(frame), dtype=np.float32)
    max_dates = frame.groupby(USER_COL)[DATE_COL].transform("max")
    age_days = (max_dates - frame[DATE_COL]).dt.days.clip(lower=0).to_numpy()
    return np.exp2(-age_days / half_life_days).astype(np.float32)


def _implicit_confidence(frame: pd.DataFrame, half_life_days: float) -> np.ndarray:
    """Confianza de ALS y BPR: `(1 + rating / 10)` por recencia."""

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
    """Top-n de un lector, sin vistos, con desempate estable por id_libro."""

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
        """Construye la matriz dispersa ítem × usuario, ponderada por rating y recencia."""

        eligible = set(catalog[ITEM_COL].astype(str))
        fit_data = interactions[interactions[ITEM_COL].isin(eligible)].copy()
        self.item_ids_ = np.array(sorted(eligible), dtype=object)
        user_ids = np.array(sorted(fit_data[USER_COL].astype(str).unique()), dtype=object)
        self.item_index_ = {
            item: index for index, item in enumerate(self.item_ids_)
        }
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
        """Ranking por producto interno entre el perfil del lector y cada ítem."""

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
        """Factoriza la matriz usuario × ítem con confianza de rating y recencia."""

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
        """Ranking de ALS o BPR, sin los ítems que el lector ya vio."""

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
        """Recalcula el factor de un lector que no está en la matriz de entrenamiento."""

        if self.algorithm != "als":
            raise RuntimeError("El fold-in en caliente solo está definido para ALS")
        if self.model_ is None or self.item_ids_ is None:
            raise RuntimeError("El retriever debe entrenarse antes de recomendar")
        item_index = {str(item): index for index, item in enumerate(self.item_ids_)}
        if history.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        usable = history[
            (history[USER_COL].astype(str) == user_id)
            & history[ITEM_COL].astype(str).isin(item_index)
        ].copy()
        if usable.empty:
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)

        confidence = _implicit_confidence(usable, self.half_life_days)
        indices = usable[ITEM_COL].astype(str).map(item_index).to_numpy(dtype=int)
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
        """Texto de título, autor, género, editorial y resumen de cada libro."""

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
        """Vectoriza el catálogo. El historial no entra: el perfil se arma al recomendar."""

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
        """Coseno entre el perfil de texto del lector y cada libro no leído."""

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


def _two_tower_candidates(*args: object) -> tuple:
    """Corre en el proceso hijo: ahí sí se importa torch."""

    from ..two_tower import top_candidates

    return top_candidates(*args)


class TwoTowerRetriever:
    """El two-tower como fuente. Entrena y puntúa en un proceso hijo.

    torch y LightGBM cargan copias distintas de libomp y juntas se caen en
    macOS (LightGBM #6595). Este proceso nunca importa torch. El
    entrenamiento ocurre en ``recommend``, con la historia guardada en ``fit``.
    """

    source_name = "two_tower"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._interactions: pd.DataFrame | None = None
        self._catalog: pd.DataFrame | None = None

    def fit(self, interactions: pd.DataFrame, catalog: pd.DataFrame) -> TwoTowerRetriever:
        self._interactions = interactions
        self._catalog = catalog
        return self

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        if self._interactions is None or self._catalog is None:
            raise RuntimeError("El recomendador debe entrenarse antes de recomendar")
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
            known, indices, scores, item_ids = pool.submit(
                _two_tower_candidates,
                self._interactions,
                self._catalog,
                self.config,
                list(users),
                n,
            ).result()
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


class EaseRetriever:
    """EASE como fuente. La matriz se ajusta en este proceso.

    λ, el mínimo de lecturas y el peso de la matriz no los elige el study de
    two_stage: salen del JSON de EASE o de `EaseConfig`. No rellena con
    popularidad. Un lector sin peso en los libros del modelo no aporta filas.
    """

    source_name = "ease"

    def __init__(self, parameters: EaseParameters, *, half_life_days: float) -> None:
        self.parameters = parameters
        self.half_life_days = half_life_days
        self._spectrum: EaseSpectrum | None = None

    def fit(self, interactions: pd.DataFrame, catalog: pd.DataFrame) -> EaseRetriever:
        """Diagonaliza XᵀX con los libros del catálogo que pasan el mínimo."""

        eligible = set(catalog[ITEM_COL].astype(str))
        fit_data = interactions.loc[interactions[ITEM_COL].astype(str).isin(eligible)].copy()
        fit_data[USER_COL] = fit_data[USER_COL].astype(str)
        fit_data[ITEM_COL] = fit_data[ITEM_COL].astype(str)
        matrix, item_ids = interaction_matrix(
            fit_data,
            min_interactions=self.parameters.min_interactions,
            weight_mode=self.parameters.weight_mode,
            half_life_days=self.half_life_days,
        )
        LOGGER.info(
            "EASE como fuente: %s libros (λ=%s, mínimo=%s, peso=%s)",
            len(item_ids),
            self.parameters.reg_lambda,
            self.parameters.min_interactions,
            self.parameters.weight_mode,
        )
        self._spectrum = spectrum_from_matrix(matrix, item_ids)
        return self

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """Top-n de la matriz ya ajustada, sin los libros que el lector ya vio."""

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
    implicit_factors: int = 64,
    implicit_iterations: int = 20,
    half_life_days: float = 365.0,
    tfidf_max_features: int = 60_000,
    enabled: tuple[str, ...] | None = None,
    two_tower: PipelineConfig | None = None,
    ease: EaseParameters | None = None,
) -> list[Retriever]:
    """Fuentes activas. Sin `enabled`, entrena las cinco clásicas y, si se pasan, two-tower y EASE."""

    wanted = None if enabled is None else set(enabled)
    known = {"popularidad", "item_cosine", "als", "bpr", "tfidf", "two_tower", "ease"}
    if wanted is not None:
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(
                "Fuentes de retrieval desconocidas: " + ", ".join(unknown)
            )

    def include(name: str) -> bool:
        return wanted is None or name in wanted

    sources: list[Retriever] = []
    if include("popularidad"):
        sources.append(PopularityRecommender())
    if include("item_cosine"):
        sources.append(ItemCosineRetriever(half_life_days=half_life_days))
    if include("als"):
        sources.append(
            ImplicitMFRetriever(
                "als",
                factors=implicit_factors,
                iterations=implicit_iterations,
                half_life_days=half_life_days,
                random_state=random_state,
            )
        )
    if include("bpr"):
        sources.append(
            ImplicitMFRetriever(
                "bpr",
                factors=implicit_factors,
                iterations=implicit_iterations,
                half_life_days=half_life_days,
                random_state=random_state,
            )
        )
    if include("tfidf"):
        sources.append(
            TfidfRetriever(
                max_features=tfidf_max_features,
                half_life_days=half_life_days,
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
    """Instancia las fuentes activas con los hiperparámetros fijos del pipeline."""

    retrieval = config.retrieval
    return build_default_retrievers(
        random_state=config.random_seed,
        implicit_factors=retrieval.implicit_factors,
        implicit_iterations=retrieval.implicit_iterations,
        half_life_days=retrieval.recency_half_life_days,
        tfidf_max_features=retrieval.tfidf_max_features,
        enabled=retrieval.enabled_sources(),
        two_tower=config if retrieval.two_tower_source else None,
        ease=ease_source_parameters(config) if retrieval.ease_source else None,
    )


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


def per_source_depth(cutoff: int, *, rrf_k: float, n_sources: int) -> int:
    """Profundidad de cada lista para no alterar el top fusionado.

    Con pesos parejos, los `cutoff` primeros de una fuente superan a un libro
    que todas las fuentes recién rankean después de esta profundidad. Más atrás
    el tail no entra al corte del ranking mergeado. El catálogo completo no
    hace falta.
    """

    if cutoff < 1:
        raise ValueError("cutoff debe ser positivo")
    if rrf_k <= 0:
        raise ValueError("rrf_k debe ser positivo")
    if n_sources < 1:
        raise ValueError("n_sources debe ser positivo")
    weight = 1.0 / n_sources
    return int(np.ceil((cutoff + rrf_k) / weight - rrf_k))


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Deja pesos relativos que suman 1.

    El orden de RRF no cambia si se escalan todos los pesos: solo importan
    las proporciones. Un peso nulo apaga esa fuente.
    """

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
    """Candidatos fusionados y rankings fuente conservados."""

    candidates: pd.DataFrame
    source_candidates: pd.DataFrame


class CandidateFusion:
    """RRF ponderado truncado por un corte del ranking fusionado."""

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
        """Suma 1/(k+rank) por fuente y se queda con el prefijo del ranking fusionado."""

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
