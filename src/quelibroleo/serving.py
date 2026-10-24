"""Inferencia de un lector con las fuentes ya ajustadas.

``evaluate`` y ``submit`` no pasan por acá. El artefacto se exporta aparte,
después de un ``submit``, y cada request solo llama a ``recommend`` y
``predict``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .baselines import PopularityRecommender
from .config import PipelineConfig
from .data import (
    DATE_COL,
    ITEM_COL,
    RATING_COL,
    USER_COL,
    load_dataset,
    observed_catalog,
)
from .ranker import LambdaRanker
from .ranker_columns import ColumnBuilder
from .retrieval import (
    CANDIDATE_COLUMNS,
    CandidateFusion,
    ImplicitMFRetriever,
    fit_retrievers,
    per_source_depth,
    retrievers_from_config,
)

LOGGER = logging.getLogger(__name__)

SERVING_FILENAME = "serving.joblib"
TOP_N = 20


class UnknownBooksError(ValueError):
    """Algún libro puntuado no está en el catálogo."""

    def __init__(self, book_ids: list[str]) -> None:
        self.book_ids = book_ids
        super().__init__(f"Libros fuera del catálogo: {', '.join(book_ids)}")


@dataclass
class ServingModel:
    """Ranker, fuentes y builder listos para puntuar sin volver a ajustar."""

    ranker: LambdaRanker
    retrievers: list[Any]
    feature_builder: ColumnBuilder
    weights: dict[str, float]
    rrf_cutoff: int
    rrf_k: float


def export_serving_artifact(
    config: PipelineConfig,
    *,
    pipeline_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Ajusta las fuentes sobre toda la historia y guarda ``serving.joblib``.

    El ranker sale de ``final_pipeline.joblib``. No reentrena LambdaRank ni
    escribe la entrega.
    """

    artifacts = Path(config.artifacts_dir)
    pipeline = Path(pipeline_path or artifacts / "final_pipeline.joblib")
    if not pipeline.exists():
        raise FileNotFoundError(
            f"No existe {pipeline}. Corré submit antes de export-serving."
        )

    trained = joblib.load(pipeline)
    bundle = load_dataset(config.db_path)
    interactions = bundle.interactions
    catalog = observed_catalog(interactions, bundle.books)
    LOGGER.info(
        "Ajustando fuentes de serving sobre %s interacciones",
        len(interactions),
    )
    retrievers = fit_retrievers(retrievers_from_config(config), interactions, catalog)
    builder = ColumnBuilder().fit(interactions, catalog)
    model = ServingModel(
        ranker=trained.ranker,
        retrievers=retrievers,
        feature_builder=builder,
        weights=dict(trained.weights),
        rrf_cutoff=int(trained.rrf_cutoff),
        rrf_k=float(config.retrieval.rrf_k),
    )
    destination = Path(output_path or artifacts / SERVING_FILENAME)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    LOGGER.info("Artefacto de serving guardado en %s", destination)
    return destination


def load_serving_model(path: str | Path) -> ServingModel:
    """Lee el artefacto exportado."""

    model = joblib.load(path)
    if not isinstance(model, ServingModel):
        raise TypeError(f"{path} no contiene un ServingModel")
    return model


@dataclass
class ServingIndex:
    """Modelo en memoria, catálogo para la ficha e historial por lector."""

    model: ServingModel
    books: dict[str, dict[str, str]]
    history: dict[str, pd.DataFrame]
    extra_seen: dict[str, set[str]]
    known_ids: set[str]

    def recommend(
        self,
        user_id: str,
        ratings: list[tuple[str, int]],
    ) -> dict[str, Any]:
        """Top-20 del lector, mezclando el historial guardado con la sesión."""

        missing = unknown_book_ids(
            [item_id for item_id, _rating in ratings], set(self.books)
        )
        if missing:
            raise UnknownBooksError(missing)
        stored = self.history.get(user_id, _empty_history())
        history = build_request_history(stored, user_id, ratings)
        item_ids = recommend_user(
            self.model,
            user_id,
            history,
            extra_seen=self.extra_seen.get(user_id, set()),
        )
        books = []
        for rank, item_id in enumerate(item_ids, start=1):
            meta = self.books.get(item_id, {})
            books.append(
                {
                    "id_libro": item_id,
                    "titulo": meta.get("titulo", ""),
                    "autor": meta.get("autor", ""),
                    "genero": meta.get("genero", ""),
                    "editorial": meta.get("editorial", ""),
                    "anio_edicion": meta.get("anio_edicion", ""),
                    "isbn": meta.get("isbn", ""),
                    "resumen": meta.get("resumen", ""),
                    "img_src": meta.get("img_src", ""),
                    "rank": rank,
                }
            )
        return {
            "id_lector": user_id,
            "conocido": user_id in self.known_ids,
            "libros": books,
        }


def load_serving_index(artifact: str | Path, db_path: str | Path) -> ServingIndex:
    """Carga el joblib y la base de solo lectura una vez, al prender el proceso."""

    model = load_serving_model(artifact)
    bundle = load_dataset(db_path)
    books = {
        str(row[ITEM_COL]): _book_card(row) for row in bundle.books.to_dict(orient="records")
    }
    history_frame = bundle.interactions[[USER_COL, ITEM_COL, DATE_COL, RATING_COL]]
    history = {
        str(user): frame.reset_index(drop=True)
        for user, frame in history_frame.groupby(USER_COL, sort=False)
    }
    extra_seen: dict[str, set[str]] = {}
    invalid = bundle.invalid_interactions
    if not invalid.empty:
        for user, frame in invalid.groupby(USER_COL, sort=False):
            extra_seen[str(user)] = set(frame[ITEM_COL].astype(str))
    known_ids = (
        set(bundle.readers[USER_COL].astype(str)) | set(history) | set(extra_seen)
    )
    LOGGER.info(
        "Serving listo: %s libros, %s lectores con historial",
        len(books),
        len(history),
    )
    return ServingIndex(
        model=model,
        books=books,
        history=history,
        extra_seen=extra_seen,
        known_ids=known_ids,
    )


def _book_card(row: dict[str, Any]) -> dict[str, str]:
    """Ficha de un libro para la respuesta de la API."""

    return {
        "titulo": _text(row.get("titulo")),
        "autor": _text(row.get("autor")),
        "genero": _text(row.get("genero")),
        "editorial": _text(row.get("editorial")),
        "anio_edicion": _year(row.get("anio_edicion")),
        "isbn": _isbn(row.get("isbn")),
        "resumen": _text(row.get("resumen")),
        "img_src": _text(row.get("img_src")),
    }


def _missing(value: object) -> bool:
    """True si el valor es nulo o NaN."""

    if value is None:
        return True
    return bool(pd.isna(value))


def _text(value: object) -> str:
    """Texto recortado, o cadena vacía si falta."""

    if _missing(value):
        return ""
    return str(value).strip()


def _year(value: object) -> str:
    """Año de edición entre 1000 y 2100, o cadena vacía."""

    if _missing(value):
        return ""
    try:
        year = int(float(str(value)))
    except (TypeError, ValueError):
        return ""
    if 1000 <= year <= 2100:
        return str(year)
    return ""


def _isbn(value: object) -> str:
    """ISBN como texto, sin el decimal que deja un float."""

    if _missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def unknown_book_ids(book_ids: list[str], catalog: set[str]) -> list[str]:
    """Ids puntuados que no están en el catálogo, en el orden recibido."""

    missing: list[str] = []
    seen: set[str] = set()
    for book_id in book_ids:
        if book_id in catalog or book_id in seen:
            continue
        seen.add(book_id)
        missing.append(book_id)
    return missing


def build_request_history(
    stored: pd.DataFrame,
    user_id: str,
    ratings: list[tuple[str, int]],
) -> pd.DataFrame:
    """Historial de la base más la sesión. La sesión reemplaza y queda última."""

    columns = [USER_COL, ITEM_COL, DATE_COL, RATING_COL]
    if stored.empty:
        base = _empty_history()
    else:
        base = stored.loc[stored[USER_COL].astype(str) == user_id, columns].copy()
        base[DATE_COL] = pd.to_datetime(base[DATE_COL])
        base[RATING_COL] = pd.to_numeric(base[RATING_COL]).astype("int8")
    if not ratings:
        return base.reset_index(drop=True)

    if base.empty or base[DATE_COL].isna().all():
        start = pd.Timestamp("2026-01-01")
    else:
        start = pd.Timestamp(base[DATE_COL].max())
    rows = [
        {
            USER_COL: user_id,
            ITEM_COL: item_id,
            DATE_COL: start + pd.Timedelta(days=offset),
            RATING_COL: rating,
        }
        for offset, (item_id, rating) in enumerate(ratings, start=1)
    ]
    session = pd.DataFrame(rows, columns=columns)
    session[RATING_COL] = session[RATING_COL].astype("int8")
    session[DATE_COL] = pd.to_datetime(session[DATE_COL])
    if not base.empty:
        base = base.loc[~base[ITEM_COL].astype(str).isin(session[ITEM_COL])].copy()
    return pd.concat([base, session], ignore_index=True)


def recommend_user(
    model: ServingModel,
    user_id: str,
    history: pd.DataFrame,
    *,
    extra_seen: set[str] | None = None,
    n: int = TOP_N,
) -> list[str]:
    """Top-n libros para un lector. No ajusta fuentes ni escribe el builder."""

    if n < 1:
        raise ValueError("n debe ser positivo")
    user_id = str(user_id)
    profile = _profile(history, user_id)
    blocked = set(profile[ITEM_COL].astype(str)) | set(extra_seen or ())
    per_source = per_source_depth(
        model.rrf_cutoff,
        rrf_k=model.rrf_k,
        n_sources=len(model.retrievers),
    )
    frames = [
        _recommend_source(retriever, user_id, profile, per_source)
        for retriever in model.retrievers
    ]
    frames = [frame for frame in frames if not frame.empty]
    if frames:
        candidates = pd.concat(frames, ignore_index=True)
    else:
        candidates = pd.DataFrame(columns=CANDIDATE_COLUMNS)

    fusion = CandidateFusion(weights=model.weights, rrf_k=model.rrf_k)
    fused = fusion.fuse(candidates, model.rrf_cutoff)
    chosen: list[str] = []
    if not fused.candidates.empty:
        features = _builder_for_request(model.feature_builder, profile).transform(
            fused.candidates,
            fused.source_candidates,
        )
        _ensure_ranker_columns(features.frame, model.ranker.feature_columns_)
        ranked = model.ranker.predict(features)
        for item_id in ranked[ITEM_COL].astype(str):
            if item_id in blocked or item_id in chosen:
                continue
            chosen.append(item_id)
            if len(chosen) == n:
                break

    if len(chosen) < n:
        chosen = _fill_with_popularity(
            model,
            user_id,
            blocked | set(chosen),
            chosen,
            n,
        )
    if len(chosen) != n:
        raise RuntimeError(f"No se pudieron completar {n} libros para {user_id}")
    return chosen


def _empty_history() -> pd.DataFrame:
    """Historial vacío con las columnas y los tipos del resto del pipeline."""

    return pd.DataFrame(
        {
            USER_COL: pd.Series(dtype="str"),
            ITEM_COL: pd.Series(dtype="str"),
            DATE_COL: pd.Series(dtype="datetime64[ns]"),
            RATING_COL: pd.Series(dtype="int8"),
        }
    )


def _profile(history: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """Filas de un lector, con fecha y rating ya tipados."""

    columns = [USER_COL, ITEM_COL, DATE_COL, RATING_COL]
    if history.empty:
        return _empty_history()
    profile = history.loc[history[USER_COL].astype(str) == user_id, columns].copy()
    if profile.empty:
        return _empty_history()
    profile[DATE_COL] = pd.to_datetime(profile[DATE_COL])
    profile[RATING_COL] = pd.to_numeric(profile[RATING_COL]).astype("int8")
    return profile.reset_index(drop=True)


def _recommend_source(
    retriever: Any,
    user_id: str,
    history: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """Candidatos de una fuente. ALS recalcula el factor de este lector."""

    if isinstance(retriever, ImplicitMFRetriever) and retriever.algorithm == "als":
        return retriever.recommend_fold_in(user_id, history, n)
    return retriever.recommend([user_id], history, n)


def _builder_for_request(
    builder: ColumnBuilder,
    history: pd.DataFrame,
) -> ColumnBuilder:
    """Copia superficial: las tablas de usuario son nuevas, el resto se comparte."""

    clone = ColumnBuilder(recent_n=builder.recent_n, smoothing=builder.smoothing)
    clone.__dict__ = dict(builder.__dict__)
    frames = builder.user_frames_for(history)
    clone.user_stats_ = frames["user_stats"]
    clone.user_author_ = frames["user_author"]
    clone.user_genre_ = frames["user_genre"]
    clone.user_last_ = frames["user_last"]
    clone.user_recent_author_ = frames["user_recent_author"]
    clone.user_recent_genre_ = frames["user_recent_genre"]
    return clone


def _ensure_ranker_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Rellena con NaN las features que el ranker espera y este request no trae."""

    for column in columns:
        if column not in frame:
            frame[column] = np.nan


def _fill_with_popularity(
    model: ServingModel,
    user_id: str,
    blocked: set[str],
    chosen: list[str],
    n: int,
) -> list[str]:
    """Completa el top con popularidad, sin repetir ni devolver lo ya bloqueado."""

    popularity = _popularity(model)
    blocker = pd.DataFrame(
        {USER_COL: user_id, ITEM_COL: list(blocked)},
    )
    extra = popularity.recommend([user_id], blocker, n)
    for item_id in extra[ITEM_COL].astype(str):
        if item_id in blocked or item_id in chosen:
            continue
        chosen.append(item_id)
        if len(chosen) == n:
            break
    return chosen


def _popularity(model: ServingModel) -> PopularityRecommender:
    """Fuente de popularidad guardada en el artefacto."""

    for retriever in model.retrievers:
        if isinstance(retriever, PopularityRecommender):
            return retriever
    raise RuntimeError("El artefacto no incluye la fuente de popularidad")
