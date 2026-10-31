"""Tabla numérica que el LambdaRank de ``two_stage`` usa para ordenar candidatos.

Retrieval ya eligió los libros. Acá cada candidato se describe con scores de
las fuentes, estadísticas de lector e ítem, afinidad de autor y género, y
señales de la secuencia de lectura. El género del lector entra como 1 (mujer)
o 0 (hombre). Un modelo único, sin etapa de retrieval, no usa este archivo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import DATE_COL, ITEM_COL, RATING_COL, USER_COL

_GENDER_CODE = {"Mujer": 1, "Hombre": 0}
_TRANSITION_LAGS = 3
# El más reciente pesa 3, el anterior 2 y el tercero 1.
_TRANSITION_WEIGHTS = {1: 3.0, 2: 2.0, 3: 1.0}


def _safe_source(value: str) -> str:
    """Deja el nombre de una fuente usable como sufijo de columna."""

    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


@dataclass(frozen=True)
class RankerTable:
    """Filas de candidatos y los nombres de las columnas numéricas del ranker."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]


class ColumnBuilder:
    """Arma esas columnas a partir del historial anterior al corte."""

    def __init__(self, recent_n: int = 10, smoothing: float = 10.0) -> None:
        self.recent_n = recent_n
        self.smoothing = smoothing
        self.catalog_: pd.DataFrame | None = None
        self.global_mean_: float = 0.0
        self.user_stats_: pd.DataFrame | None = None
        self.item_stats_: pd.DataFrame | None = None
        self.user_author_: pd.DataFrame | None = None
        self.user_genre_: pd.DataFrame | None = None
        self.user_last_: pd.DataFrame | None = None
        self.user_recent_author_: pd.DataFrame | None = None
        self.user_recent_genre_: pd.DataFrame | None = None
        self.author_transition_: pd.DataFrame | None = None
        self.genre_transition_: pd.DataFrame | None = None
        self.author_stats_: pd.DataFrame | None = None
        self.genre_stats_: pd.DataFrame | None = None
        self.readers_: pd.DataFrame | None = None

    def fit(
        self,
        train: pd.DataFrame,
        catalog: pd.DataFrame,
        readers: pd.DataFrame | None = None,
    ) -> ColumnBuilder:
        """Estima estadísticas de lector, ítem, autor, género y transiciones."""

        metadata_columns = [
            ITEM_COL,
            "autor",
            "genero",
            "anio_edicion_num",
        ]
        available = [column for column in metadata_columns if column in catalog]
        metadata = catalog[available].drop_duplicates(ITEM_COL).copy()
        for column in ("autor", "genero"):
            if column not in metadata:
                metadata[column] = ""
            metadata[column] = metadata[column].fillna("").astype(str)
        if "anio_edicion_num" not in metadata:
            metadata["anio_edicion_num"] = np.nan
        self.catalog_ = metadata
        self.readers_ = _reader_demographics(readers)
        self.global_mean_ = float(train[RATING_COL].mean())

        ordered = train.sort_values(
            [USER_COL, DATE_COL, ITEM_COL], kind="mergesort"
        )
        enriched = ordered.merge(metadata, on=ITEM_COL, how="left")
        enriched[["autor", "genero"]] = enriched[["autor", "genero"]].fillna("")

        activity = self._user_activity_tables(enriched)
        self.user_stats_ = activity["user_stats"]
        self.user_author_ = activity["user_author"]
        self.user_genre_ = activity["user_genre"]
        self.user_last_ = activity["user_last"]
        self.user_recent_author_ = activity["user_recent_author"]
        self.user_recent_genre_ = activity["user_recent_genre"]

        item_stats = (
            enriched.groupby(ITEM_COL)
            .agg(
                item_count=(USER_COL, "size"),
                item_mean_rating=(RATING_COL, "mean"),
                item_std_rating=(RATING_COL, "std"),
                item_distinct_users=(USER_COL, "nunique"),
            )
            .reset_index()
        )
        item_stats["item_bayes_rating"] = (
            item_stats["item_count"] * item_stats["item_mean_rating"]
            + self.smoothing * self.global_mean_
        ) / (item_stats["item_count"] + self.smoothing)
        self.item_stats_ = item_stats
        self.author_stats_ = (
            enriched.groupby("autor", as_index=False)
            .agg(
                author_read_count=(ITEM_COL, "size"),
                author_mean_rating=(RATING_COL, "mean"),
            )
        )
        self.genre_stats_ = (
            enriched.groupby("genero", as_index=False)
            .agg(genre_read_count=(ITEM_COL, "size"))
        )

        transition_data = enriched[
            [USER_COL, "autor", "genero"]
        ].copy()
        transition_data["previous_author"] = transition_data.groupby(USER_COL)[
            "autor"
        ].shift()
        transition_data["previous_genre"] = transition_data.groupby(USER_COL)[
            "genero"
        ].shift()
        self.author_transition_ = self._transition_probabilities(
            transition_data,
            previous="previous_author",
            current="autor",
            output="author_transition_prob",
        )
        self.genre_transition_ = self._transition_probabilities(
            transition_data,
            previous="previous_genre",
            current="genero",
            output="genre_transition_prob",
        )
        return self

    @staticmethod
    def _transition_probabilities(
        frame: pd.DataFrame,
        *,
        previous: str,
        current: str,
        output: str,
    ) -> pd.DataFrame:
        """Probabilidad de pasar de un autor o género al siguiente."""

        valid = frame[frame[previous].notna()]
        counts = valid.groupby([previous, current]).size().rename("_pair")
        totals = valid.groupby(previous).size().rename("_total")
        result = counts.reset_index().merge(
            totals.reset_index(), on=previous, how="left"
        )
        result[output] = result["_pair"] / result["_total"]
        return result[[previous, current, output]]

    def _user_activity_tables(self, enriched: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Estadísticas, afinidades y últimos ítems de cada lector."""

        work = enriched.copy()
        work["_read_year"] = work[DATE_COL].dt.year
        work["_decade"] = work["anio_edicion_num"] // 10 * 10
        user_stats = (
            work.groupby(USER_COL)
            .agg(
                user_count=(ITEM_COL, "size"),
                user_mean_rating=(RATING_COL, "mean"),
                user_std_rating=(RATING_COL, "std"),
                user_first_date=(DATE_COL, "min"),
                user_last_date=(DATE_COL, "max"),
                user_mean_read_year=("_read_year", "mean"),
                user_distinct_authors=("autor", "nunique"),
                user_distinct_genres=("genero", "nunique"),
            )
            .reset_index()
        )
        user_stats["user_first_year"] = user_stats["user_first_date"].dt.year
        user_stats["user_last_year"] = user_stats["user_last_date"].dt.year
        user_stats["user_active_days"] = (
            user_stats["user_last_date"] - user_stats["user_first_date"]
        ).dt.days
        user_stats = user_stats.drop(columns="user_first_date")
        gaps = work.groupby(USER_COL, sort=False)[DATE_COL].diff().dt.days
        gap_mean = (
            gaps.groupby(work[USER_COL], sort=False)
            .mean()
            .rename("user_mean_days_between_reads")
            .reset_index()
        )
        user_stats = user_stats.merge(gap_mean, on=USER_COL, how="left")
        decade_counts = (
            work.dropna(subset=["_decade"]).groupby(USER_COL)["_decade"].nunique()
        )
        user_stats["user_decade_diversity"] = (
            user_stats[USER_COL].map(decade_counts).fillna(0) / user_stats["user_count"]
        )
        user_stats["user_author_diversity"] = (
            user_stats["user_distinct_authors"] / user_stats["user_count"]
        )
        user_stats["user_genre_diversity"] = (
            user_stats["user_distinct_genres"] / user_stats["user_count"]
        )
        user_stats = _attach_reader_demographics(user_stats, self.readers_)
        user_author = (
            enriched.groupby([USER_COL, "autor"])
            .agg(
                user_author_count=(ITEM_COL, "size"),
                user_author_mean=(RATING_COL, "mean"),
            )
            .reset_index()
        )
        user_genre = (
            enriched.groupby([USER_COL, "genero"])
            .agg(
                user_genre_count=(ITEM_COL, "size"),
                user_genre_mean=(RATING_COL, "mean"),
            )
            .reset_index()
        )
        last = enriched.groupby(USER_COL, sort=False).tail(1)
        user_last = last[[USER_COL, DATE_COL, "autor", "genero"]].rename(
            columns={
                DATE_COL: "user_cutoff_date",
                "autor": "last_author",
                "genero": "last_genre",
            }
        )
        user_last = user_last.merge(
            _recent_labels(enriched, "autor", "author"), on=USER_COL, how="left"
        )
        user_last = user_last.merge(
            _recent_labels(enriched, "genero", "genre"), on=USER_COL, how="left"
        )
        recent = enriched.groupby(USER_COL, sort=False).tail(self.recent_n)
        return {
            "user_stats": user_stats,
            "user_author": user_author,
            "user_genre": user_genre,
            "user_last": user_last,
            "user_recent_author": (
                recent.groupby([USER_COL, "autor"])
                .size()
                .rename("recent_author_count")
                .reset_index()
            ),
            "user_recent_genre": (
                recent.groupby([USER_COL, "genero"])
                .size()
                .rename("recent_genre_count")
                .reset_index()
            ),
        }

    def user_frames_for(self, history: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Tablas de usuario para un historial, sin modificar las de ``fit``."""

        if any(
            value is None
            for value in (
                self.catalog_,
                self.user_stats_,
                self.user_author_,
                self.user_genre_,
                self.user_last_,
                self.user_recent_author_,
                self.user_recent_genre_,
            )
        ):
            raise RuntimeError("ColumnBuilder debe entrenarse antes de transformar")

        stored = {
            "user_stats": self.user_stats_,
            "user_author": self.user_author_,
            "user_genre": self.user_genre_,
            "user_last": self.user_last_,
            "user_recent_author": self.user_recent_author_,
            "user_recent_genre": self.user_recent_genre_,
        }
        if history.empty:
            return {key: frame.iloc[0:0].copy() for key, frame in stored.items()}

        metadata = self.catalog_
        ordered = history.sort_values([USER_COL, DATE_COL, ITEM_COL], kind="mergesort")
        enriched = ordered.merge(metadata, on=ITEM_COL, how="left")
        enriched[["autor", "genero"]] = enriched[["autor", "genero"]].fillna("")
        enriched[RATING_COL] = pd.to_numeric(enriched[RATING_COL])
        return self._user_activity_tables(enriched)

    def transform(
        self,
        fused_candidates: pd.DataFrame,
        source_candidates: pd.DataFrame,
    ) -> RankerTable:
        """Arma las columnas numéricas de cada candidato ya fusionado."""

        if any(
            value is None
            for value in (
                self.catalog_,
                self.user_stats_,
                self.item_stats_,
                self.user_author_,
                self.user_genre_,
                self.user_last_,
                self.author_stats_,
                self.genre_stats_,
                self.author_transition_,
                self.genre_transition_,
            )
        ):
            raise RuntimeError("ColumnBuilder debe entrenarse antes de transformar")

        frame = fused_candidates.copy()
        if frame.empty:
            return RankerTable(frame=frame, feature_columns=())

        source = source_candidates.merge(
            frame[[USER_COL, ITEM_COL]].drop_duplicates(),
            on=[USER_COL, ITEM_COL],
            how="inner",
        )
        source["source_safe"] = source["source"].map(_safe_source)
        score_wide = source.pivot_table(
            index=[USER_COL, ITEM_COL],
            columns="source_safe",
            values="score",
            aggfunc="first",
        )
        rank_wide = source.pivot_table(
            index=[USER_COL, ITEM_COL],
            columns="source_safe",
            values="rank",
            aggfunc="first",
        )
        score_wide.columns = [f"score_{column}" for column in score_wide.columns]
        rank_wide.columns = [f"rank_{column}" for column in rank_wide.columns]
        wide = score_wide.join(rank_wide, how="outer").reset_index()
        frame = frame.merge(wide, on=[USER_COL, ITEM_COL], how="left")
        for column in [column for column in frame if column.startswith("rank_")]:
            source_name = column.removeprefix("rank_")
            frame[f"present_{source_name}"] = frame[column].notna().astype("int8")
            frame[f"reciprocal_{source_name}"] = 1.0 / (60.0 + frame[column])

        frame = frame.merge(self.catalog_, on=ITEM_COL, how="left")
        frame = frame.merge(self.user_stats_, on=USER_COL, how="left")
        frame = frame.merge(self.item_stats_, on=ITEM_COL, how="left")
        frame = frame.merge(self.user_author_, on=[USER_COL, "autor"], how="left")
        frame = frame.merge(self.user_genre_, on=[USER_COL, "genero"], how="left")
        frame = frame.merge(self.author_stats_, on="autor", how="left")
        frame = frame.merge(self.genre_stats_, on="genero", how="left")
        frame = frame.merge(self.user_last_, on=USER_COL, how="left")
        frame = frame.merge(
            self.user_recent_author_, on=[USER_COL, "autor"], how="left"
        )
        frame = frame.merge(
            self.user_recent_genre_, on=[USER_COL, "genero"], how="left"
        )
        frame = frame.merge(
            self.author_transition_,
            left_on=["last_author", "autor"],
            right_on=["previous_author", "autor"],
            how="left",
        )
        frame = frame.merge(
            self.genre_transition_,
            left_on=["last_genre", "genero"],
            right_on=["previous_genre", "genero"],
            how="left",
        )

        frame["same_last_author"] = (
            (frame["autor"] != "") & (frame["autor"] == frame["last_author"])
        ).astype("int8")
        frame["same_last_genre"] = (
            (frame["genero"] != "") & (frame["genero"] == frame["last_genre"])
        ).astype("int8")
        frame["recent_author_share"] = (
            frame["recent_author_count"].fillna(0) / self.recent_n
        )
        frame["recent_genre_share"] = (
            frame["recent_genre_count"].fillna(0) / self.recent_n
        )
        frame["user_author_share"] = (
            frame["user_author_count"].fillna(0) / frame["user_count"]
        )
        frame["user_genre_share"] = (
            frame["user_genre_count"].fillna(0) / frame["user_count"]
        )
        frame["item_author_share"] = frame["item_count"] / frame["author_read_count"]
        frame["item_genre_share"] = frame["item_count"] / frame["genre_read_count"]
        frame.loc[frame["autor"].fillna("") == "", "item_author_share"] = np.nan
        frame.loc[frame["genero"].fillna("") == "", "item_genre_share"] = np.nan
        frame["item_author_rating_ratio"] = (
            frame["item_mean_rating"] / frame["author_mean_rating"]
        )
        frame["publication_age"] = (
            frame["user_cutoff_date"].dt.year - frame["anio_edicion_num"]
        )
        frame["edition_vs_user_mean_year"] = (
            frame["anio_edicion_num"] - frame["user_mean_read_year"]
        )
        frame["author_transition_window"] = _window_transition(
            frame,
            self.author_transition_,
            lag_prefix="author",
            current="autor",
            previous="previous_author",
            probability="author_transition_prob",
        )
        frame["genre_transition_window"] = _window_transition(
            frame,
            self.genre_transition_,
            lag_prefix="genre",
            current="genero",
            previous="previous_genre",
            probability="genre_transition_prob",
        )
        frame["log_item_count"] = np.log1p(frame["item_count"])
        frame["log_user_count"] = np.log1p(frame["user_count"])

        excluded = {
            USER_COL,
            ITEM_COL,
            "autor",
            "genero",
            "last_author",
            "last_genre",
            "previous_author",
            "previous_genre",
            "user_cutoff_date",
            "user_last_date",
            "user_mean_read_year",
            "author_read_count",
            "genre_read_count",
            "author_lag1",
            "author_lag2",
            "author_lag3",
            "genre_lag1",
            "genre_lag2",
            "genre_lag3",
        }
        feature_columns = tuple(
            column
            for column in frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
        )
        numeric_columns = list(feature_columns)
        frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        return RankerTable(frame=frame, feature_columns=feature_columns)


def _reader_demographics(readers: pd.DataFrame | None) -> pd.DataFrame | None:
    """Género codificado y año de nacimiento. `Mujer` es 1 y `Hombre` es 0."""

    if readers is None or readers.empty:
        return None
    frame = readers.drop_duplicates(USER_COL).copy()
    if "genero" in frame.columns:
        frame["user_gender"] = frame["genero"].map(_GENDER_CODE)
    else:
        frame["user_gender"] = np.nan
    if "nacimiento" in frame.columns:
        frame["birth_year"] = pd.to_numeric(frame["nacimiento"], errors="coerce")
    else:
        frame["birth_year"] = np.nan
    return frame[[USER_COL, "user_gender", "birth_year"]]


def _attach_reader_demographics(
    user_stats: pd.DataFrame,
    readers: pd.DataFrame | None,
) -> pd.DataFrame:
    """Edad en años: año de la lectura menos año de nacimiento."""

    if readers is None:
        user_stats = user_stats.copy()
        user_stats["user_gender"] = np.nan
        user_stats["user_age_first"] = np.nan
        user_stats["user_age_last"] = np.nan
        return user_stats
    stats = user_stats.merge(readers, on=USER_COL, how="left")
    stats["user_age_first"] = stats["user_first_year"] - stats["birth_year"]
    stats["user_age_last"] = stats["user_last_year"] - stats["birth_year"]
    return stats.drop(columns="birth_year")


def _recent_labels(frame: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Últimas lecturas de un autor o género. El lag 1 es la más reciente."""

    tail = frame.groupby(USER_COL, sort=False).tail(_TRANSITION_LAGS).copy()
    tail["_lag"] = tail.groupby(USER_COL, sort=False).cumcount(ascending=False) + 1
    wide = tail.pivot(index=USER_COL, columns="_lag", values=column)
    wide = wide.reindex(columns=list(range(1, _TRANSITION_LAGS + 1)))
    wide.columns = [f"{prefix}_lag{lag}" for lag in wide.columns]
    return wide.reset_index()


def _window_transition(
    frame: pd.DataFrame,
    transition: pd.DataFrame,
    *,
    lag_prefix: str,
    current: str,
    previous: str,
    probability: str,
) -> np.ndarray:
    """Promedio ponderado de P(candidato | cada una de las últimas lecturas).

    Un par que no aparece en el historial aporta 0. Si el lector tiene menos
    de tres lecturas, ese puesto no entra al denominador.
    """

    lookup = transition.set_index([previous, current])[probability]
    weighted = np.zeros(len(frame), dtype=float)
    weight_sum = np.zeros(len(frame), dtype=float)
    current_values = frame[current].to_numpy()
    for lag, weight in _TRANSITION_WEIGHTS.items():
        previous_values = frame[f"{lag_prefix}_lag{lag}"]
        present = previous_values.notna().to_numpy()
        keys = pd.MultiIndex.from_arrays(
            [previous_values.to_numpy(), current_values]
        )
        found = lookup.reindex(keys).to_numpy(dtype=float)
        found = np.where(np.isnan(found), 0.0, found)
        weighted += np.where(present, found * weight, 0.0)
        weight_sum += np.where(present, weight, 0.0)
    result = np.full(len(frame), np.nan)
    np.divide(weighted, weight_sum, out=result, where=weight_sum > 0)
    return result


def attach_relevance(
    features: RankerTable,
    target: pd.DataFrame,
    *,
    query_suffix: str = "",
) -> RankerTable:
    """Asigna rating futuro a positivos y cero al resto de candidatos."""

    labels = target[[USER_COL, ITEM_COL, RATING_COL]].drop_duplicates(
        [USER_COL, ITEM_COL]
    )
    frame = features.frame.merge(
        labels, on=[USER_COL, ITEM_COL], how="left", validate="one_to_one"
    )
    frame["relevance"] = frame[RATING_COL].fillna(0).astype("int8")
    frame["query_id"] = frame[USER_COL].astype(str) + query_suffix
    return RankerTable(frame=frame, feature_columns=features.feature_columns)
