"""Tabla numérica que el LambdaRank de ``two_stage`` usa para ordenar candidatos.

Retrieval ya eligió los libros. Acá cada candidato se describe con scores de
las fuentes, estadísticas de lector e ítem, afinidad de autor y género, y
señales de la secuencia de lectura. Un modelo único, sin etapa de retrieval,
no usa este archivo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import DATE_COL, ITEM_COL, RATING_COL, USER_COL


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

    def fit(
        self,
        train: pd.DataFrame,
        catalog: pd.DataFrame,
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

        user_stats = (
            enriched.groupby(USER_COL)
            .agg(
                user_count=(ITEM_COL, "size"),
                user_mean_rating=(RATING_COL, "mean"),
                user_std_rating=(RATING_COL, "std"),
                user_first_date=(DATE_COL, "min"),
                user_last_date=(DATE_COL, "max"),
                user_distinct_authors=("autor", "nunique"),
                user_distinct_genres=("genero", "nunique"),
            )
            .reset_index()
        )
        user_stats["user_active_days"] = (
            user_stats["user_last_date"] - user_stats["user_first_date"]
        ).dt.days
        user_stats = user_stats.drop(columns="user_first_date")
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
        frame["publication_age"] = (
            frame["user_cutoff_date"].dt.year - frame["anio_edicion_num"]
        )
        frame.loc[frame["publication_age"] < 0, "publication_age"] = np.nan
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
