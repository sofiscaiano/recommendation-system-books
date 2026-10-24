"""Baselines no personalizados."""

from __future__ import annotations

import pandas as pd

from .data import ITEM_COL, USER_COL


class PopularityRecommender:
    """Ranking global por cantidad de interacciones."""

    source_name = "popularidad"

    def __init__(self) -> None:
        self.ranking_: pd.DataFrame | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        eligible_items: set[str] | None = None,
    ) -> PopularityRecommender:
        """Ordena el catálogo por cantidad de interacciones en el train."""

        counts = (
            interactions.groupby(ITEM_COL, as_index=False)
            .size()
            .rename(columns={"size": "score"})
        )
        if eligible_items is not None:
            counts = counts[counts[ITEM_COL].isin(eligible_items)]
        self.ranking_ = counts.sort_values(
            ["score", ITEM_COL],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return self

    def recommend(
        self,
        users: list[str] | tuple[str, ...],
        history: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """Aplica ese orden a cada lector y salta los libros ya leídos."""

        if self.ranking_ is None:
            raise RuntimeError("El recomendador debe entrenarse antes de recomendar")
        if n < 1:
            return pd.DataFrame(
                columns=[USER_COL, ITEM_COL, "source", "score", "rank"]
            )

        seen = history.groupby(USER_COL)[ITEM_COL].agg(set).to_dict()
        rows: list[dict[str, object]] = []
        item_scores = self.ranking_[[ITEM_COL, "score"]].itertuples(
            index=False, name=None
        )
        cached = list(item_scores)
        for user in users:
            user_seen = seen.get(user, set())
            rank = 0
            for item, score in cached:
                if item in user_seen:
                    continue
                rank += 1
                rows.append(
                    {
                        USER_COL: user,
                        ITEM_COL: item,
                        "source": self.source_name,
                        "score": float(score),
                        "rank": rank,
                    }
                )
                if rank == n:
                    break
        return pd.DataFrame.from_records(rows)
