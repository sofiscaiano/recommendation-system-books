from __future__ import annotations

import pandas as pd

from .baselines import PopularityRecommender
from .data import ITEM_COL, USER_COL


def fill_submission(
    ranked: pd.DataFrame,
    users: list[str],
    history: pd.DataFrame,
    popularity: PopularityRecommender,
) -> pd.DataFrame:
    top = ranked[ranked["rank"] <= 20][[USER_COL, ITEM_COL, "rank"]].copy()
    existing = top.groupby(USER_COL)[ITEM_COL].agg(list).to_dict()
    fallback = popularity.recommend(users, history, n=100)
    fallback_by_user = fallback.groupby(USER_COL)[ITEM_COL].agg(list).to_dict()
    seen_by_user = history.groupby(USER_COL)[ITEM_COL].agg(set).to_dict()
    rows = []
    for user in users:
        chosen: list[str] = []
        for item in existing.get(user, []) + fallback_by_user.get(user, []):
            if item in seen_by_user.get(user, set()) or item in chosen:
                continue
            chosen.append(item)
            if len(chosen) == 20:
                break
        if len(chosen) != 20:
            raise RuntimeError(f"No se pudieron completar 20 libros para {user}")
        rows.extend({USER_COL: user, ITEM_COL: item} for item in chosen)
    return pd.DataFrame.from_records(rows, columns=[USER_COL, ITEM_COL])
