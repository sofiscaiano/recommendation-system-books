from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quelibroleo.features import FeatureBuilder
from quelibroleo.model import RatingModel


def seen_items_by_user(
    df: pd.DataFrame,
    user_col: str = "id_lector",
    item_col: str = "id_libro",
) -> dict[str, set]:
    return df.groupby(user_col, sort=False)[item_col].agg(set).to_dict()


def recommend_top_k(
    model: RatingModel,
    features: FeatureBuilder,
    usuarios: pd.DataFrame,
    libros: pd.DataFrame,
    seen: Mapping[str, set] | None = None,
    k: int = 20,
    batch_size: int = 8,
    user_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Top-k sobre el catálogo completo (menos ítems ya vistos), por lotes."""
    seen = seen or {}
    users = features.user_frame(usuarios)
    items = features.item_frame(libros)
    if user_ids is not None:
        users = users[users["id_lector"].isin(user_ids)]
    users = users.reset_index(drop=True)
    items = items.reset_index(drop=True)

    bloques: list[pd.DataFrame] = []
    n_users_total = len(users)

    for start in range(0, n_users_total, batch_size):
        if start % (batch_size * 10) == 0:
            print(f"  recomendando {start}/{n_users_total}")
        batch_users = users.iloc[start : start + batch_size].reset_index(drop=True)
        pairs = batch_users.merge(items, how="cross")

        if seen:
            visto = pd.DataFrame(
                [
                    (lector, libro)
                    for lector in batch_users["id_lector"]
                    for libro in seen.get(lector, ())
                ],
                columns=["id_lector", "id_libro"],
            )
            if not visto.empty:
                visto["_seen"] = 1
                pairs = pairs.merge(visto, on=["id_lector", "id_libro"], how="left")
                pairs = pairs[pairs["_seen"].isna()].drop(columns="_seen")

        if pairs.empty:
            continue

        pairs = features.attach_affinities(pairs)
        pairs["edad"] = features.ref_year - pd.to_numeric(pairs["nacimiento"], errors="coerce")
        pairs["score_pred"] = model.predict(pairs).to_numpy()

        top = (
            pairs[["id_lector", "id_libro", "score_pred"]]
            .sort_values(["id_lector", "score_pred"], ascending=[True, False])
            .groupby("id_lector", sort=False)
            .head(k)
            .copy()
        )
        top["ranking"] = top.groupby("id_lector").cumcount() + 1
        bloques.append(top)

    if not bloques:
        return pd.DataFrame(columns=["id_lector", "id_libro", "score_pred", "ranking"])
    return pd.concat(bloques, ignore_index=True)


def fill_missing_with_ejemplo(
    df_pred: pd.DataFrame,
    ejemplo: pd.DataFrame,
    user_ids: Sequence[str],
    k: int = 20,
) -> pd.DataFrame:
    procesados = set(df_pred["id_lector"].unique()) if not df_pred.empty else set()
    faltantes = [u for u in user_ids if u not in procesados]
    if not faltantes:
        return df_pred

    relleno = ejemplo[ejemplo["id_lector"].isin(faltantes)].copy()
    if "ranking" not in relleno.columns:
        relleno["ranking"] = relleno.groupby("id_lector").cumcount() + 1
    if "score_pred" not in relleno.columns:
        relleno["score_pred"] = np.nan

    cols = ["id_lector", "id_libro", "score_pred", "ranking"]
    combinado = pd.concat([df_pred[cols], relleno[cols]], ignore_index=True)
    return combinado.sort_values(["id_lector", "ranking"]).reset_index(drop=True)
