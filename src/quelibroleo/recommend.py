from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quelibroleo.features import FeatureBuilder
from quelibroleo.model import RatingModel


def candidate_books(
    libros: pd.DataFrame,
    interacciones: pd.DataFrame,
    *,
    only_interacted_items: bool,
    item_col: str = "id_libro",
) -> pd.DataFrame:
    """Catálogo a rankear. Si ``only_interacted_items``, descarta libros sin ninguna interacción."""
    if not only_interacted_items:
        return libros
    ids = interacciones[item_col].dropna().unique()
    return libros[libros[item_col].isin(ids)].copy()


def seen_items_by_user(
    df: pd.DataFrame,
    user_col: str = "id_lector",
    item_col: str = "id_libro",
) -> dict[str, set]:
    """Mapea cada lector al conjunto de libros con los que ya interactuó."""
    return df.groupby(user_col, sort=False)[item_col].agg(set).to_dict()


def _user_value_pairs(history: pd.DataFrame, col: str, user_col: str = "id_lector") -> pd.DataFrame:
    """Pares únicos lector-valor (género o autor) presentes en el historial."""
    return history[[user_col, col]].dropna().drop_duplicates()


def filter_pairs_by_seen_affinity(
    pairs: pd.DataFrame,
    *,
    seen_genero: pd.DataFrame | None,
    seen_autor: pd.DataFrame | None,
    user_col: str = "id_lector",
) -> pd.DataFrame:
    """Deja libros de géneros y/o autores ya vistos. Si ambos filtros están activos, es unión (OR)."""
    if seen_genero is None and seen_autor is None:
        return pairs

    keep = np.zeros(len(pairs), dtype=bool)
    if seen_genero is not None:
        matched = pairs.merge(seen_genero.assign(_ok=1), on=[user_col, "genero"], how="left")["_ok"].eq(1)
        keep |= matched.to_numpy()
    if seen_autor is not None:
        matched = pairs.merge(seen_autor.assign(_ok=1), on=[user_col, "autor"], how="left")["_ok"].eq(1)
        keep |= matched.to_numpy()
    return pairs.iloc[keep]


def recommend_top_k(
    model: RatingModel,
    features: FeatureBuilder,
    usuarios: pd.DataFrame,
    libros: pd.DataFrame,
    seen: Mapping[str, set] | None = None,
    k: int = 20,
    batch_size: int = 8,
    user_ids: Sequence[str] | None = None,
    history: pd.DataFrame | None = None,
    filter_seen_genres: bool = True,
    filter_seen_authors: bool = True,
) -> pd.DataFrame:
    """Top-k sobre candidatos, menos ya vistos. Filtros de género/autor usan ``history``."""
    seen = seen or {}
    users = features.user_frame(usuarios)
    items = features.item_frame(libros)
    if user_ids is not None:
        users = users[users["id_lector"].isin(user_ids)]
    users = users.reset_index(drop=True)
    items = items.reset_index(drop=True)

    hist = history if history is not None else pd.DataFrame(columns=["id_lector", "genero", "autor"])
    seen_genero = _user_value_pairs(hist, "genero") if filter_seen_genres else None
    seen_autor = _user_value_pairs(hist, "autor") if filter_seen_authors else None

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

        pairs = filter_pairs_by_seen_affinity(
            pairs,
            seen_genero=seen_genero,
            seen_autor=seen_autor,
        )

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
    """Completa lectores sin ranking propio con las filas de ``ejemplo.csv``."""
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
