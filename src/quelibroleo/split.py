from __future__ import annotations

import pandas as pd


def leave_last_k(
    df: pd.DataFrame,
    k: int = 20,
    min_ratings: int = 21,
    user_col: str = "id_lector",
    date_col: str = "fecha",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold-out temporal: últimas k interacciones por usuario con al menos min_ratings."""
    counts = df.groupby(user_col)["rating"].transform("count")
    filtrado = df[counts >= min_ratings]
    ordenado = filtrado.sort_values(by=[user_col, date_col])
    df_test = ordenado.groupby(user_col, sort=False).tail(k)
    df_train = ordenado.drop(df_test.index)
    return df_train, df_test
