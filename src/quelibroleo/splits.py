from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DATE_COL, ITEM_COL, USER_COL


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    evaluation_users: tuple[str, ...]
    holdout_k: int


@dataclass(frozen=True)
class RollingFold:
    fold: int
    train: pd.DataFrame
    target: pd.DataFrame
    query_users: tuple[str, ...]


def _ordered(interactions: pd.DataFrame) -> pd.DataFrame:
    required = {USER_COL, ITEM_COL, DATE_COL}
    missing = required - set(interactions.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    if interactions[DATE_COL].isna().any():
        raise ValueError("El split temporal no acepta fechas inválidas")

    ordered = interactions.sort_values(
        [USER_COL, DATE_COL, ITEM_COL],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered["_position"] = ordered.groupby(USER_COL, sort=False).cumcount()
    ordered["_user_size"] = ordered.groupby(USER_COL, sort=False)[ITEM_COL].transform(
        "size"
    )
    return ordered


def leave_last_k(
    interactions: pd.DataFrame,
    k: int = 20,
) -> TemporalSplit:
    if k < 1:
        raise ValueError("k debe ser positivo")

    ordered = _ordered(interactions)
    eligible = ordered["_user_size"] > k
    in_test = eligible & (ordered["_position"] >= ordered["_user_size"] - k)
    evaluation_users = tuple(
        sorted(ordered.loc[in_test, USER_COL].drop_duplicates().astype(str))
    )
    helper_columns = ["_position", "_user_size"]

    return TemporalSplit(
        train=ordered.loc[~in_test].drop(columns=helper_columns).reset_index(drop=True),
        test=ordered.loc[in_test].drop(columns=helper_columns).reset_index(drop=True),
        evaluation_users=evaluation_users,
        holdout_k=k,
    )


def rolling_origin_folds(
    interactions: pd.DataFrame,
    *,
    n_folds: int = 3,
    horizon: int = 20,
    min_history: int = 5,
) -> list[RollingFold]:
    if n_folds < 1 or horizon < 1 or min_history < 1:
        raise ValueError("n_folds, horizon y min_history deben ser positivos")

    ordered = _ordered(interactions)
    folds: list[RollingFold] = []
    user_sizes = ordered.groupby(USER_COL, sort=False)[ITEM_COL].size()

    for fold_index in range(n_folds):
        windows_from_end = n_folds - fold_index
        target_end = user_sizes - (windows_from_end - 1) * horizon
        target_start = target_end - horizon
        query_user_index = target_start[target_start >= min_history].index
        query_user_set = set(query_user_index)

        start_by_user = target_start.to_dict()
        end_by_user = target_end.to_dict()
        is_query = ordered[USER_COL].isin(query_user_set)
        starts = ordered[USER_COL].map(start_by_user).fillna(np.inf)
        ends = ordered[USER_COL].map(end_by_user).fillna(-np.inf)

        train_mask = (~is_query) | (ordered["_position"] < starts)
        target_mask = (
            is_query & (ordered["_position"] >= starts) & (ordered["_position"] < ends)
        )

        helper_columns = ["_position", "_user_size"]
        target = (
            ordered.loc[target_mask].drop(columns=helper_columns).reset_index(drop=True)
        )
        train = (
            ordered.loc[train_mask].drop(columns=helper_columns).reset_index(drop=True)
        )
        query_users = tuple(sorted(target[USER_COL].drop_duplicates().astype(str)))
        if query_users:
            folds.append(
                RollingFold(
                    fold=fold_index,
                    train=train,
                    target=target,
                    query_users=query_users,
                )
            )

    if not folds:
        raise ValueError("No hay historiales suficientes para construir folds")
    return folds


def assert_no_within_user_leakage(fold: RollingFold) -> None:
    query_train = fold.train[fold.train[USER_COL].isin(fold.query_users)]
    train_max = query_train.groupby(USER_COL)[DATE_COL].max()
    target_min = fold.target.groupby(USER_COL)[DATE_COL].min()
    common = train_max.index.intersection(target_min.index)
    if (train_max.loc[common] > target_min.loc[common]).any():
        raise AssertionError("Se detectó futuro del mismo lector dentro de train")
