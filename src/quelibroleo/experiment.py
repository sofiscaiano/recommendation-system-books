"""Datos fijos de un experimento. Ningún modelo arma su propio corte."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .baselines import PopularityRecommender
from .config import PipelineConfig
from .data import ITEM_COL, USER_COL, DataQualityReport, load_dataset
from .splits import RollingFold, leave_last_k, rolling_origin_folds


@dataclass(frozen=True)
class Experiment:
    """Train, validación y test compartidos por todos los modelos.

    En evaluación, ``history`` es todo lo anterior a las últimas 20
    interacciones y ``test`` son esas 20. ``validation_folds`` parte
    ``history`` en los mismos tres cortes: el último es la validación.
    En la entrega, ``history`` es el historial completo y ``test`` está vacío.
    """

    holdout: bool
    books: pd.DataFrame
    readers: pd.DataFrame
    history: pd.DataFrame
    test: pd.DataFrame
    evaluation_users: tuple[str, ...]
    submit_users: list[str]
    invalid_interactions: pd.DataFrame
    quality: DataQualityReport


def load_experiment(config: PipelineConfig, *, holdout: bool) -> Experiment:
    """Carga ``data.db`` y aplica el único protocolo de corte."""

    bundle = load_dataset(config.db_path)
    if holdout:
        outer = leave_last_k(bundle.interactions, k=config.split.holdout_k)
        history, test, users = outer.train, outer.test, outer.evaluation_users
    else:
        history = bundle.interactions
        test = history.iloc[0:0].copy()
        users = ()
    return Experiment(
        holdout=holdout,
        books=bundle.books,
        readers=bundle.readers,
        history=history,
        test=test,
        evaluation_users=users,
        submit_users=_submit_users(config),
        invalid_interactions=bundle.invalid_interactions,
        quality=bundle.report,
    )


def validation_folds(history: pd.DataFrame, config: PipelineConfig) -> list[RollingFold]:
    """Folds rolling dentro del train. El último es la validación."""

    return rolling_origin_folds(
        history,
        n_folds=config.split.rolling_folds,
        horizon=config.split.rolling_horizon,
        min_history=config.split.min_history,
    )


def seen_history(experiment: Experiment) -> pd.DataFrame:
    """Historial que no se puede volver a recomendar, fechas inválidas incluidas."""

    return pd.concat(
        [experiment.history, experiment.invalid_interactions],
        ignore_index=True,
        sort=False,
    )


def fill_submission(
    ranked: pd.DataFrame,
    users: list[str],
    history: pd.DataFrame,
    popularity: PopularityRecommender,
) -> pd.DataFrame:
    """20 libros por lector, sin repetir ni recomendar lo ya leído."""

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


def _submit_users(config: PipelineConfig) -> list[str]:
    example = pd.read_csv(config.example_path, dtype=str)
    return list(dict.fromkeys(example[USER_COL].astype(str)))
