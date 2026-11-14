from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import PipelineConfig
from .data import USER_COL, DataQualityReport, load_dataset
from .splits import RollingFold, leave_last_k, rolling_origin_folds


@dataclass(frozen=True)
class SplitData:
    holdout: bool
    books: pd.DataFrame
    readers: pd.DataFrame
    history: pd.DataFrame
    test: pd.DataFrame
    evaluation_users: tuple[str, ...]
    submit_users: list[str]
    invalid_interactions: pd.DataFrame
    quality: DataQualityReport


def load_split(config: PipelineConfig, *, holdout: bool) -> SplitData:
    bundle = load_dataset(config.db_path)
    if holdout:
        outer = leave_last_k(bundle.interactions, k=config.split.holdout_k)
        history, test, users = outer.train, outer.test, outer.evaluation_users
    else:
        history = bundle.interactions
        test = history.iloc[0:0].copy()
        users = ()
    return SplitData(
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


def validation_folds(
    history: pd.DataFrame, config: PipelineConfig
) -> list[RollingFold]:
    return rolling_origin_folds(
        history,
        n_folds=config.split.rolling_folds,
        horizon=config.split.rolling_horizon,
        min_history=config.split.min_history,
    )


def seen_books(split: SplitData) -> pd.DataFrame:
    return pd.concat(
        [split.history, split.invalid_interactions],
        ignore_index=True,
        sort=False,
    )


def example_users(config: PipelineConfig) -> set[str]:
    return set(_submit_users(config))


def _submit_users(config: PipelineConfig) -> list[str]:
    example = pd.read_csv(config.example_path, dtype=str)
    return list(dict.fromkeys(example[USER_COL].astype(str)))
