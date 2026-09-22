"""Ranking final con LightGBM LambdaRank."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import RankerConfig
from .data import ITEM_COL, USER_COL
from .features import FeatureSet
from .metrics import ndcg_at_k


def _ordered_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    if "query_id" not in frame:
        raise ValueError("Falta query_id en el conjunto de ranking")
    ordered = frame.sort_values(
        ["query_id", USER_COL, ITEM_COL], kind="mergesort"
    ).reset_index(drop=True)
    groups = ordered.groupby("query_id", sort=False).size().tolist()
    return ordered, groups


class LambdaRanker:
    """Wrapper reproducible de ``LGBMRanker``."""

    def __init__(
        self,
        config: RankerConfig | None = None,
        *,
        random_state: int = 42,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or RankerConfig()
        self.random_state = random_state
        self.overrides = overrides or {}
        self.model_: lgb.LGBMRanker | None = None
        self.feature_columns_: tuple[str, ...] = ()
        self.best_iteration_: int | None = None

    def _parameters(self) -> dict[str, Any]:
        values = asdict(self.config)
        values.pop("early_stopping_rounds")
        values.pop("optuna_trials")
        values.update(self.overrides)
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": list(range(11)),
            "lambdarank_truncation_level": 20,
            "verbosity": -1,
            "random_state": self.random_state,
            "n_jobs": -1,
            **values,
        }

    def fit(
        self,
        train: FeatureSet,
        validation: FeatureSet | None = None,
    ) -> LambdaRanker:
        if "relevance" not in train.frame:
            raise ValueError("El train no contiene relevance")
        feature_columns = tuple(
            column for column in train.feature_columns if column in train.frame
        )
        if not feature_columns:
            raise ValueError("No hay features numéricas para LambdaRank")

        train_frame, train_groups = _ordered_groups(train.frame)
        model = lgb.LGBMRanker(**self._parameters())
        fit_kwargs: dict[str, Any] = {}
        if validation is not None and not validation.frame.empty:
            validation_frame, validation_groups = _ordered_groups(validation.frame)
            fit_kwargs.update(
                {
                    "eval_X": validation_frame.loc[:, list(feature_columns)],
                    "eval_y": validation_frame["relevance"],
                    "eval_group": [validation_groups],
                    "callbacks": [
                        lgb.early_stopping(
                            self.config.early_stopping_rounds,
                            verbose=False,
                        )
                    ],
                }
            )

        model.fit(
            train_frame.loc[:, list(feature_columns)],
            train_frame["relevance"],
            group=train_groups,
            eval_at=[20],
            **fit_kwargs,
        )
        self.model_ = model
        self.feature_columns_ = feature_columns
        self.best_iteration_ = model.best_iteration_
        return self

    def predict(self, features: FeatureSet) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("El ranker debe entrenarse antes de predecir")
        missing = set(self.feature_columns_) - set(features.frame)
        if missing:
            raise ValueError(f"Faltan features de inferencia: {sorted(missing)}")
        result = features.frame[[USER_COL, ITEM_COL]].copy()
        result["score"] = self.model_.predict(
            features.frame.loc[:, list(self.feature_columns_)],
            num_iteration=self.best_iteration_,
        )
        result = result.sort_values(
            [USER_COL, "score", ITEM_COL],
            ascending=[True, False, True],
            kind="mergesort",
        )
        result["rank"] = result.groupby(USER_COL).cumcount() + 1
        return result.reset_index(drop=True)

    def feature_importance(self) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("El ranker todavía no fue entrenado")
        return pd.DataFrame(
            {
                "feature": self.feature_columns_,
                "importance_gain": self.model_.booster_.feature_importance(
                    importance_type="gain"
                ),
            }
        ).sort_values("importance_gain", ascending=False)


def tune_lambdarank(
    train: FeatureSet,
    validation: FeatureSet,
    validation_truth: pd.DataFrame,
    *,
    base_config: RankerConfig | None = None,
    n_trials: int = 30,
    random_state: int = 42,
) -> tuple[dict[str, Any], object]:
    """Ajusta LambdaRank con Optuna usando NDCG@20 end-to-end."""

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("El tuning requiere `optuna`") from exc

    config = base_config or RankerConfig()

    def objective(trial: Any) -> float:
        parameters = {
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.015, 0.12, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 10, 100
            ),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 20.0, log=True
            ),
        }
        ranker = LambdaRanker(
            config,
            random_state=random_state,
            overrides=parameters,
        ).fit(train, validation)
        recommendations = ranker.predict(validation)
        return ndcg_at_k(recommendations, validation_truth, k=20)

    from .tracking import optuna_callback

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[optuna_callback("lambdarank")],
    )
    best = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in study.best_params.items()
    }
    return best, study
