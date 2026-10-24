"""Ranking final con LightGBM LambdaRank."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import lightgbm as lgb
import pandas as pd

from ..config import LABEL_GAIN_MODES, RankerConfig
from ..data import ITEM_COL, USER_COL
from .columns import RankerTable

LABEL_GAINS: dict[str, tuple[int, ...]] = {
    "linear": tuple(range(11)),
    "exponential": tuple(2**rating - 1 for rating in range(11)),
}


def label_gain_values(mode: str) -> list[int]:
    """Ganancia por rating. `exponential` coincide con el NDCG de evaluación."""

    if mode not in LABEL_GAIN_MODES:
        raise ValueError("label_gain debe ser 'linear' o 'exponential'")
    return list(LABEL_GAINS[mode])


def _ordered_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Ordena por query y devuelve el tamaño de cada grupo para LightGBM."""

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
        """Parámetros de LGBMRanker, con la ganancia de etiqueta ya resuelta."""

        values = asdict(self.config)
        values.pop("early_stopping_rounds")
        values.pop("search")
        values.update(self.overrides)
        gain = label_gain_values(str(values.pop("label_gain")))
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": gain,
            "lambdarank_truncation_level": 20,
            "verbosity": -1,
            "random_state": self.random_state,
            "n_jobs": -1,
            **values,
        }

    def fit(
        self,
        train: RankerTable,
        validation: RankerTable | None = None,
    ) -> LambdaRanker:
        """Entrena por grupo. Si hay validación, corta con early stopping."""

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

    def predict(self, features: RankerTable) -> pd.DataFrame:
        """Puntúa y ordena por score descendente, con desempate por id_libro."""

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
        """Importancia por ganancia del booster ya entrenado."""

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
