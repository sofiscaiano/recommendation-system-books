from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from quelibroleo.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

DEFAULT_LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


class RatingModel:
    def __init__(self, **lgbm_params) -> None:
        params = {**DEFAULT_LGBM_PARAMS, **lgbm_params}
        self.estimator = LGBMRegressor(**params)
        self.category_maps: dict[str, pd.Index] = {}
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)

    def _prepare(self, df: pd.DataFrame, *, fit_categories: bool) -> pd.DataFrame:
        X = df[self.feature_columns].copy()
        for col in NUMERIC_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        if fit_categories:
            for col in CATEGORICAL_FEATURES:
                X[col] = X[col].astype("category")
            self.category_maps = {col: X[col].cat.categories for col in CATEGORICAL_FEATURES}
        else:
            for col in CATEGORICAL_FEATURES:
                categories = self.category_maps.get(col)
                X[col] = pd.Categorical(X[col], categories=categories)
        return X

    def fit(self, df: pd.DataFrame, target_col: str = "rating") -> RatingModel:
        X = self._prepare(df, fit_categories=True)
        self.estimator.fit(X, df[target_col])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = self._prepare(df, fit_categories=False)
        return pd.Series(self.estimator.predict(X), index=df.index)
