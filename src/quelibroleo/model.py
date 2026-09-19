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
    """Regresor LightGBM de rating sobre las columnas de ``FEATURE_COLUMNS``."""

    def __init__(self, **lgbm_params) -> None:
        """Crea el estimador; ``lgbm_params`` pisa ``DEFAULT_LGBM_PARAMS``."""
        params = {**DEFAULT_LGBM_PARAMS, **lgbm_params}
        self.estimator = LGBMRegressor(**params)
        self.category_maps: dict[str, pd.Index] = {}
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)

    def _prepare(self, df: pd.DataFrame, *, fit_categories: bool) -> pd.DataFrame:
        """Convierte numéricos y categorías; en fit guarda el mapa de niveles."""
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
        """Entrena el regresor sobre ``df`` usando ``target_col`` como etiqueta."""
        X = self._prepare(df, fit_categories=True)
        self.estimator.fit(X, df[target_col])
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predice ratings alineados al índice de ``df``."""
        X = self._prepare(df, fit_categories=False)
        return pd.Series(self.estimator.predict(X), index=df.index)

    def feature_importance(self) -> pd.DataFrame:
        """Importancia LightGBM: ``gain`` (reducción de pérdida) y ``split`` (usos)."""
        booster = self.estimator.booster_
        table = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "gain": booster.feature_importance(importance_type="gain"),
                "split": booster.feature_importance(importance_type="split"),
            }
        )
        total_gain = float(table["gain"].sum())
        table["gain_pct"] = 0.0 if total_gain == 0 else 100.0 * table["gain"] / total_gain
        return table.sort_values("gain", ascending=False).reset_index(drop=True)
