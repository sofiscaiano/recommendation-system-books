from __future__ import annotations

import pandas as pd

CATEGORICAL_FEATURES = [
    "autor",
    "genero",
    "editorial",
    "sexo",
    "vive_en_ciudad",
    "vive_en_pais",
]

NUMERIC_FEATURES = [
    "anio_edicion",
    "nacimiento",
    "user_mean_rating",
    "user_count",
    "item_mean_rating",
    "item_count",
    "edad",
    "user_genero_mean",
    "user_autor_mean",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


class FeatureBuilder:
    """Estadísticas de usuario/ítem calculadas solo en train."""

    def __init__(self) -> None:
        self.global_mean: float = 0.0
        self.ref_year: int = 2020
        self.user_stats: pd.DataFrame = pd.DataFrame()
        self.item_stats: pd.DataFrame = pd.DataFrame()
        self.user_genero: pd.DataFrame = pd.DataFrame()
        self.user_autor: pd.DataFrame = pd.DataFrame()
        self.genero_global: pd.DataFrame = pd.DataFrame()
        self.autor_global: pd.DataFrame = pd.DataFrame()

    def fit(self, df_train: pd.DataFrame) -> FeatureBuilder:
        self.global_mean = float(df_train["rating"].mean())
        self.ref_year = int(df_train["fecha"].dt.year.max())

        self.user_stats = (
            df_train.groupby("id_lector", as_index=False)
            .agg(user_mean_rating=("rating", "mean"), user_count=("rating", "size"))
        )
        self.item_stats = (
            df_train.groupby("id_libro", as_index=False)
            .agg(item_mean_rating=("rating", "mean"), item_count=("rating", "size"))
        )
        self.user_genero = (
            df_train.groupby(["id_lector", "genero"], as_index=False)
            .agg(user_genero_mean=("rating", "mean"))
        )
        self.user_autor = (
            df_train.groupby(["id_lector", "autor"], as_index=False)
            .agg(user_autor_mean=("rating", "mean"))
        )
        self.genero_global = (
            df_train.groupby("genero", as_index=False)
            .agg(genero_global_mean=("rating", "mean"))
        )
        self.autor_global = (
            df_train.groupby("autor", as_index=False)
            .agg(autor_global_mean=("rating", "mean"))
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        date_col: str = "fecha",
        ref_year: int | None = None,
    ) -> pd.DataFrame:
        res = df.merge(self.user_stats, on="id_lector", how="left")
        res = res.merge(self.item_stats, on="id_libro", how="left")
        res = res.merge(self.user_genero, on=["id_lector", "genero"], how="left")
        res = res.merge(self.user_autor, on=["id_lector", "autor"], how="left")
        res = res.merge(self.genero_global, on="genero", how="left")
        res = res.merge(self.autor_global, on="autor", how="left")

        res["user_mean_rating"] = res["user_mean_rating"].fillna(self.global_mean)
        res["user_count"] = res["user_count"].fillna(0)
        res["item_mean_rating"] = res["item_mean_rating"].fillna(self.global_mean)
        res["item_count"] = res["item_count"].fillna(0)
        res["user_genero_mean"] = res["user_genero_mean"].fillna(
            res["genero_global_mean"].fillna(self.global_mean)
        )
        res["user_autor_mean"] = res["user_autor_mean"].fillna(
            res["autor_global_mean"].fillna(self.global_mean)
        )

        year = ref_year if ref_year is not None else self.ref_year
        if date_col in res.columns:
            year_series = res[date_col].dt.year
            res["edad"] = year_series - res["nacimiento"]
        else:
            res["edad"] = year - res["nacimiento"]

        res["edad"] = pd.to_numeric(res["edad"], errors="coerce")
        return res.drop(columns=["genero_global_mean", "autor_global_mean"], errors="ignore")

    def user_frame(self, lectores: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["id_lector", "sexo", "nacimiento", "vive_en_ciudad", "vive_en_pais"] if c in lectores.columns]
        users = lectores[cols].drop_duplicates("id_lector")
        users = users.merge(self.user_stats, on="id_lector", how="left")
        users["user_mean_rating"] = users["user_mean_rating"].fillna(self.global_mean)
        users["user_count"] = users["user_count"].fillna(0)
        return users

    def item_frame(self, libros: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["id_libro", "autor", "genero", "editorial", "anio_edicion"] if c in libros.columns]
        items = libros[cols].drop_duplicates("id_libro")
        items = items.merge(self.item_stats, on="id_libro", how="left")
        items["item_mean_rating"] = items["item_mean_rating"].fillna(self.global_mean)
        items["item_count"] = items["item_count"].fillna(0)
        return items

    def attach_affinities(self, pairs: pd.DataFrame) -> pd.DataFrame:
        res = pairs.merge(self.user_genero, on=["id_lector", "genero"], how="left")
        res = res.merge(self.user_autor, on=["id_lector", "autor"], how="left")
        res = res.merge(self.genero_global, on="genero", how="left")
        res = res.merge(self.autor_global, on="autor", how="left")
        res["user_genero_mean"] = res["user_genero_mean"].fillna(
            res["genero_global_mean"].fillna(self.global_mean)
        )
        res["user_autor_mean"] = res["user_autor_mean"].fillna(
            res["autor_global_mean"].fillna(self.global_mean)
        )
        return res.drop(columns=["genero_global_mean", "autor_global_mean"], errors="ignore")
