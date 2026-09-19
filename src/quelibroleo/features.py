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
    "item_mean_rating",
    "item_count",
    "edad",
    "user_genero_mean",
    "user_autor_mean",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


class FeatureBuilder:
    """Estadísticas de usuario/ítem calculadas solo en train. Sin imputación global."""

    def __init__(self) -> None:
        """Inicializa estadísticas vacías; ``fit`` las completa con train."""
        self.ref_year: int = 2020
        self.item_stats: pd.DataFrame = pd.DataFrame()
        self.user_genero: pd.DataFrame = pd.DataFrame()
        self.user_autor: pd.DataFrame = pd.DataFrame()

    def fit(self, df_train: pd.DataFrame) -> FeatureBuilder:
        """Aprende stats de ítem y afinidad usuario-género/autor solo en train."""
        self.ref_year = int(df_train["fecha"].dt.year.max())

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
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        date_col: str = "fecha",
        ref_year: int | None = None,
    ) -> pd.DataFrame:
        """Adjunta stats de train y calcula ``edad`` (año de ``fecha`` o ``ref_year``)."""
        res = df.merge(self.item_stats, on="id_libro", how="left")
        res = res.merge(self.user_genero, on=["id_lector", "genero"], how="left")
        res = res.merge(self.user_autor, on=["id_lector", "autor"], how="left")

        year = ref_year if ref_year is not None else self.ref_year
        if date_col in res.columns:
            year_series = res[date_col].dt.year
            res["edad"] = year_series - res["nacimiento"]
        else:
            res["edad"] = year - res["nacimiento"]

        res["edad"] = pd.to_numeric(res["edad"], errors="coerce")
        return res

    def user_frame(self, lectores: pd.DataFrame) -> pd.DataFrame:
        """Perfil de lector para el producto cartesiano de ranking."""
        cols = [c for c in ["id_lector", "sexo", "nacimiento", "vive_en_ciudad", "vive_en_pais"] if c in lectores.columns]
        return lectores[cols].drop_duplicates("id_lector")

    def item_frame(self, libros: pd.DataFrame) -> pd.DataFrame:
        """Catálogo de libros con stats de ítem aprendidas en train."""
        cols = [c for c in ["id_libro", "autor", "genero", "editorial", "anio_edicion"] if c in libros.columns]
        items = libros[cols].drop_duplicates("id_libro")
        return items.merge(self.item_stats, on="id_libro", how="left")

    def attach_affinities(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Pega medias de rating usuario-género y usuario-autor a pares lector-libro."""
        res = pairs.merge(self.user_genero, on=["id_lector", "genero"], how="left")
        return res.merge(self.user_autor, on=["id_lector", "autor"], how="left")
