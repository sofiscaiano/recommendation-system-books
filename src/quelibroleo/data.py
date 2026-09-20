"""Carga y validación de los datos de Qué Libro Leo."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

USER_COL = "id_lector"
ITEM_COL = "id_libro"
DATE_COL = "fecha"
RATING_COL = "rating"


@dataclass(frozen=True)
class DataQualityReport:
    """Resumen de anomalías detectadas durante la carga."""

    interaction_count: int
    valid_interaction_count: int
    invalid_date_count: int
    orphan_user_count: int
    orphan_user_interactions: int
    orphan_item_count: int
    orphan_item_interactions: int
    catalog_items_without_interactions: int


@dataclass(frozen=True)
class DatasetBundle:
    """Tablas normalizadas y reporte de calidad."""

    interactions: pd.DataFrame
    invalid_interactions: pd.DataFrame
    books: pd.DataFrame
    readers: pd.DataFrame
    report: DataQualityReport

    @property
    def modeled_interactions(self) -> pd.DataFrame:
        """Interacciones con fecha válida y libro presente en catálogo."""

        valid_items = set(self.books[ITEM_COL])
        return self.interactions[
            self.interactions[ITEM_COL].isin(valid_items)
        ].reset_index(drop=True)


def _read_sqlite(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"No existe la base SQLite: {path}")

    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        interactions = pd.read_sql_query(
            "SELECT id_lector, id_libro, fecha, rating FROM interacciones",
            connection,
        )
        books = pd.read_sql_query("SELECT * FROM libros", connection)
        readers = pd.read_sql_query("SELECT * FROM lectores", connection)
    finally:
        connection.close()
    return interactions, books, readers


def load_dataset(path: str | Path = "data.db") -> DatasetBundle:
    """Carga ``data.db`` en modo de solo lectura y valida sus claves.

    La fila con fecha inválida queda separada y nunca participa en splits
    temporales. Las interacciones huérfanas se conservan en ``interactions``
    para poder auditarlas; ``modeled_interactions`` excluye libros que no
    existen en el catálogo.
    """

    db_path = Path(path)
    raw, books, readers = _read_sqlite(db_path)

    parsed_dates = pd.to_datetime(raw[DATE_COL], format="%d-%m-%Y", errors="coerce")
    invalid_mask = parsed_dates.isna()
    invalid = raw.loc[invalid_mask].copy()
    interactions = raw.loc[~invalid_mask].copy()
    interactions[DATE_COL] = parsed_dates.loc[~invalid_mask]
    interactions[RATING_COL] = pd.to_numeric(
        interactions[RATING_COL], errors="raise"
    ).astype("int8")

    if not interactions[RATING_COL].between(1, 10).all():
        raise ValueError("Se encontraron ratings fuera del rango 1..10")
    if interactions.duplicated([USER_COL, ITEM_COL]).any():
        raise ValueError("La base contiene pares lector-libro duplicados")

    catalog_users = set(readers[USER_COL])
    catalog_items = set(books[ITEM_COL])
    interaction_users = set(interactions[USER_COL])
    interaction_items = set(interactions[ITEM_COL])
    orphan_users = interaction_users - catalog_users
    orphan_items = interaction_items - catalog_items

    report = DataQualityReport(
        interaction_count=len(raw),
        valid_interaction_count=len(interactions),
        invalid_date_count=int(invalid_mask.sum()),
        orphan_user_count=len(orphan_users),
        orphan_user_interactions=int(
            interactions[USER_COL].isin(orphan_users).sum()
        ),
        orphan_item_count=len(orphan_items),
        orphan_item_interactions=int(
            interactions[ITEM_COL].isin(orphan_items).sum()
        ),
        catalog_items_without_interactions=len(catalog_items - interaction_items),
    )

    books = books.copy()
    for column in ("titulo", "autor", "genero", "editorial", "resumen"):
        books[column] = books[column].fillna("").astype(str).str.strip()
    books["anio_edicion_num"] = pd.to_numeric(
        books["anio_edicion"], errors="coerce"
    )
    books.loc[
        ~books["anio_edicion_num"].between(1000, 2100), "anio_edicion_num"
    ] = pd.NA

    return DatasetBundle(
        interactions=interactions.reset_index(drop=True),
        invalid_interactions=invalid.reset_index(drop=True),
        books=books.reset_index(drop=True),
        readers=readers.reset_index(drop=True),
        report=report,
    )


def observed_catalog(
    interactions: pd.DataFrame,
    books: pd.DataFrame,
) -> pd.DataFrame:
    """Restringe el catálogo a libros observados en el train indicado."""

    observed = interactions[[ITEM_COL]].drop_duplicates()
    return observed.merge(books, on=ITEM_COL, how="inner", validate="one_to_one")
