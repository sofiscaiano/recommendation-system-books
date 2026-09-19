from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

INTERACTIONS_QUERY = """
SELECT
    b.id_libro,
    b.titulo,
    b.autor,
    b.genero,
    b.editorial,
    b.anio_edicion,
    b.isbn,
    b.resumen,
    b.img_src,
    c.id_lector,
    c.nombre,
    c.genero AS sexo,
    c.vive_en,
    c.nacimiento,
    a.fecha,
    a.rating
FROM interacciones AS a
INNER JOIN libros AS b ON a.id_libro = b.id_libro
INNER JOIN lectores AS c ON a.id_lector = c.id_lector
"""

LIBROS_QUERY = """
SELECT
    id_libro,
    titulo,
    autor,
    genero,
    editorial,
    anio_edicion,
    isbn,
    resumen,
    img_src
FROM libros
"""

LECTORES_QUERY = """
SELECT
    id_lector,
    nombre,
    genero AS sexo,
    vive_en,
    nacimiento
FROM lectores
"""


def format_data(data: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Parsea ``fecha`` y convierte ``nacimiento`` / ``anio_edicion`` a entero nullable."""
    res = data if inplace else data.copy()
    if "fecha" in res.columns:
        res["fecha"] = pd.to_datetime(res["fecha"], format="%d-%m-%Y", errors="coerce")
    if "nacimiento" in res.columns:
        res["nacimiento"] = pd.to_numeric(res["nacimiento"], errors="coerce").astype("Int64")
    if "anio_edicion" in res.columns:
        res["anio_edicion"] = pd.to_numeric(res["anio_edicion"], errors="coerce").astype("Int64")
    return res


def extraer_ciudad_pais(
    df: pd.DataFrame,
    col_origen: str = "vive_en",
    col_ciudad: str = "vive_en_ciudad",
    col_pais: str = "vive_en_pais",
    inplace: bool = False,
) -> pd.DataFrame:
    """Separa ``vive_en`` en ciudad y país usando el primer ``-`` como corte."""
    res = df if inplace else df.copy()

    if res.empty or col_origen not in res.columns:
        res[col_ciudad] = None
        res[col_pais] = None
        return res

    texto_limpio = res[col_origen].fillna("").astype(str).str.strip()
    partes = texto_limpio.str.split("-", n=1, expand=True)
    if 1 not in partes.columns:
        partes[1] = None

    tiene_guion = partes[1].notna()
    ciudad = np.where(tiene_guion, partes[0].str.strip(), None)
    pais = np.where(tiene_guion, partes[1].str.strip(), partes[0].str.strip())
    res[col_ciudad] = np.where((ciudad == "") | (ciudad == "None"), None, ciudad)
    res[col_pais] = np.where((pais == "") | (pais == "None"), None, pais)
    return res


def _read_sql(db_path: str | Path, query: str) -> pd.DataFrame:
    """Ejecuta ``query`` contra SQLite y devuelve un DataFrame."""
    with sqlite3.connect(db_path) as conexion:
        return pd.read_sql_query(query, conexion)


def load_interactions(db_path: str | Path) -> pd.DataFrame:
    """Carga interacciones joinadas con libro y lector, ya formateadas."""
    df = _read_sql(db_path, INTERACTIONS_QUERY)
    df = format_data(df)
    return extraer_ciudad_pais(df)


def load_catalog(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve ``(libros, lectores)`` del catálogo, con tipos y ``vive_en`` parseados."""
    libros = format_data(_read_sql(db_path, LIBROS_QUERY))
    lectores = format_data(_read_sql(db_path, LECTORES_QUERY))
    lectores = extraer_ciudad_pais(lectores)
    return libros, lectores
