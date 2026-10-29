"""Dónde falla el hold-out: por tipo de lector, y si fue retrieval o ranking.

Lee el top-20 de ``artifacts/evaluation_recommendations.csv`` y, si la
evaluación lo guardó, ``evaluation_retrieval_hits.csv`` (1 si el libro de test
entró a candidatos). No reentrena nada.

``pierde_retrieval`` es ``1 - oracle``: NDCG que no existe porque el libro no
entró a candidatos. ``pierde_ranking`` es ``oracle - ndcg``: NDCG que el ranker
dejó pasar teniendo el libro. Las dos suman ``1 - ndcg``.

Uso, desde la raíz del repo::

    uv run python diagnostico.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quelibroleo.config import PipelineConfig
from quelibroleo.data import ITEM_COL, RATING_COL, USER_COL
from quelibroleo.metrics import ndcg_per_user
from quelibroleo.split_data import load_split

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
POBLACION = ("ejemplo.csv", "resto del hold-out")
AUTORES = ("no repite autores", "repite algunos", "repite la mayoría")


def user_metrics(
    recommendations: pd.DataFrame,
    test: pd.DataFrame,
    history: pd.DataFrame,
    books: pd.DataFrame,
    ejemplo: set[str],
    hits: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Una fila por lector del test. Sin ``hits`` no hay columnas de retrieval."""

    test = _keys(test)
    top = _keys(recommendations)
    top = top[top["rank"] <= 20] if "rank" in top else top.groupby(USER_COL, sort=False).head(20)
    users = pd.Index(sorted(test[USER_COL].unique()), name=USER_COL)
    ndcg = ndcg_per_user(top, test).reindex(users, fill_value=0.0)

    top_keys = top[[USER_COL, ITEM_COL]].drop_duplicates()
    top_keys["en_top"] = 1
    marked = test.merge(top_keys, on=[USER_COL, ITEM_COL], how="left")
    marked["en_top"] = marked["en_top"].fillna(0.0)
    en_top = marked.groupby(USER_COL)["en_top"].sum().reindex(users, fill_value=0.0)

    recall, oracle, retrieved = _retrieval_columns(test, hits, users)
    frame = pd.DataFrame(
        {
            USER_COL: users.astype(str),
            "ndcg": ndcg.to_numpy(dtype=float),
            "recall": recall,
            "oracle": oracle,
            "en_candidatos": retrieved,
            "en_top": en_top.to_numpy(dtype=float),
            "n_lecturas": _history_size(history).reindex(users, fill_value=0).to_numpy(dtype=float),
            "fraccion_autor": _author_repeat(history, test, books).reindex(users, fill_value=0.0).to_numpy(),
        }
    )
    frame["poblacion"] = np.where(frame[USER_COL].isin(ejemplo), POBLACION[0], POBLACION[1])
    frame["historial"] = _history_band(frame["n_lecturas"])
    frame["autores"] = _author_band(frame["fraccion_autor"])
    return frame


def summarize(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Promedios del grupo. Con retrieval, las dos pérdidas suman ``1 - ndcg``."""

    has_retrieval = frame["oracle"].notna().all()
    rows = []
    for name, part in frame.groupby(column, sort=False, observed=True):
        ndcg = float(part["ndcg"].mean())
        row = {
            "grupo": str(name),
            "usuarios": len(part),
            "ndcg": ndcg,
            "en_top": float(part["en_top"].mean()),
        }
        if has_retrieval:
            oracle = float(part["oracle"].mean())
            row.update(
                {
                    "en_candidatos": float(part["en_candidatos"].mean()),
                    "recall": float(part["recall"].mean()),
                    "oracle": oracle,
                    "pierde_retrieval": 1.0 - oracle,
                    "pierde_ranking": oracle - ndcg,
                }
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    if has_retrieval:
        table["falla"] = np.where(
            table["pierde_ranking"] >= table["pierde_retrieval"], "ranking", "retrieval"
        )
    return table


def render(frame: pd.DataFrame, *, run_name: str, has_retrieval: bool) -> str:
    """Texto corto: el promedio y tres cortes de lector."""

    ndcg = float(frame["ndcg"].mean())
    placed = float(frame["en_top"].mean())
    lines = [
        (
            f"Hold-out: {len(frame)} lectores. NDCG@20 {ndcg:.3f}. "
            f"{placed:.1f} de los 20 libros del test quedan en el top-20."
        )
    ]
    if has_retrieval:
        oracle = float(frame["oracle"].mean())
        recall = float(frame["recall"].mean())
        retrieved = float(frame["en_candidatos"].mean())
        lost_retrieval = 1.0 - oracle
        lost_ranking = oracle - ndcg
        dominant = "ranking" if lost_ranking >= lost_retrieval else "retrieval"
        lines.append(
            f"Recall {recall:.3f}: {retrieved:.1f} de esos 20 entran a candidatos. "
            f"Oracle {oracle:.3f}."
        )
        lines.append(
            f"De los {1 - ndcg:.3f} que faltan para 1, {lost_retrieval:.3f} se pierden en retrieval "
            f"y {lost_ranking:.3f} en ranking. Domina el {dominant}."
        )
    else:
        lines.append(
            "Esta evaluación no guardó si cada libro entró al retrieval. "
            "El próximo evaluate escribe artifacts/evaluation_retrieval_hits.csv."
        )
    lines.append("")
    cuts = (
        ("población", "poblacion", list(POBLACION)),
        ("historial antes del hold-out", "historial", _history_order(frame["historial"])),
        ("autores del test ya leídos", "autores", list(AUTORES)),
    )
    for title, column, order in cuts:
        table = _ordered(summarize(frame, column), order)
        lines.append(title)
        lines.append(_format(table))
        lines.append(_worst(table, has_retrieval=has_retrieval))
        lines.append("")
    lines.append(f"Corrida {run_name or 'sin nombre'}.")
    return "\n".join(lines).rstrip()


def main() -> None:
    """Imprime el diagnóstico de la última evaluación guardada."""

    recommendations = ARTIFACTS / "evaluation_recommendations.csv"
    if not recommendations.is_file():
        raise SystemExit("Falta artifacts/evaluation_recommendations.csv. Corré evaluate antes.")
    config = PipelineConfig(
        db_path=ROOT / "data.db",
        example_path=ROOT / "ejemplo.csv",
        artifacts_dir=ARTIFACTS,
    )
    split = load_split(config, holdout=True)
    hits_path = ARTIFACTS / "evaluation_retrieval_hits.csv"
    hits = _hits_for_run(hits_path, _evaluation_run_name())
    ejemplo = set(pd.read_csv(config.example_path, dtype=str)[USER_COL])
    frame = user_metrics(
        pd.read_csv(recommendations),
        split.test,
        split.history,
        split.books,
        ejemplo,
        hits,
    )
    print(render(frame, run_name=_evaluation_run_name(), has_retrieval=hits is not None))


def _retrieval_columns(
    test: pd.DataFrame,
    hits: pd.DataFrame | None,
    users: pd.Index,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recall, oracle y cantidad recuperada. NaN si no hay archivo de hits."""

    empty = np.full(len(users), np.nan)
    if hits is None:
        return empty, empty.copy(), empty.copy()
    flagged = _keys(hits)[[USER_COL, ITEM_COL, "in_retrieval"]]
    flagged["in_retrieval"] = flagged["in_retrieval"].astype(int)
    marked = test.merge(flagged, on=[USER_COL, ITEM_COL], how="left")
    marked["in_retrieval"] = marked["in_retrieval"].fillna(0).astype(int)
    size = marked.groupby(USER_COL).size().reindex(users, fill_value=0)
    retrieved = marked.groupby(USER_COL)["in_retrieval"].sum().reindex(users, fill_value=0.0)
    recall = np.divide(retrieved, size.where(size > 0, np.nan)).fillna(0.0)
    found = marked[marked["in_retrieval"] == 1].sort_values(
        [USER_COL, RATING_COL, ITEM_COL], ascending=[True, False, True]
    )
    if found.empty:
        oracle_rows = pd.DataFrame(columns=[USER_COL, ITEM_COL, "rank"])
    else:
        oracle_rows = found[[USER_COL, ITEM_COL]].copy()
        oracle_rows["rank"] = oracle_rows.groupby(USER_COL).cumcount() + 1
    oracle = ndcg_per_user(oracle_rows, test).reindex(users, fill_value=0.0)
    return recall.to_numpy(dtype=float), oracle.to_numpy(dtype=float), retrieved.to_numpy(dtype=float)


def _hits_for_run(path: Path, run_name: str) -> pd.DataFrame | None:
    """Lee los hits solo si son de la misma corrida que el top-20."""

    if not path.is_file():
        return None
    hits = pd.read_csv(path)
    if "run_name" in hits.columns and run_name and str(hits["run_name"].iloc[0]) != run_name:
        return None
    return hits


def _keys(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied[USER_COL] = copied[USER_COL].astype(str)
    copied[ITEM_COL] = copied[ITEM_COL].astype(str)
    return copied


def _history_size(history: pd.DataFrame) -> pd.Series:
    sized = history.copy()
    sized[USER_COL] = sized[USER_COL].astype(str)
    return sized.groupby(USER_COL).size()


def _author_repeat(history: pd.DataFrame, test: pd.DataFrame, books: pd.DataFrame) -> pd.Series:
    """Fracción de libros del test cuyo autor ya apareció en la historia."""

    authors = books[[ITEM_COL, "autor"]].copy()
    authors[ITEM_COL] = authors[ITEM_COL].astype(str)
    authors["autor"] = authors["autor"].fillna("").astype(str).str.strip()
    known = _keys(history)[[USER_COL, ITEM_COL]].merge(authors, on=ITEM_COL, how="left")
    known = known.loc[known["autor"].ne(""), [USER_COL, "autor"]].drop_duplicates()
    known["visto"] = 1.0
    named = _keys(test)[[USER_COL, ITEM_COL]].merge(authors, on=ITEM_COL, how="left")
    named = named.loc[named["autor"].ne("")]
    named = named.merge(known, on=[USER_COL, "autor"], how="left")
    named["visto"] = named["visto"].fillna(0.0)
    if named.empty:
        return pd.Series(dtype=float)
    return named.groupby(USER_COL)["visto"].mean()


def _history_band(counts: pd.Series) -> pd.Series:
    """Cuartiles etiquetados con el mínimo y el máximo de lecturas del grupo."""

    try:
        bands = pd.qcut(counts, 4, duplicates="drop")
    except ValueError:
        label = f"{int(counts.min())}-{int(counts.max())} lecturas"
        return pd.Series(label, index=counts.index)
    names = {
        interval: f"{int(part.min())}-{int(part.max())} lecturas"
        for interval, part in counts.groupby(bands, observed=True)
    }
    return bands.map(names).astype(str)


def _author_band(fraction: pd.Series) -> pd.Series:
    band = pd.Series(AUTORES[1], index=fraction.index)
    band = band.mask(fraction <= 0, AUTORES[0])
    return band.mask(fraction >= 0.5, AUTORES[2])


def _history_order(labels: pd.Series) -> list[str]:
    return sorted(labels.unique(), key=lambda label: int(str(label).split("-", 1)[0]))


def _ordered(table: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    present = [name for name in order if name in set(table["grupo"])]
    extra = [name for name in table["grupo"] if name not in present]
    ranking = {name: index for index, name in enumerate(present + extra)}
    return table.sort_values("grupo", key=lambda column: column.map(ranking))


def _format(table: pd.DataFrame) -> str:
    show = table.copy()
    for column in ("ndcg", "recall", "oracle", "pierde_retrieval", "pierde_ranking"):
        if column in show:
            show[column] = show[column].map(lambda value: f"{value:.3f}")
    for column in ("en_candidatos", "en_top"):
        if column in show:
            show[column] = show[column].map(lambda value: f"{value:.1f}")
    return show.rename(
        columns={
            "en_candidatos": "de20_candidatos",
            "en_top": "de20_top",
            "pierde_retrieval": "pierde_retr",
            "pierde_ranking": "pierde_rank",
        }
    ).to_string(index=False)


def _worst(table: pd.DataFrame, *, has_retrieval: bool) -> str:
    row = table.loc[table["ndcg"].idxmin()]
    text = f"Peor: {row['grupo']} ({int(row['usuarios'])} lectores, ndcg {row['ndcg']:.3f})."
    if has_retrieval:
        text += (
            f" Pierde {row['pierde_retrieval']:.3f} en retrieval"
            f" y {row['pierde_ranking']:.3f} en ranking."
        )
    return text


def _evaluation_run_name() -> str:
    path = ARTIFACTS / "evaluation.json"
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("run_name", ""))


if __name__ == "__main__":
    main()
