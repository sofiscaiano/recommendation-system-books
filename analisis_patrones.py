"""Análisis de lealtad autor/género/editorial: hold-out vs submission.

Correr: uv run python analisis_patrones.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quelibroleo.data import load_catalog, load_interactions
from quelibroleo.split import leave_last_k

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analisis_patrones_resultados.csv"


def _sets(df: pd.DataFrame, user_col: str, value_col: str) -> pd.Series:
    return df.dropna(subset=[value_col]).groupby(user_col)[value_col].agg(set)


def _hit_rate(test: pd.DataFrame, train_sets: pd.Series, col: str, user_col: str = "id_lector") -> pd.Series:
    mapped = test[user_col].map(train_sets)
    present = mapped.notna()
    vals = test[col]
    hits = pd.Series(False, index=test.index)
    hits.loc[present] = [
        (pd.notna(v) and v in s) for v, s in zip(vals[present], mapped[present], strict=True)
    ]
    return hits


def _user_rates(test: pd.DataFrame, hits: pd.Series, user_col: str = "id_lector") -> pd.Series:
    tmp = test[[user_col]].copy()
    tmp["_hit"] = hits.to_numpy()
    return tmp.groupby(user_col)["_hit"].mean()


def _hist_bins(rates: pd.Series, edges: list[float]) -> list[tuple[str, int]]:
    cats = pd.cut(rates, bins=edges, include_lowest=True, right=True)
    counts = cats.value_counts(sort=False)
    out = []
    for interval, n in counts.items():
        out.append((str(interval), int(n)))
    return out


def _top_share(df: pd.DataFrame, col: str, n: int = 8) -> pd.DataFrame:
    vc = df[col].fillna("(vacío)").value_counts(normalize=True).head(n)
    return pd.DataFrame({"valor": vc.index.astype(str), "share": vc.to_numpy()})


def main() -> None:
    print("Cargando...")
    df = load_interactions(ROOT / "data.db")
    libros, _lectores = load_catalog(ROOT / "data.db")
    train, test = leave_last_k(df, k=20)
    print(f"Train {train.shape}  Test {test.shape}  usuarios eval {test['id_lector'].nunique()}")

    dims = {
        "autor": "autor",
        "genero": "genero",
        "editorial": "editorial",
        "id_libro": "id_libro",
    }

    rows_summary = []
    user_rate_tables = {}
    hist_payload = {}

    for name, col in dims.items():
        tsets = _sets(train, "id_lector", col)
        hits = _hit_rate(test, tsets, col)
        urates = _user_rates(test, hits)
        micro = float(hits.mean())
        macro = float(urates.mean())
        median = float(urates.median())
        p10, p90 = float(urates.quantile(0.10)), float(urates.quantile(0.90))
        users_all = float((urates == 1).mean())
        users_none = float((urates == 0).mean())
        print(
            f"Test ∩ train {name:10s}  micro={micro:.3f}  macro={macro:.3f}  "
            f"med={median:.3f}  p10={p10:.3f}  p90={p90:.3f}  "
            f"%users_all={users_all:.3f}  %users_none={users_none:.3f}"
        )
        rows_summary.append(
            {
                "fase": "holdout_test_vs_train",
                "dimension": name,
                "micro": micro,
                "macro": macro,
                "mediana": median,
                "p10": p10,
                "p90": p90,
                "pct_usuarios_todo": users_all,
                "pct_usuarios_nada": users_none,
            }
        )
        user_rate_tables[name] = urates
        hist_payload[name] = _hist_bins(urates, [0, 0.2, 0.4, 0.6, 0.8, 1.01])

    # Concentración: top-1 género/autor de train cubre qué % del test
    train_top_genero = (
        train.groupby(["id_lector", "genero"]).size().reset_index(name="n")
        .sort_values(["id_lector", "n"], ascending=[True, False])
        .groupby("id_lector", sort=False)
        .head(1)
        .set_index("id_lector")["genero"]
    )
    test_top_match = test["genero"] == test["id_lector"].map(train_top_genero)
    print(f"Test en el género #1 del train (micro): {float(test_top_match.mean()):.3f}")
    rows_summary.append(
        {
            "fase": "holdout_test_vs_train",
            "dimension": "genero_top1_train",
            "micro": float(test_top_match.mean()),
            "macro": float(_user_rates(test, test_top_match).mean()),
            "mediana": float(_user_rates(test, test_top_match).median()),
            "p10": float(_user_rates(test, test_top_match).quantile(0.10)),
            "p90": float(_user_rates(test, test_top_match).quantile(0.90)),
            "pct_usuarios_todo": float((_user_rates(test, test_top_match) == 1).mean()),
            "pct_usuarios_nada": float((_user_rates(test, test_top_match) == 0).mean()),
        }
    )

    train_top_autor = (
        train.groupby(["id_lector", "autor"]).size().reset_index(name="n")
        .sort_values(["id_lector", "n"], ascending=[True, False])
        .groupby("id_lector", sort=False)
        .head(1)
        .set_index("id_lector")["autor"]
    )
    test_top_autor_match = test["autor"] == test["id_lector"].map(train_top_autor)
    print(f"Test en el autor #1 del train (micro): {float(test_top_autor_match.mean()):.3f}")
    rows_summary.append(
        {
            "fase": "holdout_test_vs_train",
            "dimension": "autor_top1_train",
            "micro": float(test_top_autor_match.mean()),
            "macro": float(_user_rates(test, test_top_autor_match).mean()),
            "mediana": float(_user_rates(test, test_top_autor_match).median()),
            "p10": float(_user_rates(test, test_top_autor_match).quantile(0.10)),
            "p90": float(_user_rates(test, test_top_autor_match).quantile(0.90)),
            "pct_usuarios_todo": float((_user_rates(test, test_top_autor_match) == 1).mean()),
            "pct_usuarios_nada": float((_user_rates(test, test_top_autor_match) == 0).mean()),
        }
    )

    # Baseline de azar: un libro aleatorio del catálogo interactuado vs sets de train
    pool = df[["id_libro", "autor", "genero", "editorial"]].drop_duplicates("id_libro")
    rng = np.random.default_rng(42)
    sample_idx = rng.integers(0, len(pool), size=len(test))
    fake = test[["id_lector"]].copy()
    for c in ["id_libro", "autor", "genero", "editorial"]:
        fake[c] = pool.iloc[sample_idx][c].to_numpy()
    print("Baseline azar (sample catálogo interactuado, mismo n que test):")
    for name, col in dims.items():
        tsets = _sets(train, "id_lector", col)
        hits = _hit_rate(fake, tsets, col)
        print(f"  {name:10s} micro={float(hits.mean()):.3f}  macro={float(_user_rates(fake, hits).mean()):.3f}")
        rows_summary.append(
            {
                "fase": "azar_catalogo_interactuado",
                "dimension": name,
                "micro": float(hits.mean()),
                "macro": float(_user_rates(fake, hits).mean()),
                "mediana": float(_user_rates(fake, hits).median()),
                "p10": float(_user_rates(fake, hits).quantile(0.10)),
                "p90": float(_user_rates(fake, hits).quantile(0.90)),
                "pct_usuarios_todo": float((_user_rates(fake, hits) == 1).mean()),
                "pct_usuarios_nada": float((_user_rates(fake, hits) == 0).mean()),
            }
        )

    # Diversidad train vs test
    def nunique_per_user(part: pd.DataFrame, col: str) -> pd.Series:
        return part.groupby("id_lector")[col].nunique()

    for col in ["autor", "genero", "editorial"]:
        nt, nte = nunique_per_user(train, col), nunique_per_user(test, col)
        common = nt.index.intersection(nte.index)
        print(
            f"Únicos {col:10s}  train med={float(nt.median()):.1f}  "
            f"test med={float(nte.median()):.1f}  (test tiene 20 filas fijas)"
        )

    # --- Submission ---
    sub = pd.read_csv(ROOT / "submission.csv")
    ejemplo = pd.read_csv(ROOT / "ejemplo.csv")
    hist_users = set(df["id_lector"].unique())
    sub_users = sub["id_lector"].drop_duplicates()
    n_cold = int((~sub_users.isin(hist_users)).sum())
    print(f"Submission: {sub['id_lector'].nunique()} lectores, {n_cold} sin historial (relleno ejemplo)")

    sub_join = sub.merge(
        libros[["id_libro", "autor", "genero", "editorial"]],
        on="id_libro",
        how="left",
    )
    miss_cat = float(sub_join["autor"].isna().mean())
    print(f"Filas submission sin match en catálogo: {miss_cat:.4f}")

    # Usar historial completo (como en inferencia final)
    full_sets = {name: _sets(df, "id_lector", col) for name, col in dims.items()}
    sub_hist = sub_join[sub_join["id_lector"].isin(hist_users)].copy()

    for name, col in dims.items():
        hits = _hit_rate(sub_hist, full_sets[name], col)
        urates = _user_rates(sub_hist, hits)
        print(
            f"Sub ∩ historial {name:10s}  micro={float(hits.mean()):.3f}  "
            f"macro={float(urates.mean()):.3f}  med={float(urates.median()):.3f}  "
            f"%users_all={float((urates == 1).mean()):.3f}  %users_none={float((urates == 0).mean()):.3f}"
        )
        rows_summary.append(
            {
                "fase": "submission_vs_historial",
                "dimension": name,
                "micro": float(hits.mean()),
                "macro": float(urates.mean()),
                "mediana": float(urates.median()),
                "p10": float(urates.quantile(0.10)),
                "p90": float(urates.quantile(0.90)),
                "pct_usuarios_todo": float((urates == 1).mean()),
                "pct_usuarios_nada": float((urates == 0).mean()),
            }
        )
        hist_payload[f"sub_{name}"] = _hist_bins(urates, [0, 0.2, 0.4, 0.6, 0.8, 1.01])

    sub_top_g = sub_hist["genero"] == sub_hist["id_lector"].map(
        df.groupby(["id_lector", "genero"]).size().reset_index(name="n")
        .sort_values(["id_lector", "n"], ascending=[True, False])
        .groupby("id_lector", sort=False)
        .head(1)
        .set_index("id_lector")["genero"]
    )
    print(f"Sub en género #1 historial (micro): {float(sub_top_g.mean()):.3f}")
    rows_summary.append(
        {
            "fase": "submission_vs_historial",
            "dimension": "genero_top1_hist",
            "micro": float(sub_top_g.mean()),
            "macro": float(_user_rates(sub_hist, sub_top_g).mean()),
            "mediana": float(_user_rates(sub_hist, sub_top_g).median()),
            "p10": float(_user_rates(sub_hist, sub_top_g).quantile(0.10)),
            "p90": float(_user_rates(sub_hist, sub_top_g).quantile(0.90)),
            "pct_usuarios_todo": float((_user_rates(sub_hist, sub_top_g) == 1).mean()),
            "pct_usuarios_nada": float((_user_rates(sub_hist, sub_top_g) == 0).mean()),
        }
    )

    sub_top_a = sub_hist["autor"] == sub_hist["id_lector"].map(
        df.groupby(["id_lector", "autor"]).size().reset_index(name="n")
        .sort_values(["id_lector", "n"], ascending=[True, False])
        .groupby("id_lector", sort=False)
        .head(1)
        .set_index("id_lector")["autor"]
    )
    print(f"Sub en autor #1 historial (micro): {float(sub_top_a.mean()):.3f}")
    rows_summary.append(
        {
            "fase": "submission_vs_historial",
            "dimension": "autor_top1_hist",
            "micro": float(sub_top_a.mean()),
            "macro": float(_user_rates(sub_hist, sub_top_a).mean()),
            "mediana": float(_user_rates(sub_hist, sub_top_a).median()),
            "p10": float(_user_rates(sub_hist, sub_top_a).quantile(0.10)),
            "p90": float(_user_rates(sub_hist, sub_top_a).quantile(0.90)),
            "pct_usuarios_todo": float((_user_rates(sub_hist, sub_top_a) == 1).mean()),
            "pct_usuarios_nada": float((_user_rates(sub_hist, sub_top_a) == 0).mean()),
        }
    )

    # Géneros: distribución global historial vs test vs submission
    print("\nTop géneros (share):")
    for label, part in [
        ("train", train),
        ("test", test),
        ("historial_completo", df),
        ("submission", sub_hist),
        ("ejemplo_cold", ejemplo.merge(libros[["id_libro", "genero"]], on="id_libro", how="left")),
    ]:
        print(f"  {label}")
        print(_top_share(part, "genero", 6).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Popularidad de ítems recomendados vs test
    pop = df.groupby("id_libro").size().rename("n_inter")
    test_pop = test["id_libro"].map(pop)
    sub_pop = sub_hist["id_libro"].map(pop)
    print(
        f"\nPopularidad (n interacciones del libro)  "
        f"test med={float(test_pop.median()):.0f} p90={float(test_pop.quantile(0.9)):.0f}  "
        f"sub med={float(sub_pop.median()):.0f} p90={float(sub_pop.quantile(0.9)):.0f}"
    )

    # Libros más recomendados
    top_rec = (
        sub.merge(libros[["id_libro", "titulo", "autor", "genero"]], on="id_libro", how="left")
        .groupby(["id_libro", "titulo", "autor", "genero"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(15)
        .reset_index(name="n_recs")
    )
    print("\nLibros más recomendados en submission:")
    print(top_rec.to_string(index=False))

    # Diversidad en ranking: autores únicos en las 20 recs
    n_aut_sub = sub_hist.groupby("id_lector")["autor"].nunique()
    n_gen_sub = sub_hist.groupby("id_lector")["genero"].nunique()
    n_aut_hist = df[df["id_lector"].isin(sub_hist["id_lector"])].groupby("id_lector")["autor"].nunique()
    print(
        f"\nAutores únicos /20 recs: med={float(n_aut_sub.median()):.1f}  "
        f"géneros únicos /20 recs: med={float(n_gen_sub.median()):.1f}  "
        f"autores únicos en historial (mismos users): med={float(n_aut_hist.median()):.1f}"
    )

    # ¿El modelo recomienda siempre los mismos libros?
    print(f"Libros distintos en submission: {sub['id_libro'].nunique()} / {len(sub)} filas")
    rec_counts = sub["id_libro"].value_counts()
    print(
        f"Libros que cubren 50% de las recs: "
        f"{int((rec_counts.cumsum() / rec_counts.sum() <= 0.5).sum() + 1)}"
    )

    summary = pd.DataFrame(rows_summary)
    summary.to_csv(OUT, index=False)
    print(f"\nEscrito {OUT}")

    # Histogramas compactos para copiar al canvas
    print("\n--- HIST user-level hit rates (test) ---")
    for k, v in hist_payload.items():
        print(k, v)


if __name__ == "__main__":
    main()
