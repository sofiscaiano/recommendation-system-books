from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quelibroleo.config import Settings
from quelibroleo.data import load_catalog, load_interactions
from quelibroleo.features import FeatureBuilder
from quelibroleo.metrics import ndcg_at_k, rating_errors
from quelibroleo.model import RatingModel
from quelibroleo.recommend import (
    candidate_books,
    fill_missing_with_ejemplo,
    recommend_top_k,
    seen_items_by_user,
)
from quelibroleo.split import leave_last_k

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    defaults = Settings()
    parser = argparse.ArgumentParser(description="Entrena, evalúa y genera submission Quelibroleo.")
    parser.add_argument("--db", type=Path, default=ROOT / "data.db")
    parser.add_argument("--ejemplo", type=Path, default=ROOT / "ejemplo.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=ROOT / "eval_recs.csv",
        help="Top-k del hold-out (id_lector, id_libro, score_pred, ranking). Ignorado con --skip-eval.",
    )
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--eval-user-frac",
        type=float,
        default=defaults.eval_user_frac,
        help="Fracción aleatoria de lectores del hold-out para eval (p. ej. 0.1 = 10%%). No afecta submission.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=defaults.eval_seed,
        help="Semilla del muestreo de lectores en eval.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Solo submission, sin hold-out interno.")
    parser.add_argument(
        "--only-interacted-items",
        action=argparse.BooleanOptionalAction,
        default=defaults.only_interacted_items,
        help="Si True, rankea solo libros con al menos una interacción. Desactivar con --no-only-interacted-items.",
    )
    parser.add_argument(
        "--filter-seen-genres",
        action=argparse.BooleanOptionalAction,
        default=defaults.filter_seen_genres,
        help="Si True, solo recomienda géneros que el lector ya interactuó. Desactivar con --no-filter-seen-genres.",
    )
    parser.add_argument(
        "--filter-seen-authors",
        action=argparse.BooleanOptionalAction,
        default=defaults.filter_seen_authors,
        help="Si True, solo recomienda autores que el lector ya interactuó. Desactivar con --no-filter-seen-authors.",
    )
    return parser.parse_args()


def _sample_eval_users(df_test: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"eval_user_frac debe estar en (0, 1], recibido {frac}")
    users = df_test["id_lector"].drop_duplicates().to_numpy()
    n_all = len(users)
    if frac >= 1.0 or n_all == 0:
        return df_test
    n = max(1, int(round(n_all * frac)))
    rng = np.random.default_rng(seed)
    sampled = rng.choice(users, size=n, replace=False)
    print(f"Eval sample: {n}/{n_all} lectores (frac={frac}, seed={seed})")
    return df_test[df_test["id_lector"].isin(sampled)].copy()


def run_eval(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    lectores: pd.DataFrame,
    libros: pd.DataFrame,
    features: FeatureBuilder,
    model: RatingModel,
    k: int,
    batch_size: int,
    eval_output: Path,
    filter_seen_genres: bool,
    filter_seen_authors: bool,
    eval_user_frac: float,
    eval_seed: int,
) -> None:
    df_test = _sample_eval_users(df_test, eval_user_frac, eval_seed)
    df_test_feat = features.transform(df_test)
    preds_rating = model.predict(df_test_feat)
    mse, rmse = rating_errors(df_test_feat["rating"], preds_rating)
    print(f"Hold-out rating  MSE: {mse:.4f}  RMSE: {rmse:.4f}")

    eval_users = df_test["id_lector"].unique()
    df_pred = recommend_top_k(
        model,
        features,
        lectores,
        libros,
        seen=seen_items_by_user(df_train),
        k=k,
        batch_size=batch_size,
        user_ids=eval_users,
        history=df_train,
        filter_seen_genres=filter_seen_genres,
        filter_seen_authors=filter_seen_authors,
    )
    ndcg = ndcg_at_k(df_test, df_pred, k=k)
    print(f"Mean NDCG@{k}: {ndcg:.5f}")

    cols = ["id_lector", "id_libro", "score_pred", "ranking"]
    df_pred[cols].to_csv(eval_output, index=False)
    print(f"Escrito {eval_output} ({df_pred['id_lector'].nunique()} lectores, {len(df_pred)} filas)")


def run_submission(
    df_all: pd.DataFrame,
    lectores: pd.DataFrame,
    libros: pd.DataFrame,
    features: FeatureBuilder,
    model: RatingModel,
    ejemplo_path: Path,
    output_path: Path,
    k: int,
    batch_size: int,
    filter_seen_genres: bool,
    filter_seen_authors: bool,
) -> None:
    ejemplo = pd.read_csv(ejemplo_path)
    user_ids = ejemplo["id_lector"].drop_duplicates().tolist()
    historial = set(df_all["id_lector"].unique())
    con_historial = [u for u in user_ids if u in historial]
    print(f"Submission: {len(con_historial)} lectores con historial de {len(user_ids)}")

    df_pred = recommend_top_k(
        model,
        features,
        lectores,
        libros,
        seen=seen_items_by_user(df_all),
        k=k,
        batch_size=batch_size,
        user_ids=con_historial,
        history=df_all,
        filter_seen_genres=filter_seen_genres,
        filter_seen_authors=filter_seen_authors,
    )
    df_final = fill_missing_with_ejemplo(df_pred, ejemplo, user_ids, k=k)
    df_final[["id_lector", "id_libro"]].to_csv(output_path, index=False)
    print(f"Escrito {output_path} ({df_final['id_lector'].nunique()} lectores, {len(df_final)} filas)")


def main() -> None:
    args = parse_args()
    print("Cargando datos...")
    df = load_interactions(args.db)
    libros, lectores = load_catalog(args.db)
    n_catalogo = len(libros)
    libros = candidate_books(libros, df, only_interacted_items=args.only_interacted_items)
    print(
        f"Libros candidatos: {len(libros)} / {n_catalogo} "
        f"(only_interacted_items={args.only_interacted_items}, "
        f"filter_seen_genres={args.filter_seen_genres}, "
        f"filter_seen_authors={args.filter_seen_authors})"
    )

    df_train, df_test = leave_last_k(df, k=args.k)
    print(f"Train: {df_train.shape}  Test: {df_test.shape}")

    features = FeatureBuilder().fit(df_train)
    print("Entrenando LightGBM (hold-out)...")
    model = RatingModel().fit(features.transform(df_train))
    importance = model.feature_importance()
    print("Feature importance (ordenada por gain):")
    print(importance.to_string(index=False, formatters={"gain": "{:.1f}".format, "gain_pct": "{:.2f}%".format}))

    if not args.skip_eval:
        print("Evaluación interna...")
        run_eval(
            df_train,
            df_test,
            lectores,
            libros,
            features,
            model,
            args.k,
            args.batch_size,
            args.eval_output,
            args.filter_seen_genres,
            args.filter_seen_authors,
            args.eval_user_frac,
            args.eval_seed,
        )

    print("Reentrenando con todas las interacciones...")
    features = FeatureBuilder().fit(df)
    model = RatingModel().fit(features.transform(df))

    print("Generando submission...")
    run_submission(
        df,
        lectores,
        libros,
        features,
        model,
        args.ejemplo,
        args.output,
        args.k,
        args.batch_size,
        args.filter_seen_genres,
        args.filter_seen_authors,
    )


if __name__ == "__main__":
    main()
