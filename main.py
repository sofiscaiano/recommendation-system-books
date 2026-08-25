from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quelibroleo.data import load_catalog, load_interactions
from quelibroleo.features import FeatureBuilder
from quelibroleo.metrics import ndcg_at_k, rating_errors
from quelibroleo.model import RatingModel
from quelibroleo.recommend import fill_missing_with_ejemplo, recommend_top_k, seen_items_by_user
from quelibroleo.split import leave_last_k

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena, evalúa y genera submission Quelibroleo.")
    parser.add_argument("--db", type=Path, default=ROOT / "data.db")
    parser.add_argument("--ejemplo", type=Path, default=ROOT / "ejemplo.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-eval", action="store_true", help="Solo submission, sin hold-out interno.")
    return parser.parse_args()


def run_eval(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    lectores: pd.DataFrame,
    libros: pd.DataFrame,
    features: FeatureBuilder,
    model: RatingModel,
    k: int,
    batch_size: int,
) -> None:
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
    )
    ndcg = ndcg_at_k(df_test, df_pred, k=k)
    print(f"Mean NDCG@{k}: {ndcg:.5f}")


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
    )
    df_final = fill_missing_with_ejemplo(df_pred, ejemplo, user_ids, k=k)
    df_final[["id_lector", "id_libro"]].to_csv(output_path, index=False)
    print(f"Escrito {output_path} ({df_final['id_lector'].nunique()} lectores, {len(df_final)} filas)")


def main() -> None:
    args = parse_args()
    print("Cargando datos...")
    df = load_interactions(args.db)
    libros, lectores = load_catalog(args.db)

    df_train, df_test = leave_last_k(df, k=args.k)
    print(f"Train: {df_train.shape}  Test: {df_test.shape}")

    features = FeatureBuilder().fit(df_train)
    print("Entrenando LightGBM (hold-out)...")
    model = RatingModel().fit(features.transform(df_train))

    if not args.skip_eval:
        print("Evaluación interna...")
        run_eval(df_train, df_test, lectores, libros, features, model, args.k, args.batch_size)

    print("Reentrenando con todas las interacciones...")
    features = FeatureBuilder().fit(df)
    model = RatingModel().fit(features.transform(df))

    print("Generando submission...")
    run_submission(
        df, lectores, libros, features, model, args.ejemplo, args.output, args.k, args.batch_size
    )


if __name__ == "__main__":
    main()
