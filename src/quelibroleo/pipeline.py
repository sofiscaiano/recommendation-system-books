"""Orquestación end-to-end del recomendador en dos etapas."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .baselines import PopularityRecommender
from .config import PipelineConfig
from .data import ITEM_COL, USER_COL, load_dataset, observed_catalog
from .features import FeatureBuilder, FeatureSet, attach_relevance
from .metrics import (
    catalog_coverage,
    long_tail_exposure,
    ndcg_at_k,
    ndcg_per_user,
    paired_bootstrap_delta,
    recall_at_k,
    retrieval_metrics,
    segment_metrics,
)
from .ranker import LambdaRanker, tune_lambdarank
from .retrieval import (
    CandidateFusion,
    FusionResult,
    build_default_retrievers,
    fit_retrievers,
    retrieve_all,
)
from .splits import leave_last_k, rolling_origin_folds
from .tuning import FusionSelection, tune_fusion

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldMaterial:
    """Artefactos previos a la fusión para un fold."""

    fold: int
    train: pd.DataFrame
    target: pd.DataFrame
    catalog: pd.DataFrame
    source_candidates: pd.DataFrame
    feature_builder: FeatureBuilder


@dataclass(frozen=True)
class EvaluationReport:
    """Métricas principales y diagnósticos serializables."""

    baseline_ndcg_at_20: float
    model_ndcg_at_20: float
    model_recall_at_20: float
    retrieval_recall: float
    retrieval_weighted_recall: float
    retrieval_oracle_ndcg_at_20: float
    catalog_coverage_at_20: float
    long_tail_exposure_at_20: float
    paired_delta: dict[str, float]
    segments: dict[str, dict[str, float]]
    candidate_budget: int
    evaluation_users: int


@dataclass
class TrainedPipeline:
    """Componentes necesarios para puntuar un fold o el catálogo final."""

    ranker: LambdaRanker
    weights: dict[str, float]
    quotas: dict[str, int]
    budget: int
    fold_metrics: list[dict[str, float]]
    ablation_metrics: list[dict[str, object]]


def _make_retrievers(config: PipelineConfig) -> list[Any]:
    retrieval = config.retrieval
    return build_default_retrievers(
        random_state=config.random_seed,
        svd_factors=retrieval.svd_factors,
        svd_epochs=retrieval.svd_epochs,
        implicit_factors=retrieval.implicit_factors,
        implicit_iterations=retrieval.implicit_iterations,
        half_life_days=retrieval.recency_half_life_days,
        tfidf_max_features=retrieval.tfidf_max_features,
    )


def _per_source_depth(
    budget: int,
    quotas: dict[str, int],
    source_count: int = 7,
) -> int:
    """Sobre-recupera por fuente sin multiplicar M por todas las fuentes."""

    fair_share = int(np.ceil(2.0 * budget / source_count))
    return max(fair_share, max(quotas.values(), default=0))


def _collect_material(
    train: pd.DataFrame,
    target: pd.DataFrame,
    users: tuple[str, ...],
    books: pd.DataFrame,
    config: PipelineConfig,
    fold: int,
    per_source: int,
    cache_dir: Path | None = None,
) -> FoldMaterial:
    cache_path: Path | None = None
    if cache_dir is not None:
        fingerprint = hashlib.sha256()
        fingerprint.update(
            str(
                pd.util.hash_pandas_object(
                    train[[USER_COL, ITEM_COL, "fecha", "rating"]],
                    index=False,
                ).sum()
            ).encode()
        )
        fingerprint.update(
            repr(("retrieval-cache-v2", config.retrieval, tuple(users), per_source)).encode()
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"fold_{fold}_{fingerprint.hexdigest()[:16]}.joblib"
        if cache_path.exists():
            LOGGER.info("Cargando candidatos cacheados del fold %s", fold)
            cached = joblib.load(cache_path)
            return FoldMaterial(
                fold=fold,
                train=train,
                target=target,
                catalog=cached["catalog"],
                source_candidates=cached["source_candidates"],
                feature_builder=cached["feature_builder"],
            )

    LOGGER.info(
        "Entrenando retrievers del fold %s para %s usuarios",
        fold,
        len(users),
    )
    catalog = observed_catalog(train, books)
    retrievers = fit_retrievers(_make_retrievers(config), train, catalog)
    source_candidates = retrieve_all(
        retrievers,
        users,
        train,
        per_source=per_source,
    )
    builder = FeatureBuilder().fit(train, catalog)
    material = FoldMaterial(
        fold=fold,
        train=train,
        target=target,
        catalog=catalog,
        source_candidates=source_candidates,
        feature_builder=builder,
    )
    if cache_path is not None:
        LOGGER.info("Guardando cache del fold %s", fold)
        joblib.dump(
            {
                "catalog": catalog,
                "source_candidates": source_candidates,
                "feature_builder": builder,
            },
            cache_path,
            compress=3,
        )
    return material


def _fusion_from_config(config: PipelineConfig) -> tuple[CandidateFusion, int]:
    retrieval = config.retrieval
    return (
        CandidateFusion(
            weights=retrieval.source_weights,
            quotas=retrieval.source_quotas,
            rrf_k=retrieval.rrf_k,
        ),
        retrieval.selected_budget,
    )


def _material_to_features(
    material: FoldMaterial,
    fusion: CandidateFusion,
    budget: int,
) -> tuple[FeatureSet, FusionResult]:
    fused = fusion.fuse(material.source_candidates, budget)
    features = material.feature_builder.transform(
        fused.candidates,
        fused.source_candidates,
    )
    labeled = attach_relevance(
        features,
        material.target,
        query_suffix=f"__fold_{material.fold}",
    )
    return labeled, fused


def _concat_feature_sets(sets: list[FeatureSet]) -> FeatureSet:
    if not sets:
        raise ValueError("No hay folds de features para concatenar")
    feature_columns = tuple(
        sorted({column for feature_set in sets for column in feature_set.feature_columns})
    )
    frames = []
    for feature_set in sets:
        frame = feature_set.frame.copy()
        for column in feature_columns:
            if column not in frame:
                frame[column] = np.nan
        frames.append(frame)
    return FeatureSet(
        frame=pd.concat(frames, ignore_index=True, sort=False),
        feature_columns=feature_columns,
    )


def train_ranker_from_history(
    interactions: pd.DataFrame,
    books: pd.DataFrame,
    config: PipelineConfig,
    *,
    tune: bool = False,
    execute_ablations: bool = True,
) -> TrainedPipeline:
    """Entrena LambdaRank exclusivamente con pseudo-futuros rolling."""

    split_config = config.split
    folds = rolling_origin_folds(
        interactions,
        n_folds=split_config.rolling_folds,
        horizon=split_config.rolling_horizon,
        min_history=split_config.min_history,
    )
    if len(folds) < 2:
        raise ValueError("Se requieren al menos dos folds rolling para entrenar")

    maximum_budget = (
        max(config.retrieval.candidate_budgets)
        if tune
        else config.retrieval.selected_budget
    )
    per_source = _per_source_depth(
        maximum_budget,
        config.retrieval.source_quotas,
    )
    materials = [
        _collect_material(
            fold.train,
            fold.target,
            fold.query_users,
            books,
            config,
            fold.fold,
            per_source,
            Path(config.artifacts_dir) / "cache",
        )
        for fold in folds
    ]

    if tune:
        selection: FusionSelection = tune_fusion(
            materials[-1].source_candidates,
            materials[-1].target,
            budgets=config.retrieval.candidate_budgets,
            trials_per_budget=max(5, config.ranker.optuna_trials // 2),
            rrf_k=config.retrieval.rrf_k,
            random_state=config.random_seed,
        )
        fusion = CandidateFusion(
            weights=selection.weights,
            quotas=selection.quotas,
            rrf_k=config.retrieval.rrf_k,
        )
        budget = selection.budget
        weights = selection.weights
        quotas = selection.quotas
    else:
        fusion, budget = _fusion_from_config(config)
        weights = dict(config.retrieval.source_weights)
        quotas = dict(config.retrieval.source_quotas)

    built = [_material_to_features(material, fusion, budget) for material in materials]
    LOGGER.info("Candidatos fusionados; entrenando LambdaRank")
    feature_sets = [features for features, _ in built]
    train_set = _concat_feature_sets(feature_sets[:-1])
    validation_set = feature_sets[-1]

    overrides: dict[str, Any] = {}
    if tune:
        overrides, _ = tune_lambdarank(
            train_set,
            validation_set,
            materials[-1].target,
            base_config=config.ranker,
            n_trials=config.ranker.optuna_trials,
            random_state=config.random_seed,
        )
    ranker = LambdaRanker(
        config.ranker,
        random_state=config.random_seed,
        overrides=overrides,
    ).fit(train_set, validation_set)
    ablation_metrics: list[dict[str, object]] = []
    if execute_ablations:
        LOGGER.info("LambdaRank principal entrenado; ejecutando ablaciones")
        feature_ablations = run_ablations(
            train_set,
            validation_set,
            materials[-1].target,
            config,
        )
        source_ablation_rows = []
        validation_sources = materials[-1].source_candidates
        for source in sorted(validation_sources["source"].unique()):
            without_source = validation_sources[
                validation_sources["source"] != source
            ]
            ablated_candidates = fusion.fuse(without_source, budget).candidates
            metrics = retrieval_metrics(
                ablated_candidates,
                materials[-1].target,
            )
            source_ablation_rows.append(
                {
                    "stage": "retrieval",
                    "ablation": f"without_{source}",
                    "ndcg_at_20": metrics.oracle_ndcg,
                    "recall": metrics.recall,
                }
            )
        ablation_metrics = (
            feature_ablations.to_dict(orient="records") + source_ablation_rows
        )
    else:
        LOGGER.info("LambdaRank final entrenado; se omiten ablaciones repetidas")

    fold_metrics = []
    for material, (features, fused) in zip(materials, built, strict=True):
        metrics = retrieval_metrics(fused.candidates, material.target)
        fold_metrics.append(
            {
                "fold": float(material.fold),
                "recall": metrics.recall,
                "weighted_recall": metrics.weighted_recall,
                "oracle_ndcg_at_20": metrics.oracle_ndcg,
                "mean_candidates": metrics.mean_candidates,
            }
        )
    return TrainedPipeline(
        ranker=ranker,
        weights=weights,
        quotas=quotas,
        budget=budget,
        fold_metrics=fold_metrics,
        ablation_metrics=ablation_metrics,
    )


def _score_history(
    trained: TrainedPipeline,
    train: pd.DataFrame,
    users: tuple[str, ...],
    books: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, FusionResult, FeatureSet]:
    material = _collect_material(
        train,
        target=train.iloc[0:0].copy(),
        users=users,
        books=books,
        config=config,
        fold=999,
        per_source=_per_source_depth(trained.budget, trained.quotas),
        cache_dir=Path(config.artifacts_dir) / "cache",
    )
    fusion = CandidateFusion(
        weights=trained.weights,
        quotas=trained.quotas,
        rrf_k=config.retrieval.rrf_k,
    )
    fused = fusion.fuse(material.source_candidates, trained.budget)
    features = material.feature_builder.transform(
        fused.candidates,
        fused.source_candidates,
    )
    recommendations = trained.ranker.predict(features)
    return recommendations, fused, features


def evaluate(
    config: PipelineConfig | None = None,
    *,
    tune: bool = False,
    save_artifacts: bool = True,
) -> tuple[EvaluationReport, TrainedPipeline]:
    """Evalúa una sola vez sobre el leave-last-20 exterior."""

    config = config or PipelineConfig()
    bundle = load_dataset(config.db_path)
    interactions = bundle.interactions
    outer = leave_last_k(interactions, k=config.split.holdout_k)
    LOGGER.info(
        "Holdout listo: %s usuarios y %s interacciones de test",
        len(outer.evaluation_users),
        len(outer.test),
    )
    trained = train_ranker_from_history(
        outer.train,
        bundle.books,
        config,
        tune=tune,
    )

    popularity = PopularityRecommender().fit(
        outer.train,
        set(observed_catalog(outer.train, bundle.books)[ITEM_COL]),
    )
    baseline = popularity.recommend(
        outer.evaluation_users,
        outer.train,
        n=20,
    )
    recommendations, fused, _ = _score_history(
        trained,
        outer.train,
        outer.evaluation_users,
        bundle.books,
        config,
    )
    LOGGER.info("Ranking del holdout completado")
    top20 = recommendations[recommendations["rank"] <= 20].copy()

    baseline_per_user = ndcg_per_user(baseline, outer.test)
    model_per_user = ndcg_per_user(top20, outer.test)
    retrieval = retrieval_metrics(fused.candidates, outer.test)
    eligible_items = set(observed_catalog(outer.train, bundle.books)[ITEM_COL])
    report = EvaluationReport(
        baseline_ndcg_at_20=float(baseline_per_user.mean()),
        model_ndcg_at_20=float(model_per_user.mean()),
        model_recall_at_20=recall_at_k(top20, outer.test),
        retrieval_recall=retrieval.recall,
        retrieval_weighted_recall=retrieval.weighted_recall,
        retrieval_oracle_ndcg_at_20=retrieval.oracle_ndcg,
        catalog_coverage_at_20=catalog_coverage(top20, eligible_items),
        long_tail_exposure_at_20=long_tail_exposure(top20, outer.train),
        paired_delta=paired_bootstrap_delta(
            model_per_user,
            baseline_per_user,
            random_state=config.random_seed,
        ),
        segments=segment_metrics(
            top20,
            fused.candidates,
            outer.test,
            outer.train,
            eligible_items,
        ),
        candidate_budget=trained.budget,
        evaluation_users=len(outer.evaluation_users),
    )

    if save_artifacts:
        artifacts = Path(config.artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "config.json").write_text(
            json.dumps(config.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (artifacts / "data_quality.json").write_text(
            json.dumps(asdict(bundle.report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (artifacts / "evaluation.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(trained.fold_metrics).to_csv(
            artifacts / "retrieval_folds.csv", index=False
        )
        top20.to_csv(artifacts / "evaluation_recommendations.csv", index=False)
        trained.ranker.feature_importance().to_csv(
            artifacts / "feature_importance.csv", index=False
        )
        pd.DataFrame(trained.ablation_metrics).to_csv(
            artifacts / "feature_ablations.csv", index=False
        )
        pd.DataFrame(
            [{"segment": name, **values} for name, values in report.segments.items()]
        ).to_csv(artifacts / "evaluation_segments.csv", index=False)
        joblib.dump(trained, artifacts / "evaluated_pipeline.joblib")
    return report, trained


def _fill_submission(
    ranked: pd.DataFrame,
    users: list[str],
    history: pd.DataFrame,
    popularity: PopularityRecommender,
) -> pd.DataFrame:
    top = ranked[ranked["rank"] <= 20][[USER_COL, ITEM_COL, "rank"]].copy()
    existing = top.groupby(USER_COL)[ITEM_COL].agg(list).to_dict()
    fallback = popularity.recommend(users, history, n=100)
    fallback_by_user = fallback.groupby(USER_COL)[ITEM_COL].agg(list).to_dict()
    seen_by_user = history.groupby(USER_COL)[ITEM_COL].agg(set).to_dict()
    rows = []
    for user in users:
        chosen: list[str] = []
        for item in existing.get(user, []) + fallback_by_user.get(user, []):
            if item in seen_by_user.get(user, set()) or item in chosen:
                continue
            chosen.append(item)
            if len(chosen) == 20:
                break
        if len(chosen) != 20:
            raise RuntimeError(f"No se pudieron completar 20 libros para {user}")
        rows.extend({USER_COL: user, ITEM_COL: item} for item in chosen)
    return pd.DataFrame.from_records(rows, columns=[USER_COL, ITEM_COL])


def train_final_and_submit(
    config: PipelineConfig | None = None,
    *,
    tune: bool = False,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Reentrena con toda la historia válida y genera la entrega."""

    config = config or PipelineConfig()
    bundle = load_dataset(config.db_path)
    interactions = bundle.interactions
    trained = train_ranker_from_history(
        interactions,
        bundle.books,
        config,
        tune=tune,
        execute_ablations=False,
    )
    example = pd.read_csv(config.example_path, dtype=str)
    users = list(dict.fromkeys(example[USER_COL].astype(str)))
    ranked, _, _ = _score_history(
        trained,
        interactions,
        tuple(users),
        bundle.books,
        config,
    )
    catalog = observed_catalog(interactions, bundle.books)
    popularity = PopularityRecommender().fit(
        interactions,
        set(catalog[ITEM_COL]),
    )
    seen_history = pd.concat(
        [interactions, bundle.invalid_interactions],
        ignore_index=True,
        sort=False,
    )
    submission = _fill_submission(ranked, users, seen_history, popularity)
    destination = Path(output_path or config.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)

    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained, artifacts / "final_pipeline.joblib")
    return submission


def run_ablations(
    train_features: FeatureSet,
    validation_features: FeatureSet,
    validation_truth: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Compara familias de features manteniendo fijo el conjunto candidato."""

    groups = {
        "without_retrieval": lambda column: not column.startswith(
            ("score_", "rank_", "present_", "reciprocal_", "rrf_", "retrieval_")
        ),
        "without_sequence": lambda column: not any(
            token in column
            for token in ("recent_", "transition_", "same_last_", "publication_age")
        ),
    }
    rows = []
    for name, keep in groups.items():
        columns = tuple(column for column in train_features.feature_columns if keep(column))
        train_subset = FeatureSet(train_features.frame, columns)
        validation_subset = FeatureSet(validation_features.frame, columns)
        ranker = LambdaRanker(
            config.ranker,
            random_state=config.random_seed,
        ).fit(train_subset, validation_subset)
        score = ndcg_at_k(ranker.predict(validation_subset), validation_truth)
        rows.append(
            {
                "stage": "ranker",
                "ablation": name,
                "ndcg_at_20": score,
                "features": len(columns),
            }
        )
    return pd.DataFrame(rows)
