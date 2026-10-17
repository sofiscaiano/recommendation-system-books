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
    rows_against_reference,
    segment_metrics,
)
from .ranker import LambdaRanker
from .retrieval import (
    CandidateFusion,
    FusionResult,
    build_default_retrievers,
    fit_retrievers,
    per_source_depth,
    retrieve_all,
)
from .splits import leave_last_k, rolling_origin_folds
from .tuning import resolve_saved_or_default, tune_pipeline, tuned_hyperparameters_path

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
    rrf_cutoff: int
    fold_metrics: list[dict[str, float]]
    ablation_metrics: list[dict[str, object]]
    hyperparameters_origin: str = "default"
    hyperparameters_run_name: str = ""


def _make_retrievers(config: PipelineConfig) -> list[Any]:
    retrieval = config.retrieval
    return build_default_retrievers(
        random_state=config.random_seed,
        implicit_factors=retrieval.implicit_factors,
        implicit_iterations=retrieval.implicit_iterations,
        half_life_days=retrieval.recency_half_life_days,
        tfidf_max_features=retrieval.tfidf_max_features,
    )


def _collect_material(
    train: pd.DataFrame,
    target: pd.DataFrame,
    users: tuple[str, ...],
    books: pd.DataFrame,
    config: PipelineConfig,
    fold: int,
    per_source: int,
    cache_dir: Path | None = None,
    used_caches: set[Path] | None = None,
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
            repr(("retrieval-cache-v4", config.retrieval, tuple(users), per_source)).encode()
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"fold_{fold}_{fingerprint.hexdigest()[:16]}.joblib"
        if used_caches is not None:
            used_caches.add(cache_path)
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


def _retain_latest_caches(cache_dir: Path, command: str, used: set[Path]) -> None:
    """Deja la última generación de evaluate y la de submit."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in used)
    (cache_dir / f"last_{command}.txt").write_text(
        "\n".join(names) + ("\n" if names else ""),
        encoding="utf-8",
    )
    other = "submit" if command == "evaluate" else "evaluate"
    other_manifest = cache_dir / f"last_{other}.txt"
    keep = set(names)
    if other_manifest.exists():
        keep.update(
            line.strip() for line in other_manifest.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    else:
        LOGGER.info(
            "Sin manifiesto de %s; no se borran caches ajenos a esta corrida",
            other,
        )
        return
    for path in cache_dir.glob("fold_*.joblib"):
        if path.name not in keep:
            LOGGER.info("Borrando cache fuera de las últimas generaciones: %s", path.name)
            path.unlink()


def _active_source_names(config: PipelineConfig) -> tuple[str, ...]:
    return tuple(retriever.source_name for retriever in _make_retrievers(config))


def _weights_for_active_sources(
    weights: dict[str, float],
    config: PipelineConfig,
) -> dict[str, float]:
    """Descarta pesos de fuentes que el pipeline ya no entrena."""

    active = set(_active_source_names(config))
    kept = {source: weight for source, weight in weights.items() if source in active}
    dropped = sorted(set(weights) - active)
    if dropped:
        LOGGER.info(
            "Se ignoran pesos de fuentes inactivas: %s",
            ", ".join(dropped),
        )
    if not kept:
        raise ValueError("No hay pesos para las fuentes activas")
    return kept


def _fusion_from_config(config: PipelineConfig) -> tuple[CandidateFusion, int]:
    retrieval = config.retrieval
    return (
        CandidateFusion(
            weights=_weights_for_active_sources(retrieval.source_weights, config),
            rrf_k=retrieval.rrf_k,
        ),
        retrieval.rrf_cutoff,
    )


def _material_to_features(
    material: FoldMaterial,
    fusion: CandidateFusion,
    cutoff: int,
) -> tuple[FeatureSet, FusionResult]:
    fused = fusion.fuse(material.source_candidates, cutoff)
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
    used_caches: set[Path] | None = None,
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

    if tune:
        saved = None
        hyperparameters_origin = "tune"
        from .tracking import active_run_name

        hyperparameters_run_name = active_run_name()
    else:
        saved, hyperparameters_origin = resolve_saved_or_default(config)
        hyperparameters_run_name = saved.run_name if hyperparameters_origin == "saved" else ""
    depth_cutoff = (
        int(config.retrieval.search.rrf_cutoff.high)
        if tune
        else saved.rrf_cutoff
    )
    per_source = per_source_depth(
        depth_cutoff,
        rrf_k=config.retrieval.rrf_k,
        n_sources=len(_make_retrievers(config)),
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
            used_caches,
        )
        for fold in folds
    ]

    overrides: dict[str, Any] = {}
    if tune:
        selection = tune_pipeline(materials, config)
        fusion = CandidateFusion(
            weights=selection.weights,
            rrf_k=config.retrieval.rrf_k,
        )
        cutoff = selection.rrf_cutoff
        overrides = selection.ranker
    elif saved is not None and saved.ranker:
        fusion = CandidateFusion(
            weights=_weights_for_active_sources(saved.weights, config),
            rrf_k=config.retrieval.rrf_k,
        )
        cutoff = saved.rrf_cutoff
        overrides = dict(saved.ranker)
    else:
        fusion, cutoff = _fusion_from_config(config)
    weights = dict(fusion.weights)

    built = [_material_to_features(material, fusion, cutoff) for material in materials]
    LOGGER.info("Candidatos fusionados; entrenando LambdaRank")
    feature_sets = [features for features, _ in built]
    train_set = _concat_feature_sets(feature_sets[:-1])
    validation_set = feature_sets[-1]
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
            ablated_candidates = fusion.fuse(without_source, cutoff).candidates
            metrics = retrieval_metrics(
                ablated_candidates,
                materials[-1].target,
            )
            source_ablation_rows.append(
                {
                    "stage": "retrieval",
                    "ablation": f"without_{source}",
                    "ndcg_at_20": metrics.oracle_ndcg,
                    "features": None,
                    "recall": metrics.recall,
                }
            )
        _, full_fusion = built[-1]
        full_retrieval = retrieval_metrics(
            full_fusion.candidates,
            materials[-1].target,
        )
        ablation_metrics = feature_ablations.to_dict(
            orient="records"
        ) + rows_against_reference(
            source_ablation_rows,
            stage="retrieval",
            ndcg_at_20=full_retrieval.oracle_ndcg,
            recall=full_retrieval.recall,
        )
    else:
        LOGGER.info("LambdaRank final entrenado; se omiten ablaciones repetidas")

    if tune:
        tuned_path = tuned_hyperparameters_path(config.artifacts_dir)
        tuned_path.parent.mkdir(parents=True, exist_ok=True)
        from .tracking import with_run_name

        tuned_path.write_text(
            json.dumps(
                with_run_name(
                    {
                        "rrf_cutoff": cutoff,
                        "source_weights": weights,
                        "ranker": overrides,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Hiperparámetros de Optuna guardados en %s", tuned_path)

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
        rrf_cutoff=cutoff,
        fold_metrics=fold_metrics,
        ablation_metrics=ablation_metrics,
        hyperparameters_origin=hyperparameters_origin,
        hyperparameters_run_name=hyperparameters_run_name,
    )


def _score_history(
    trained: TrainedPipeline,
    train: pd.DataFrame,
    users: tuple[str, ...],
    books: pd.DataFrame,
    config: PipelineConfig,
    used_caches: set[Path] | None = None,
) -> tuple[pd.DataFrame, FusionResult, FeatureSet]:
    material = _collect_material(
        train,
        target=train.iloc[0:0].copy(),
        users=users,
        books=books,
        config=config,
        fold=999,
        per_source=per_source_depth(
            trained.rrf_cutoff,
            rrf_k=config.retrieval.rrf_k,
            n_sources=len(_make_retrievers(config)),
        ),
        cache_dir=Path(config.artifacts_dir) / "cache",
        used_caches=used_caches,
    )
    fusion = CandidateFusion(
        weights=trained.weights,
        rrf_k=config.retrieval.rrf_k,
    )
    fused = fusion.fuse(material.source_candidates, trained.rrf_cutoff)
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

    from .tracking import log_evaluation, log_summary_artifacts, track_command

    config = config or PipelineConfig()
    with track_command("evaluate", config, tune=tune):
        report, trained = _evaluate(config, tune=tune, save_artifacts=save_artifacts)
        log_evaluation(report)
        if save_artifacts:
            log_summary_artifacts(Path(config.artifacts_dir))
        return report, trained


def _evaluate(
    config: PipelineConfig,
    *,
    tune: bool,
    save_artifacts: bool,
) -> tuple[EvaluationReport, TrainedPipeline]:
    bundle = load_dataset(config.db_path)
    interactions = bundle.interactions
    outer = leave_last_k(interactions, k=config.split.holdout_k)
    LOGGER.info(
        "Holdout listo: %s usuarios y %s interacciones de test",
        len(outer.evaluation_users),
        len(outer.test),
    )
    used_caches: set[Path] = set()
    trained = train_ranker_from_history(
        outer.train,
        bundle.books,
        config,
        tune=tune,
        used_caches=used_caches,
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
        used_caches,
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
        candidate_budget=trained.rrf_cutoff,
        evaluation_users=len(outer.evaluation_users),
    )

    if save_artifacts:
        from .tracking import with_run_name, write_run_marker

        artifacts = Path(config.artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        write_run_marker(artifacts)
        (artifacts / "config.json").write_text(
            json.dumps(with_run_name(config.as_dict()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (artifacts / "data_quality.json").write_text(
            json.dumps(with_run_name(asdict(bundle.report)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (artifacts / "evaluation.json").write_text(
            json.dumps(with_run_name(asdict(report)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _stamp_frame(pd.DataFrame(trained.fold_metrics)).to_csv(
            artifacts / "retrieval_folds.csv", index=False
        )
        _stamp_frame(top20).to_csv(artifacts / "evaluation_recommendations.csv", index=False)
        _stamp_frame(trained.ranker.feature_importance()).to_csv(
            artifacts / "feature_importance.csv", index=False
        )
        _stamp_frame(pd.DataFrame(trained.ablation_metrics)).to_csv(
            artifacts / "feature_ablations.csv", index=False
        )
        _stamp_frame(
            pd.DataFrame(
                [{"segment": name, **values} for name, values in report.segments.items()]
            )
        ).to_csv(artifacts / "evaluation_segments.csv", index=False)
        joblib.dump(trained, artifacts / "evaluated_pipeline.joblib")
    _retain_latest_caches(Path(config.artifacts_dir) / "cache", "evaluate", used_caches)
    return report, trained


def _stamp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega la corrida activa como columna, sin tocar el frame original."""

    from .tracking import active_run_name

    run_name = active_run_name()
    if not run_name:
        return frame
    stamped = frame.copy()
    stamped.insert(0, "run_name", run_name)
    return stamped


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

    from .tracking import log_summary_artifacts, track_command

    config = config or PipelineConfig()
    with track_command("submit", config, tune=tune):
        submission = _submit(config, tune=tune, output_path=output_path)
        log_summary_artifacts(Path(config.artifacts_dir))
        return submission


def _submit(
    config: PipelineConfig,
    *,
    tune: bool,
    output_path: str | Path | None,
) -> pd.DataFrame:
    bundle = load_dataset(config.db_path)
    interactions = bundle.interactions
    used_caches: set[Path] = set()
    trained = train_ranker_from_history(
        interactions,
        bundle.books,
        config,
        tune=tune,
        execute_ablations=False,
        used_caches=used_caches,
    )
    example = pd.read_csv(config.example_path, dtype=str)
    users = list(dict.fromkeys(example[USER_COL].astype(str)))
    ranked, _, _ = _score_history(
        trained,
        interactions,
        tuple(users),
        bundle.books,
        config,
        used_caches,
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
    from .tracking import archive_submission, write_run_marker

    destination = Path(output_path or config.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    archived = archive_submission(
        destination,
        origin=trained.hyperparameters_origin,
        hyperparameters_run_name=trained.hyperparameters_run_name,
        hyperparameters=_hyperparameters_snapshot(trained),
        note=config.run_note,
        seed=config.random_seed,
    )
    LOGGER.info("Copia identificada en %s", archived)

    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    write_run_marker(artifacts)
    joblib.dump(trained, artifacts / "final_pipeline.joblib")
    _retain_latest_caches(Path(config.artifacts_dir) / "cache", "submit", used_caches)
    return submission


def _hyperparameters_snapshot(trained: TrainedPipeline) -> dict[str, Any]:
    ranker_values = asdict(trained.ranker.config)
    ranker_values.pop("search", None)
    ranker_values.update(trained.ranker.overrides)
    return {
        "rrf_cutoff": trained.rrf_cutoff,
        "source_weights": dict(trained.weights),
        "ranker": ranker_values,
    }


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
    reference_columns = train_features.feature_columns
    reference_ranker = LambdaRanker(
        config.ranker,
        random_state=config.random_seed,
    ).fit(
        FeatureSet(train_features.frame, reference_columns),
        FeatureSet(validation_features.frame, reference_columns),
    )
    reference_score = ndcg_at_k(
        reference_ranker.predict(
            FeatureSet(validation_features.frame, reference_columns)
        ),
        validation_truth,
    )
    rows = []
    for name, keep in groups.items():
        columns = tuple(column for column in reference_columns if keep(column))
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
                "recall": None,
            }
        )
    return pd.DataFrame(
        rows_against_reference(
            rows,
            stage="ranker",
            ndcg_at_20=reference_score,
            features=float(len(reference_columns)),
        )
    )
