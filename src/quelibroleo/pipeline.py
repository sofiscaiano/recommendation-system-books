"""Orquestación end-to-end del recomendador en dos etapas."""

from __future__ import annotations

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
from .data import ITEM_COL, observed_catalog
from .metrics import (
    ndcg_at_k,
    ndcg_per_user,
    retrieval_metrics,
    rows_against_reference,
)
from .split_data import SplitData, load_split, seen_books, validation_folds
from .submission import fill_submission
from .tuning import resolve_saved_or_default, tune_pipeline, tuned_hyperparameters_path
from .two_stage.columns import ColumnBuilder, RankerTable, attach_relevance
from .two_stage.ranker import LambdaRanker
from .two_stage.retrieval import (
    CandidateFusion,
    FusionResult,
    fit_retrievers,
    per_source_depth,
    retrieve_all,
    retrievers_from_config,
)

LOGGER = logging.getLogger(__name__)

_TWO_STAGE_DIAGNOSTICS = (
    "retrieval_folds.csv",
    "feature_ablations.csv",
    "feature_importance.csv",
    "evaluated_pipeline.joblib",
    "final_pipeline.joblib",
)


def clear_foreign_diagnostics(artifacts: Path, model: str) -> None:
    """Quita diagnósticos de two_stage cuando corre otro modelo.

    No borra ``tuned_hyperparameters.json``: two_stage lo vuelve a usar.
    """

    if model == "two_stage":
        return
    directory = Path(artifacts)
    for name in _TWO_STAGE_DIAGNOSTICS:
        path = directory / name
        if path.is_file():
            path.unlink()


@dataclass(frozen=True)
class FoldMaterial:
    """Artefactos previos a la fusión para un fold."""

    fold: int
    train: pd.DataFrame
    target: pd.DataFrame
    catalog: pd.DataFrame
    source_candidates: pd.DataFrame
    feature_builder: ColumnBuilder


@dataclass(frozen=True)
class EvaluationReport:
    """Métricas del hold-out que se comparan entre corridas."""

    submit_ndcg_at_20: float
    submit_baseline_ndcg_at_20: float
    submit_users: int
    model_ndcg_at_20: float
    baseline_ndcg_at_20: float
    retrieval_recall: float
    retrieval_oracle_ndcg_at_20: float
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


def _collect_material(
    train: pd.DataFrame,
    target: pd.DataFrame,
    users: tuple[str, ...],
    books: pd.DataFrame,
    config: PipelineConfig,
    fold: int,
    per_source: int,
) -> FoldMaterial:
    """Entrena las fuentes y el builder de un fold, sin fusionar todavía."""

    LOGGER.info(
        "Entrenando retrievers del fold %s para %s usuarios",
        fold,
        len(users),
    )
    catalog = observed_catalog(train, books)
    retrievers = fit_retrievers(retrievers_from_config(config), train, catalog)
    source_candidates = retrieve_all(
        retrievers,
        users,
        train,
        per_source=per_source,
    )
    builder = ColumnBuilder().fit(train, catalog)
    return FoldMaterial(
        fold=fold,
        train=train,
        target=target,
        catalog=catalog,
        source_candidates=source_candidates,
        feature_builder=builder,
    )


def _active_source_names(config: PipelineConfig) -> tuple[str, ...]:
    """Nombres de las fuentes que esta corrida entrena."""

    return tuple(retriever.source_name for retriever in retrievers_from_config(config))


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
    """Fusión y corte de RRF con los pesos fijos, sin Optuna."""

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
) -> tuple[RankerTable, FusionResult]:
    """Fusiona candidatos, arma features y les pone la relevancia del target."""

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


def _concat_feature_sets(sets: list[RankerTable]) -> RankerTable:
    """Apila folds y rellena con NaN las columnas que faltan en alguno."""

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
    return RankerTable(
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

    folds = validation_folds(interactions, config)
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
        n_sources=len(retrievers_from_config(config)),
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
                        "label_gain": config.ranker.label_gain,
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
) -> tuple[pd.DataFrame, FusionResult, RankerTable]:
    """Recupera, fusiona y rankea un historial con el modelo ya entrenado."""

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
            n_sources=len(retrievers_from_config(config)),
        ),
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
) -> tuple[EvaluationReport, TrainedPipeline | None]:
    """Evalúa una sola vez sobre el leave-last-20 exterior."""

    from .tracking import log_evaluation, log_summary_artifacts, track_command

    config = config or PipelineConfig()
    with track_command("evaluate", config, tune=tune):
        report, trained = _evaluate(config, tune=tune, save_artifacts=save_artifacts)
        log_evaluation(report)
        if save_artifacts:
            log_summary_artifacts(Path(config.artifacts_dir), model=config.model)
        return report, trained


def _evaluate(
    config: PipelineConfig,
    *,
    tune: bool,
    save_artifacts: bool,
) -> tuple[EvaluationReport, TrainedPipeline | None]:
    """Corre el modelo sobre el hold-out y, si corresponde, escribe artifacts/."""

    from .models import build_model

    split = load_split(config, holdout=True)
    LOGGER.info(
        "Holdout listo: %s usuarios y %s interacciones de test",
        len(split.evaluation_users),
        len(split.test),
    )
    model = build_model(config.model, config)
    model.fit(split, tune=tune)
    recommendations = model.recommend(split.evaluation_users)
    LOGGER.info("Ranking del holdout completado")
    top20 = recommendations[recommendations["rank"] <= 20].copy()

    popularity = PopularityRecommender().fit(
        split.history,
        set(observed_catalog(split.history, split.books)[ITEM_COL]),
    )
    baseline = popularity.recommend(
        split.evaluation_users,
        split.history,
        n=20,
    )
    baseline_per_user = ndcg_per_user(baseline, split.test)
    model_per_user = ndcg_per_user(top20, split.test)
    recall, oracle, budget = _retrieval_of(model, split)
    submit_ids = set(split.submit_users)
    submit_model, submit_users = _mean_for(model_per_user, submit_ids)
    submit_baseline, _ = _mean_for(baseline_per_user, submit_ids)
    report = EvaluationReport(
        submit_ndcg_at_20=submit_model,
        submit_baseline_ndcg_at_20=submit_baseline,
        submit_users=submit_users,
        model_ndcg_at_20=float(model_per_user.mean()),
        baseline_ndcg_at_20=float(baseline_per_user.mean()),
        retrieval_recall=recall,
        retrieval_oracle_ndcg_at_20=oracle,
        candidate_budget=budget,
        evaluation_users=len(split.evaluation_users),
    )

    if save_artifacts:
        from .tracking import with_run_name, write_run_marker

        artifacts = Path(config.artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        clear_foreign_diagnostics(artifacts, config.model)
        write_run_marker(artifacts)
        (artifacts / "config.json").write_text(
            json.dumps(with_run_name(config.as_dict()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (artifacts / "data_quality.json").write_text(
            json.dumps(
                with_run_name(asdict(split.quality)),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (artifacts / "evaluation.json").write_text(
            json.dumps(with_run_name(asdict(report)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _stamp_frame(top20).to_csv(
            artifacts / "evaluation_recommendations.csv", index=False
        )
        trained = getattr(model, "trained", None)
        if isinstance(trained, TrainedPipeline):
            _stamp_frame(pd.DataFrame(trained.fold_metrics)).to_csv(
                artifacts / "retrieval_folds.csv", index=False
            )
            _stamp_frame(trained.ranker.feature_importance()).to_csv(
                artifacts / "feature_importance.csv", index=False
            )
            _stamp_frame(pd.DataFrame(trained.ablation_metrics)).to_csv(
                artifacts / "feature_ablations.csv", index=False
            )
            joblib.dump(trained, artifacts / "evaluated_pipeline.joblib")
    trained = getattr(model, "trained", None)
    return report, trained if isinstance(trained, TrainedPipeline) else None


def _retrieval_of(model: object, split: SplitData) -> tuple[float, float, int]:
    """Recall, NDCG oracle y corte de candidatos. Ceros si el modelo no recupera."""

    fused = getattr(model, "fused", None)
    trained = getattr(model, "trained", None)
    if fused is None or not isinstance(trained, TrainedPipeline):
        return 0.0, 0.0, 0
    metrics = retrieval_metrics(fused.candidates, split.test)
    return metrics.recall, metrics.oracle_ndcg, trained.rrf_cutoff


def _mean_for(values: pd.Series, users: set[str]) -> tuple[float, int]:
    """Media del NDCG en un subconjunto de lectores, y cuántos entraron."""

    selected = values[values.index.astype(str).isin(users)]
    if selected.empty:
        return 0.0, 0
    return float(selected.mean()), len(selected)


def _stamp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrega la corrida activa como columna, sin tocar el frame original."""

    from .tracking import active_run_name

    run_name = active_run_name()
    if not run_name:
        return frame
    stamped = frame.copy()
    stamped.insert(0, "run_name", run_name)
    return stamped


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
        log_summary_artifacts(Path(config.artifacts_dir), model=config.model)
        return submission


def _submit(
    config: PipelineConfig,
    *,
    tune: bool,
    output_path: str | Path | None,
) -> pd.DataFrame:
    """Reentrena con toda la historia y escribe el CSV de entrega."""

    from .models import build_model

    split = load_split(config, holdout=False)
    model = build_model(config.model, config)
    model.fit(split, tune=tune)
    ranked = model.recommend(split.submit_users)
    catalog = observed_catalog(split.history, split.books)
    popularity = PopularityRecommender().fit(
        split.history,
        set(catalog[ITEM_COL]),
    )
    submission = fill_submission(
        ranked,
        split.submit_users,
        seen_books(split),
        popularity,
    )
    trained = getattr(model, "trained", None)
    if isinstance(trained, TrainedPipeline):
        origin = trained.hyperparameters_origin
        tuned_run = trained.hyperparameters_run_name
        hyperparameters = _hyperparameters_snapshot(trained)
    else:
        origin = getattr(model, "hyperparameters_origin", "default")
        tuned_run = getattr(model, "hyperparameters_run_name", "")
        recorded = getattr(model, "hyperparameters", None)
        hyperparameters = recorded if isinstance(recorded, dict) else {"model": model.name}
    from .tracking import archive_submission, write_run_marker

    destination = Path(output_path or config.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    archived = archive_submission(
        destination,
        origin=origin,
        hyperparameters_run_name=tuned_run,
        hyperparameters=hyperparameters,
        note=config.run_note,
        seed=config.random_seed,
    )
    LOGGER.info("Copia identificada en %s", archived)

    artifacts = Path(config.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    clear_foreign_diagnostics(artifacts, config.model)
    write_run_marker(artifacts)
    if isinstance(trained, TrainedPipeline):
        joblib.dump(trained, artifacts / "final_pipeline.joblib")
    return submission


class TwoStage:
    """Retrieval + LambdaRank. La validación es el último fold rolling."""

    name = "two_stage"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.trained: TrainedPipeline | None = None
        self.fused: FusionResult | None = None
        self._split: SplitData | None = None

    def fit(self, split: SplitData, *, tune: bool = False) -> None:
        """Entrena retrieval y LambdaRank. Las ablaciones solo corren en evaluate."""

        self._split = split
        self.trained = train_ranker_from_history(
            split.history,
            split.books,
            self.config,
            tune=tune,
            execute_ablations=split.holdout,
        )

    def recommend(self, users: tuple[str, ...] | list[str]) -> pd.DataFrame:
        """Rankea a esos lectores y guarda los candidatos fusionados."""

        if self.trained is None or self._split is None:
            raise RuntimeError("El modelo debe entrenarse antes de recomendar")
        recommendations, fused, _ = _score_history(
            self.trained,
            self._split.history,
            tuple(users),
            self._split.books,
            self.config,
        )
        self.fused = fused
        return recommendations


def _hyperparameters_snapshot(trained: TrainedPipeline) -> dict[str, Any]:
    """Pesos, corte y parámetros de LambdaRank que produjo esta corrida."""

    ranker_values = asdict(trained.ranker.config)
    ranker_values.pop("search", None)
    ranker_values.update(trained.ranker.overrides)
    return {
        "rrf_cutoff": trained.rrf_cutoff,
        "source_weights": dict(trained.weights),
        "ranker": ranker_values,
    }


def run_ablations(
    train_features: RankerTable,
    validation_features: RankerTable,
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
        RankerTable(train_features.frame, reference_columns),
        RankerTable(validation_features.frame, reference_columns),
    )
    reference_score = ndcg_at_k(
        reference_ranker.predict(
            RankerTable(validation_features.frame, reference_columns)
        ),
        validation_truth,
    )
    rows = []
    for name, keep in groups.items():
        columns = tuple(column for column in reference_columns if keep(column))
        train_subset = RankerTable(train_features.frame, columns)
        validation_subset = RankerTable(validation_features.frame, columns)
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
