"""Búsqueda conjunta de fusión y LambdaRank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import PipelineConfig
from .metrics import ndcg_at_k
from .retrieval import CandidateFusion


@dataclass(frozen=True)
class PipelineSelection:
    """Hiperparámetros elegidos por el study de punta a punta."""

    budget: int
    weights: dict[str, float]
    quotas: dict[str, int]
    ranker: dict[str, Any]
    study: object


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _leaves_are_impossible(
    parameters: dict[str, Any],
    *,
    n_rows: int | None,
) -> bool:
    num_leaves = int(parameters["num_leaves"])
    min_child_samples = int(parameters["min_child_samples"])
    max_depth = int(parameters["max_depth"])
    leaves_exceed_depth = max_depth > 0 and num_leaves > 2**max_depth
    if n_rows is None:
        return leaves_exceed_depth
    return leaves_exceed_depth or min_child_samples * num_leaves > n_rows


def tune_pipeline(
    materials: list[Any],
    config: PipelineConfig,
) -> PipelineSelection:
    """Ajusta fusión y LambdaRank maximizando el NDCG@20 del último fold."""

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("El tuning requiere `optuna`") from exc

    if len(materials) < 2:
        raise ValueError("Se requieren al menos dos folds para el study")

    from .pipeline import _concat_feature_sets, _material_to_features
    from .ranker import LambdaRanker

    retrieval = config.retrieval
    sources = sorted(materials[-1].source_candidates["source"].unique())
    if not sources:
        raise ValueError("No hay fuentes para ajustar")
    budgets = [int(budget) for budget in retrieval.candidate_budgets if int(budget) >= 1]
    if not budgets:
        raise ValueError("No se proporcionaron presupuestos válidos")

    def objective(trial: Any) -> float:
        budget = int(trial.suggest_categorical("candidate_budget", budgets))
        weights = {
            source: retrieval.fusion_weight.suggest(trial, f"weight__{source}")
            for source in sources
        }
        quota_shares = {
            source: retrieval.fusion_quota_share.suggest(trial, f"quota__{source}")
            for source in sources
        }
        share_total = sum(quota_shares.values()) or 1.0
        quotas = {
            source: round(0.8 * budget * quota_shares[source] / share_total)
            for source in sources
        }
        parameters = {
            name: spec.suggest(trial, name)
            for name, spec in config.ranker.search_space.items()
        }
        if _leaves_are_impossible(parameters, n_rows=None):
            raise optuna.TrialPruned()

        fusion = CandidateFusion(
            weights=weights,
            quotas=quotas,
            rrf_k=retrieval.rrf_k,
        )
        built = [
            _material_to_features(material, fusion, budget)[0] for material in materials
        ]
        train = _concat_feature_sets(built[:-1])
        validation = built[-1]
        if _leaves_are_impossible(parameters, n_rows=len(train.frame)):
            raise optuna.TrialPruned()

        ranker = LambdaRanker(
            config.ranker,
            random_state=config.random_seed,
            overrides=parameters,
        ).fit(train, validation)
        recommendations = ranker.predict(validation)
        score = ndcg_at_k(recommendations, materials[-1].target, k=20)
        trial.set_user_attr("weights", weights)
        trial.set_user_attr("quotas", quotas)
        trial.set_user_attr("ranker", parameters)
        return score

    from .tracking import log_optuna_plots, optuna_callback, study_name, track_study

    stage = "pipeline"
    sampler = optuna.samplers.TPESampler(seed=config.random_seed)
    study = optuna.create_study(
        study_name=study_name(stage),
        direction="maximize",
        sampler=sampler,
    )
    with track_study(stage):
        study.optimize(
            objective,
            n_trials=config.optuna_trials,
            callbacks=[optuna_callback(stage)],
            show_progress_bar=True,
        )
        log_optuna_plots(study, stage)

    best = study.best_trial
    return PipelineSelection(
        budget=int(best.params["candidate_budget"]),
        weights={key: float(value) for key, value in best.user_attrs["weights"].items()},
        quotas={key: int(value) for key, value in best.user_attrs["quotas"].items()},
        ranker={key: _plain(value) for key, value in best.user_attrs["ranker"].items()},
        study=study,
    )
