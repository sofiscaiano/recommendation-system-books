"""Búsquedas reproducibles para retrieval y fusión."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import SearchRange
from .metrics import RetrievalMetrics, retrieval_metrics
from .retrieval import CandidateFusion


@dataclass(frozen=True)
class FusionSelection:
    """Configuración elegida junto con su validación."""

    budget: int
    weights: dict[str, float]
    quotas: dict[str, int]
    metrics: RetrievalMetrics
    studies: dict[int, object]


def tune_fusion(
    source_candidates: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    budgets: tuple[int, ...] = (100, 200, 500, 1000),
    trials_per_budget: int = 20,
    tolerance: float = 0.01,
    rrf_k: float = 60.0,
    weight_range: SearchRange | None = None,
    quota_range: SearchRange | None = None,
    random_state: int = 42,
) -> FusionSelection:
    """Ajusta pesos/cuotas y elige el menor M casi óptimo.

    Se optimiza el promedio entre recall ponderado y techo oracle de NDCG.
    Después se elige el menor presupuesto cuyo objetivo queda a ``tolerance``
    del mejor, evitando seleccionar 1000 solo porque contiene más ítems.
    """

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("El tuning de fusión requiere `optuna`") from exc

    weight_range = weight_range or SearchRange(0.1, 3.0, log=True)
    quota_range = quota_range or SearchRange(0.0, 1.0)
    sources = sorted(source_candidates["source"].unique())
    if not sources:
        raise ValueError("No hay fuentes para ajustar")

    studies: dict[int, object] = {}
    best_by_budget: dict[
        int, tuple[float, dict[str, float], dict[str, int], RetrievalMetrics]
    ] = {}

    for budget in sorted(set(budgets)):
        if budget < 1:
            continue

        def objective(trial: Any, current_budget: int = budget) -> float:
            weights = {
                source: weight_range.suggest(trial, f"weight__{source}")
                for source in sources
            }
            quota_shares = {
                source: quota_range.suggest(trial, f"quota__{source}")
                for source in sources
            }
            share_total = sum(quota_shares.values()) or 1.0
            quotas = {
                source: round(
                    0.8 * current_budget * quota_shares[source] / share_total
                )
                for source in sources
            }
            fusion = CandidateFusion(
                weights=weights,
                quotas=quotas,
                rrf_k=rrf_k,
            ).fuse(source_candidates, current_budget)
            metrics = retrieval_metrics(fusion.candidates, truth)
            trial.set_user_attr("weights", weights)
            trial.set_user_attr("quotas", quotas)
            trial.set_user_attr("metrics", metrics)
            return 0.5 * metrics.weighted_recall + 0.5 * metrics.oracle_ndcg

        from .tracking import log_optuna_plots, optuna_callback, study_name

        stage = f"fusion-m{budget}"
        sampler = optuna.samplers.TPESampler(seed=random_state + budget)
        study = optuna.create_study(
            study_name=study_name(stage),
            direction="maximize",
            sampler=sampler,
        )
        study.optimize(
            objective,
            n_trials=trials_per_budget,
            callbacks=[optuna_callback(stage)],
            show_progress_bar=True,
        )
        log_optuna_plots(study, stage)
        studies[budget] = study
        trial = study.best_trial
        best_by_budget[budget] = (
            float(trial.value),
            dict(trial.user_attrs["weights"]),
            dict(trial.user_attrs["quotas"]),
            trial.user_attrs["metrics"],
        )

    if not best_by_budget:
        raise ValueError("No se proporcionaron presupuestos válidos")
    best_value = max(result[0] for result in best_by_budget.values())
    eligible = [
        budget
        for budget, result in best_by_budget.items()
        if result[0] >= best_value - tolerance
    ]
    selected_budget = min(eligible)
    _, weights, quotas, metrics = best_by_budget[selected_budget]
    return FusionSelection(
        budget=selected_budget,
        weights=weights,
        quotas=quotas,
        metrics=metrics,
        studies=studies,
    )
