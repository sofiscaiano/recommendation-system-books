"""Búsqueda conjunta de fusión y LambdaRank."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .metrics import ndcg_at_k
from .retrieval import CandidateFusion

LOGGER = logging.getLogger(__name__)
TUNED_HYPERPARAMETERS_FILE = "tuned_hyperparameters.json"


@dataclass(frozen=True)
class PipelineSelection:
    """Hiperparámetros elegidos por el study de punta a punta."""

    rrf_cutoff: int
    weights: dict[str, float]
    ranker: dict[str, Any]
    study: object


@dataclass(frozen=True)
class TunedHyperparameters:
    """Fusión y LambdaRank listos para entrenar sin volver a buscar."""

    rrf_cutoff: int
    weights: dict[str, float]
    ranker: dict[str, Any]
    run_name: str = ""


def tuned_hyperparameters_path(artifacts_dir: Path) -> Path:
    """Ruta del tuneo guardado por `evaluate --tune`."""

    return Path(artifacts_dir) / TUNED_HYPERPARAMETERS_FILE


def has_saved_hyperparameters(artifacts_dir: Path) -> bool:
    """Indica si hay un tuneo previo para reutilizar."""

    return tuned_hyperparameters_path(artifacts_dir).is_file()


def load_tuned_hyperparameters(path: Path) -> TunedHyperparameters:
    """Lee el JSON de Optuna. Falla si el archivo existe pero no es válido."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"No se pudo leer {path}") from exc
    if isinstance(payload, dict) and "source_quotas" in payload:
        raise ValueError(
            f"{path} ajusta cuotas por fuente. "
            "La fusión ahora es RRF ponderado con un corte; corré evaluate --tune de nuevo."
        )
    try:
        if not isinstance(payload, dict):
            raise TypeError("el archivo no es un objeto")
        rrf_cutoff = _as_int(payload["rrf_cutoff"], "rrf_cutoff")
        weights = _number_map(payload["source_weights"], "source_weights")
        ranker = {
            key: _number(value, f"ranker.{key}")
            for key, value in _mapping(payload["ranker"], "ranker").items()
        }
        run_name = payload.get("run_name", "")
        if not isinstance(run_name, str):
            raise TypeError("run_name debe ser texto")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Hiperparámetros inválidos en {path}") from exc
    if rrf_cutoff < 1 or not weights or not ranker:
        raise ValueError(f"Hiperparámetros inválidos en {path}")
    return TunedHyperparameters(
        rrf_cutoff=rrf_cutoff,
        weights=weights,
        ranker=ranker,
        run_name=run_name,
    )


def resolve_saved_or_default(config: PipelineConfig) -> tuple[TunedHyperparameters, str]:
    """Devuelve el tuneo guardado, o la config fija si ese archivo no existe."""

    path = tuned_hyperparameters_path(config.artifacts_dir)
    if path.is_file():
        loaded = load_tuned_hyperparameters(path)
        LOGGER.info(
            "Reusando hiperparámetros de %s (corrida %s, corte RRF=%s)",
            path,
            loaded.run_name or "sin nombre",
            loaded.rrf_cutoff,
        )
        return loaded, "saved"
    LOGGER.info("Sin %s; se usan los hiperparámetros fijos", path)
    retrieval = config.retrieval
    return (
        TunedHyperparameters(
            rrf_cutoff=retrieval.rrf_cutoff,
            weights=dict(retrieval.source_weights),
            ranker={},
        ),
        "default",
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} debe ser un objeto con valores")
    return value


def _number_map(value: Any, label: str) -> dict[str, float]:
    return {str(key): float(_number(item, f"{label}.{key}")) for key, item in _mapping(value, label).items()}


def _number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} no es numérico")
    return value


def _as_int(value: Any, label: str) -> int:
    number = _number(value, label)
    if isinstance(number, float) and not number.is_integer():
        raise ValueError(f"{label} debe ser entero")
    parsed = int(number)
    if parsed < 0:
        raise ValueError(f"{label} debe ser positivo")
    return parsed


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

    def objective(trial: Any) -> float:
        cutoff = int(retrieval.rrf_cutoff_search.suggest(trial, "rrf_cutoff"))
        weights = {
            source: float(retrieval.fusion_weight.suggest(trial, f"weight__{source}"))
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
            rrf_k=retrieval.rrf_k,
        )
        built = [
            _material_to_features(material, fusion, cutoff)[0] for material in materials
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
        trial.set_user_attr("weights", dict(fusion.weights))
        trial.set_user_attr("rrf_cutoff", cutoff)
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
        rrf_cutoff=int(best.user_attrs["rrf_cutoff"]),
        weights={key: float(value) for key, value in best.user_attrs["weights"].items()},
        ranker={key: _plain(value) for key, value in best.user_attrs["ranker"].items()},
        study=study,
    )
