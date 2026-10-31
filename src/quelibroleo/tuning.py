"""Búsqueda conjunta de fusión y LambdaRank."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import LABEL_GAIN_MODES, PipelineConfig, search_ranges
from .data import USER_COL
from .metrics import ndcg_at_k
from .split_data import example_users
from .two_stage.columns import RankerTable
from .two_stage.retrieval import CandidateFusion

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
    label_gain: str = "linear"
    tune_on_example_users: bool = False


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
        label_gain = payload.get("label_gain", "linear")
        tune_on_example_users = payload.get("tune_on_example_users", False)
        if not isinstance(run_name, str):
            raise TypeError("run_name debe ser texto")
        if not isinstance(label_gain, str) or label_gain not in LABEL_GAIN_MODES:
            raise ValueError("label_gain debe ser 'linear' o 'exponential'")
        if not isinstance(tune_on_example_users, bool):
            raise TypeError("tune_on_example_users debe ser booleano")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Hiperparámetros inválidos en {path}") from exc
    if rrf_cutoff < 1 or not weights or not ranker:
        raise ValueError(f"Hiperparámetros inválidos en {path}")
    return TunedHyperparameters(
        rrf_cutoff=rrf_cutoff,
        weights=weights,
        ranker=ranker,
        run_name=run_name,
        label_gain=label_gain,
        tune_on_example_users=tune_on_example_users,
    )


def require_tuned_hyperparameters(config: PipelineConfig) -> TunedHyperparameters:
    """Lee el último tuneo. Si no hay archivo, pide un `evaluate --tune`."""

    path = tuned_hyperparameters_path(config.artifacts_dir)
    if not path.is_file():
        raise ValueError(
            f"No hay hiperparámetros tuneados en {path}. "
            "Corré evaluate --tune antes de una corrida sin --tune."
        )
    loaded = load_tuned_hyperparameters(path)
    LOGGER.info(
        "Reusando hiperparámetros de %s (corrida %s, corte RRF=%s, label_gain=%s, tune_on_example_users=%s)",
        path,
        loaded.run_name or "sin nombre",
        loaded.rrf_cutoff,
        loaded.label_gain,
        loaded.tune_on_example_users,
    )
    if loaded.label_gain != config.ranker.label_gain:
        LOGGER.warning(
            "El tuneo guardado se eligió con label_gain=%s y esta corrida usa %s. "
            "Esos hiperparámetros no corresponden a esta ganancia; "
            "corré evaluate --tune --label-gain %s.",
            loaded.label_gain,
            config.ranker.label_gain,
            config.ranker.label_gain,
        )
    if loaded.tune_on_example_users != config.tune_on_example_users:
        LOGGER.warning(
            "El tuneo guardado se eligió con tune_on_example_users=%s y esta corrida usa %s. "
            "Esos hiperparámetros no corresponden a esta población; "
            "corré evaluate --tune de nuevo.",
            loaded.tune_on_example_users,
            config.tune_on_example_users,
        )
    return loaded


def _mapping(value: Any, label: str) -> dict[str, Any]:
    """Exige un diccionario no vacío."""

    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} debe ser un objeto con valores")
    return value


def _number_map(value: Any, label: str) -> dict[str, float]:
    """Convierte los valores de un diccionario a float."""

    return {str(key): float(_number(item, f"{label}.{key}")) for key, item in _mapping(value, label).items()}


def _number(value: Any, label: str) -> int | float:
    """Acepta int o float. Un bool no cuenta como número."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} no es numérico")
    return value


def _as_int(value: Any, label: str) -> int:
    """Entero no negativo. Rechaza un float con parte decimal."""

    number = _number(value, label)
    if isinstance(number, float) and not number.is_integer():
        raise ValueError(f"{label} debe ser entero")
    parsed = int(number)
    if parsed < 0:
        raise ValueError(f"{label} debe ser positivo")
    return parsed


def _plain(value: Any) -> Any:
    """Saca un escalar de numpy para poder guardarlo en JSON."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def _leaves_are_impossible(
    parameters: dict[str, Any],
    *,
    n_rows: int | None,
) -> bool:
    """True si num_leaves no cabe en la profundidad o en las filas de train."""

    num_leaves = int(parameters["num_leaves"])
    min_child_samples = int(parameters["min_child_samples"])
    max_depth = int(parameters["max_depth"])
    leaves_exceed_depth = max_depth > 0 and num_leaves > 2**max_depth
    if n_rows is None:
        return leaves_exceed_depth
    return leaves_exceed_depth or min_child_samples * num_leaves > n_rows


def restrict_to_example_users(
    table: RankerTable,
    target: pd.DataFrame,
    users: set[str],
) -> tuple[RankerTable, pd.DataFrame]:
    """Deja en la validación y en la verdad solo los lectores de ejemplo.csv.

    Un lector de la entrega sin candidatos sigue en la verdad: el NDCG lo
    cuenta como cero. Si nadie de esa lista tiene filas, no hay fold usable.
    """

    kept = table.frame.loc[table.frame[USER_COL].astype(str).isin(users)].reset_index(
        drop=True
    )
    truth = target.loc[target[USER_COL].astype(str).isin(users)].reset_index(drop=True)
    if kept.empty or truth.empty:
        raise ValueError(
            "Ningún lector de ejemplo.csv entra al último fold; "
            "no se puede tunear solo sobre la entrega."
        )
    return RankerTable(frame=kept, feature_columns=table.feature_columns), truth


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
    from .two_stage.ranker import LambdaRanker

    retrieval = config.retrieval
    sources = sorted(materials[-1].source_candidates["source"].unique())
    if not sources:
        raise ValueError("No hay fuentes para ajustar")

    example_ids = example_users(config) if config.tune_on_example_users else None
    if example_ids is not None:
        present = set(materials[-1].target[USER_COL].astype(str)) & example_ids
        if not present:
            raise ValueError(
                "Ningún lector de ejemplo.csv entra al último fold; "
                "no se puede tunear solo sobre la entrega."
            )
        LOGGER.info(
            "Optuna puntúa el último fold en %s lectores de %s",
            len(present),
            config.example_path,
        )

    def objective(trial: Any) -> float:
        """NDCG@20 del último fold para un corte, unos pesos y un LambdaRank."""

        cutoff = int(retrieval.search.rrf_cutoff.suggest(trial, "rrf_cutoff"))
        weights = {
            source: float(
                retrieval.search.source_weight.suggest(trial, f"weight__{source}")
            )
            for source in sources
        }
        parameters = {
            name: spec.suggest(trial, name)
            for name, spec in search_ranges(config.ranker.search).items()
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
        target = materials[-1].target
        if example_ids is not None:
            validation, target = restrict_to_example_users(
                validation, target, example_ids
            )
        if _leaves_are_impossible(parameters, n_rows=len(train.frame)):
            raise optuna.TrialPruned()

        ranker = LambdaRanker(
            config.ranker,
            random_state=config.random_seed,
            overrides=parameters,
        ).fit(train, validation)
        recommendations = ranker.predict(validation)
        score = ndcg_at_k(recommendations, target, k=20)
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
