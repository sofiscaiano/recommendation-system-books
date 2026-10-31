"""Búsqueda conjunta de fusión y LambdaRank."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
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
TUNING_HISTORY_FILE = "tuning_history.json"


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
    """Ruta del último tuneo escrito por `evaluate --tune`."""

    return Path(artifacts_dir) / TUNED_HYPERPARAMETERS_FILE


def tuning_history_path(artifacts_dir: Path) -> Path:
    """Ruta del historial de tuneos de two_stage."""

    return Path(artifacts_dir) / TUNING_HISTORY_FILE


def has_saved_hyperparameters(artifacts_dir: Path) -> bool:
    """Indica si hay historial o el JSON suelto de un tuneo anterior."""

    directory = Path(artifacts_dir)
    return tuning_history_path(directory).is_file() or tuned_hyperparameters_path(directory).is_file()


def two_stage_tuning_fingerprint(config: PipelineConfig) -> dict[str, Any]:
    """Config que define un tuneo de two_stage, sin los valores que Optuna elige.

    No entran la nota, la carpeta de artefactos ni la ruta de la entrega: no
    cambian el study. `example_path` entra solo si el objetivo usa esos lectores.
    El two-tower entra solo si es una fuente.
    """

    retrieval = config.retrieval
    ranker = config.ranker
    fingerprint: dict[str, Any] = {
        "random_seed": config.random_seed,
        "optuna_trials": config.optuna_trials,
        "tune_on_example_users": config.tune_on_example_users,
        "db_path": str(config.db_path),
        "example_path": (
            str(config.example_path) if config.tune_on_example_users else None
        ),
        "split": asdict(config.split),
        "retrieval": {
            "rrf_k": retrieval.rrf_k,
            "recency_half_life_days": retrieval.recency_half_life_days,
            "implicit_factors": retrieval.implicit_factors,
            "implicit_iterations": retrieval.implicit_iterations,
            "tfidf_max_features": retrieval.tfidf_max_features,
            "sources": list(retrieval.enabled_sources()),
            "search": asdict(retrieval.search),
        },
        "ranker": {
            "n_estimators": ranker.n_estimators,
            "early_stopping_rounds": ranker.early_stopping_rounds,
            "label_gain": ranker.label_gain,
            "search": asdict(ranker.search),
        },
    }
    if retrieval.two_tower_source:
        fingerprint["two_tower"] = asdict(config.two_tower)
    return fingerprint


def load_tuned_hyperparameters(path: Path) -> TunedHyperparameters:
    """Lee el JSON de Optuna. Falla si el archivo existe pero no es válido."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"No se pudo leer {path}") from exc
    return _parse_tuned_payload(payload, path)


def record_tuned_hyperparameters(
    config: PipelineConfig,
    *,
    rrf_cutoff: int,
    weights: dict[str, float],
    ranker: dict[str, Any],
) -> None:
    """Agrega este tuneo al historial y deja una copia en el JSON suelto."""

    from .tracking import with_run_name

    directory = Path(config.artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = with_run_name(
        {
            "config": two_stage_tuning_fingerprint(config),
            "config_complete": True,
            "rrf_cutoff": rrf_cutoff,
            "source_weights": weights,
            "label_gain": config.ranker.label_gain,
            "tune_on_example_users": config.tune_on_example_users,
            "ranker": ranker,
        }
    )
    history_path = tuning_history_path(directory)
    if history_path.is_file():
        entries = _read_history_list(history_path)
    else:
        entries = _legacy_entries(directory)
    entries.append(payload)
    _write_history(history_path, entries)
    tuned_path = tuned_hyperparameters_path(directory)
    tuned_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Hiperparámetros de Optuna guardados en %s", history_path)


def has_matching_tuned_hyperparameters(config: PipelineConfig) -> bool:
    """True si el historial tiene un tuneo de esta misma config."""

    try:
        return _find_saved_tuning(config, persist_legacy=False) is not None
    except ValueError:
        return False


def require_tuned_hyperparameters(config: PipelineConfig) -> TunedHyperparameters:
    """Busca en el historial un tuneo con esta config. Si no hay, pide `--tune`."""

    path = tuning_history_path(config.artifacts_dir)
    loaded = _find_saved_tuning(config, persist_legacy=True)
    if loaded is None:
        raise ValueError(
            f"No hay un tuneo con esta config en {path}. "
            "Corré evaluate --tune antes de una corrida sin --tune."
        )
    LOGGER.info(
        "Reusando hiperparámetros de %s (corrida %s, corte RRF=%s, label_gain=%s, tune_on_example_users=%s)",
        path,
        loaded.run_name or "sin nombre",
        loaded.rrf_cutoff,
        loaded.label_gain,
        loaded.tune_on_example_users,
    )
    return loaded


def _find_saved_tuning(
    config: PipelineConfig,
    *,
    persist_legacy: bool,
) -> TunedHyperparameters | None:
    """Última entrada del historial cuya config coincide. None si no hay ninguna."""

    directory = Path(config.artifacts_dir)
    history_path = tuning_history_path(directory)
    if history_path.is_file():
        entries = _read_history_list(history_path)
    else:
        entries = _legacy_entries(directory)
        if persist_legacy and entries:
            _write_history(history_path, entries)
    matched = [entry for entry in entries if _same_tuning_config(entry, config)]
    if not matched:
        return None
    return _parse_tuned_payload(matched[-1], history_path)


def _same_tuning_config(entry: dict[str, Any], config: PipelineConfig) -> bool:
    """True si esa entrada se tuneó con la misma config que esta corrida.

    Un JSON anterior a este historial no guardó la config. Coincide solo en
    `label_gain`, `tune_on_example_users` y el conjunto de fuentes.
    """

    stored = entry.get("config")
    if entry.get("config_complete") is True and isinstance(stored, dict):
        return _canonical(stored) == _canonical(two_stage_tuning_fingerprint(config))
    weights = entry.get("source_weights")
    sources = entry.get("sources")
    if not isinstance(sources, list):
        if not isinstance(weights, dict):
            return False
        sources = list(weights)
    label_gain = entry.get("label_gain", "linear")
    population = entry.get("tune_on_example_users", False)
    return (
        label_gain == config.ranker.label_gain
        and population is config.tune_on_example_users
        and set(sources) == set(config.retrieval.enabled_sources())
    )


def _canonical(value: Any) -> Any:
    """Normaliza un diccionario por JSON para comparar fingerprints."""

    return json.loads(json.dumps(value, sort_keys=True))


def _read_history_list(path: Path) -> list[dict[str, Any]]:
    """Lee la lista de tuneos. Falla si el archivo no es esa lista."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"No se pudo leer {path}") from exc
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError(f"Hiperparámetros inválidos en {path}")
    return payload


def _legacy_entries(artifacts_dir: Path) -> list[dict[str, Any]]:
    """El JSON suelto, si existe, como única entrada previa al historial."""

    path = tuned_hyperparameters_path(artifacts_dir)
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"No se pudo leer {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Hiperparámetros inválidos en {path}")
    return [payload]


def _write_history(path: Path, entries: list[dict[str, Any]]) -> None:
    """Escribe el historial completo."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _parse_tuned_payload(payload: Any, path: Path) -> TunedHyperparameters:
    """Valida un objeto de tuneo. `path` solo entra en el mensaje de error."""

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
