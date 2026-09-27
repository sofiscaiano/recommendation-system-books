"""Registro de corridas y trials en MLflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from secrets import token_hex
from typing import Any

from .config import PipelineConfig
from .pipeline import EvaluationReport
from .tuning import has_saved_hyperparameters

EXPERIMENT_NAME = "quelibroleo"
_SUMMARY_ARTIFACTS = (
    "run.json",
    "config.json",
    "evaluation.json",
    "data_quality.json",
    "tuned_hyperparameters.json",
    "retrieval_folds.csv",
    "feature_ablations.csv",
    "evaluation_segments.csv",
    "feature_importance.csv",
    "run.log",
)
_current_run_name: ContextVar[str] = ContextVar("current_run_name", default="")


def new_run_name(command: str, *, tune: bool, saved: bool = False) -> str:
    """Nombre único de la corrida, compartido con Optuna y las salidas."""

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if tune:
        mode = "tune"
    elif saved:
        mode = "tuned"
    else:
        mode = "fixed"
    return f"{stamp}-{command}-{mode}-{token_hex(2)}"


def active_run_name() -> str:
    """Nombre de la corrida MLflow activa, o vacío fuera de `track_command`."""

    return _current_run_name.get()


def with_run_name(payload: dict[str, Any]) -> dict[str, Any]:
    """Antepone `run_name` a un JSON de salida cuando hay corrida activa."""

    run_name = active_run_name()
    if not run_name:
        return payload
    return {"run_name": run_name, **payload}


def write_run_marker(directory: Path) -> None:
    """Deja `run.json` en el directorio de artefactos de esta ejecución."""

    import json

    run_name = active_run_name()
    if not run_name:
        return
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(
        json.dumps({"run_name": run_name}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_output_marker(output_path: Path, payload: dict[str, Any]) -> None:
    """Sidecar de una salida CSV: el CSV de entrega no puede cambiar de columnas."""

    marker = output_path.with_name(output_path.name + ".run.json")
    marker.write_text(_dump(payload), encoding="utf-8")
    import mlflow

    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(marker))


def archive_submission(
    output_path: Path,
    *,
    origin: str,
    hyperparameters_run_name: str,
    hyperparameters: dict[str, Any],
    note: str,
    seed: int,
) -> Path:
    """Copia la entrega a una carpeta nueva y deja el manifiesto de procedencia."""

    import mlflow

    run_name = active_run_name()
    if not run_name:
        raise RuntimeError("No hay una corrida activa para archivar la entrega")
    active = mlflow.active_run()
    run_id = "" if active is None else active.info.run_id
    commit, dirty = git_state()
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "run_name": run_name,
        "mlflow_run_id": run_id,
        "git_commit": commit,
        "git_dirty": dirty,
        "hyperparameters_origin": origin,
        "hyperparameters_run_name": hyperparameters_run_name,
        "hyperparameters": hyperparameters,
        "note": note.strip(),
        "seed": seed,
        "csv_sha256": digest,
        "output": output_path.name,
    }
    folder = output_path.parent / "submissions" / run_name
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output_path, folder / "submission.csv")
    (folder / "manifest.json").write_text(_dump(manifest), encoding="utf-8")
    write_output_marker(output_path, manifest)
    if active is not None:
        mlflow.log_artifact(str(folder / "submission.csv"), artifact_path=f"submissions/{run_name}")
        mlflow.log_artifact(str(folder / "manifest.json"), artifact_path=f"submissions/{run_name}")
        tags = {"hyperparameters_origin": origin, "csv_sha256": digest}
        if hyperparameters_run_name:
            tags["hyperparameters_run_name"] = hyperparameters_run_name
        mlflow.set_tags(tags)
    return folder


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n"


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_ready(item())
    return value


def study_name(stage: str) -> str:
    """Nombre del study de Optuna, anclado al de la corrida cuando existe."""

    run_name = active_run_name()
    if not run_name:
        return stage
    return f"{run_name}__{stage}"


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def git_head() -> str:
    """Commit actual, o vacío si git no está disponible."""

    output = _git("rev-parse", "HEAD")
    if output is None:
        return ""
    return output.strip()


def git_state() -> tuple[str, bool | None]:
    """Commit de HEAD y si el árbol de trabajo tiene cambios sin commitear."""

    commit = git_head()
    status = _git("status", "--porcelain")
    if status is None:
        return commit, None
    return commit, bool(status.strip())


def tracking_uri() -> str:
    """URI local por defecto; se puede sobreescribir con MLFLOW_TRACKING_URI."""

    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def _flatten(prefix: str, value: Any, output: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child, nested, output)
        return
    if isinstance(value, (list, tuple)):
        text = ",".join(str(item) for item in value)
    else:
        text = str(value)
    output[prefix] = text[:500]


def _params(config: PipelineConfig, *, command: str, tune: bool) -> dict[str, str]:
    flat: dict[str, str] = {}
    _flatten("", config.as_dict(), flat)
    flat["command"] = command
    flat["tune"] = str(tune)
    return flat


def _metrics(report: EvaluationReport) -> dict[str, float]:
    raw = asdict(report)
    metrics: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        metrics[key] = float(value)
    delta = raw["paired_delta"]
    for key, value in delta.items():
        if isinstance(value, (int, float)):
            metrics[f"paired_delta.{key}"] = float(value)
    for segment, values in raw["segments"].items():
        for key, value in values.items():
            if isinstance(value, (int, float)):
                metrics[f"segment.{segment}.{key}"] = float(value)
    metrics["ndcg_lift_vs_popularity"] = (
        report.model_ndcg_at_20 - report.baseline_ndcg_at_20
    )
    return metrics


@contextmanager
def track_command(
    command: str,
    config: PipelineConfig,
    *,
    tune: bool,
) -> Iterator[None]:
    """Abre una corrida padre con la configuración de la ejecución."""

    import mlflow

    description = config.run_note.strip()
    if not description:
        raise ValueError("La corrida exige una descripción en config.run_note")

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    head, dirty = git_state()
    run_name = new_run_name(
        command,
        tune=tune,
        saved=has_saved_hyperparameters(config.artifacts_dir),
    )
    token = _current_run_name.set(run_name)
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(_params(config, command=command, tune=tune))
            mlflow.log_param("run_name", run_name)
            mlflow.set_tags(
                {
                    "command": command,
                    "tune": str(tune).lower(),
                    "run_name": run_name,
                    "git_commit": head,
                    "git_dirty": "" if dirty is None else str(dirty).lower(),
                    "mlflow.note.content": description,
                }
            )
            import logging

            logging.getLogger(__name__).info("Corrida %s", run_name)
            yield
    finally:
        _current_run_name.reset(token)


def log_evaluation(report: EvaluationReport) -> None:
    """Registra las métricas del hold-out en la corrida activa."""

    import mlflow

    if mlflow.active_run() is None:
        return
    mlflow.log_metrics(_metrics(report))


def log_summary_artifacts(artifacts_dir: Path) -> None:
    """Adjunta los JSON y CSV de diagnóstico, sin rankings ni modelos."""

    import mlflow

    if mlflow.active_run() is None:
        return
    for name in _SUMMARY_ARTIFACTS:
        path = artifacts_dir / name
        if path.is_file():
            mlflow.log_artifact(str(path))


def _numeric_fields(value: Any) -> dict[str, float]:
    from dataclasses import asdict, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, dict):
        return {}
    fields: dict[str, float] = {}
    for key, nested in value.items():
        if isinstance(nested, bool) or not isinstance(nested, (int, float)):
            continue
        fields[str(key)] = float(nested)
    return fields


def _trial_metrics(trial: Any) -> dict[str, float]:
    metrics = {"objective": float(trial.value)}
    for key, value in getattr(trial, "user_attrs", {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
            continue
        for field, number in _numeric_fields(value).items():
            metrics[_metric_key(f"{key}_{field}")] = number
    return metrics


def _metric_key(name: str) -> str:
    """MLflow 3 usa el punto como separador de dataset y no grafica esas claves."""

    return name.replace(".", "_").replace("/", "_")


@contextmanager
def track_study(stage: str) -> Iterator[None]:
    """Abre una corrida hija por study, donde se grafica la serie de trials."""

    import mlflow

    if mlflow.active_run() is None:
        yield
        return
    name = study_name(stage)
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.set_tags({"stage": stage, "study": name})
        yield


def optuna_callback(stage: str):
    """Registra cada trial como un step de la corrida del study."""

    def _callback(study: Any, trial: Any) -> None:
        import mlflow

        if mlflow.active_run() is None or trial.value is None:
            return
        for key, value in _trial_metrics(trial).items():
            mlflow.log_metric(_metric_key(key), value, step=int(trial.number))
        mlflow.set_tag("optuna_study", study.study_name)
        mlflow.set_tag("stage", stage)

    return _callback


def _contour_params(study: Any) -> list[str] | None:
    names = list(study.best_params)
    if len(names) <= 6:
        return names or None
    try:
        import optuna

        ranked = list(optuna.importance.get_param_importances(study))
    except (ValueError, RuntimeError, ImportError, ArithmeticError):
        ranked = names
    return ranked[:6]


def log_optuna_plots(study: Any, stage: str) -> None:
    """Guarda importancia, contorno y cortes del study en la corrida activa."""

    import logging
    import tempfile
    from pathlib import Path

    import mlflow

    if mlflow.active_run() is None:
        return
    completed = [
        trial
        for trial in study.trials
        if trial.value is not None and trial.state.name == "COMPLETE"
    ]
    if len(completed) < 2:
        return

    from optuna.visualization import plot_contour, plot_param_importances, plot_slice

    logger = logging.getLogger(__name__)
    plots = {
        "param_importances": lambda: plot_param_importances(study),
        "contour": lambda: plot_contour(study, params=_contour_params(study)),
        "slice": lambda: plot_slice(study),
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, builder in plots.items():
            try:
                figure = builder()
            except (ValueError, RuntimeError, ImportError, ArithmeticError):
                logger.warning("No se pudo construir plot_%s de %s", name, stage)
                continue
            path = root / f"{name}.html"
            figure.write_html(path, include_plotlyjs="cdn")
            mlflow.log_artifact(str(path), artifact_path=f"optuna/{stage}")
