"""Registro de corridas y trials en MLflow."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .pipeline import EvaluationReport

EXPERIMENT_NAME = "quelibroleo"
_SUMMARY_ARTIFACTS = (
    "config.json",
    "evaluation.json",
    "data_quality.json",
    "tuned_hyperparameters.json",
    "retrieval_folds.csv",
    "feature_ablations.csv",
    "evaluation_segments.csv",
    "feature_importance.csv",
)


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _previous_commit() -> str:
    import mlflow

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return ""
    runs = mlflow.tracking.MlflowClient().search_runs(
        [experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return ""
    return runs[0].data.tags.get("git_commit", "")


def change_description(note: str) -> tuple[str, str]:
    """Texto de la corrida y diff contra el commit de la corrida anterior."""

    head = _git("rev-parse", "HEAD")
    previous = _previous_commit()
    if previous:
        diff = _git("diff", "--stat", previous)
        heading = f"Cambios de código contra {previous[:7]}"
    else:
        diff = _git("diff", "--stat", "HEAD")
        heading = "Primera corrida; cambios sin commitear"
    lines = [part.strip() for part in (note.strip(), heading, diff) if part.strip()]
    summary = "\n".join(lines)
    return head, summary[:8000]


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

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    head, description = change_description(config.run_note)
    run_name = f"{command}-{'tune' if tune else 'fixed'}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_params(config, command=command, tune=tune))
        mlflow.set_tags(
            {
                "command": command,
                "tune": str(tune).lower(),
                "git_commit": head,
                "mlflow.note.content": description,
            }
        )
        yield


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
            metrics[f"{key}.{field}"] = number
    return metrics


def optuna_callback(stage: str):
    """Callback de Optuna que anida cada trial y grafica la serie en el padre."""

    def _callback(study: Any, trial: Any) -> None:
        import mlflow

        parent = mlflow.active_run()
        if parent is None or trial.value is None:
            return
        metrics = _trial_metrics(trial)
        for key, value in metrics.items():
            mlflow.log_metric(
                f"{stage}.{key}",
                value,
                step=int(trial.number),
                run_id=parent.info.run_id,
            )
        with mlflow.start_run(
            run_name=f"{stage}-trial-{trial.number}",
            nested=True,
        ):
            mlflow.log_params(
                {key: str(value)[:500] for key, value in trial.params.items()}
            )
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "stage": stage,
                    "trial_number": str(trial.number),
                    "study": study.study_name,
                }
            )

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
