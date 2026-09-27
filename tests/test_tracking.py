"""Registro de métricas sin correr el pipeline completo."""

from dataclasses import dataclass

import mlflow

from quelibroleo.config import PipelineConfig
from quelibroleo.pipeline import EvaluationReport
from quelibroleo.tracking import log_evaluation, track_command


def test_track_command_requires_note() -> None:
    import pytest

    with pytest.raises(ValueError, match="descripción"):
        with track_command("evaluate", PipelineConfig(), tune=False):
            pass


def test_track_command_logs_holdout_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    report = EvaluationReport(
        baseline_ndcg_at_20=0.1,
        model_ndcg_at_20=0.2,
        model_recall_at_20=0.3,
        retrieval_recall=0.4,
        retrieval_weighted_recall=0.5,
        retrieval_oracle_ndcg_at_20=0.6,
        catalog_coverage_at_20=0.7,
        long_tail_exposure_at_20=0.8,
        paired_delta={"mean": 0.1, "lower": 0.0, "upper": 0.2},
        segments={"activity_q1": {"ndcg_at_20": 0.15}},
        candidate_budget=200,
        evaluation_users=3,
    )
    with track_command(
        "evaluate",
        PipelineConfig(run_note="hold-out de prueba", artifacts_dir=tmp_path),
        tune=False,
    ):
        log_evaluation(report)
        run_id = mlflow.active_run().info.run_id

    stored = mlflow.tracking.MlflowClient().get_run(run_id)
    assert stored.data.metrics["model_ndcg_at_20"] == 0.2
    assert stored.data.metrics["ndcg_lift_vs_popularity"] == 0.1
    assert stored.data.metrics["paired_delta.mean"] == 0.1
    assert stored.data.params["tune"] == "False"
    assert stored.data.params["command"] == "evaluate"
    assert stored.data.tags["mlflow.note.content"] == "hold-out de prueba"
    assert stored.data.tags["git_commit"]
    assert stored.data.tags["git_dirty"] in {"true", "false"}
    assert stored.info.run_name == stored.data.tags["run_name"]
    assert "-evaluate-fixed-" in stored.info.run_name


def test_run_name_marks_reused_hyperparameters(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "tuned_hyperparameters.json").write_text("{}", encoding="utf-8")

    with track_command(
        "submit",
        PipelineConfig(run_note="reuso del tuneo", artifacts_dir=artifacts),
        tune=False,
    ):
        run_name = mlflow.active_run().info.run_name

    assert "-submit-tuned-" in run_name


def test_archive_submission_keeps_each_copy(tmp_path, monkeypatch) -> None:
    import json

    from quelibroleo.tracking import active_run_name, archive_submission

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setattr(
        "quelibroleo.tracking.git_state",
        lambda: ("abc123", False),
    )
    csv_path = tmp_path / "submission.csv"
    hyperparameters = {
        "rrf_cutoff": 200,
        "source_weights": {"svd": 1.0},
        "ranker": {"learning_rate": 0.04},
    }

    csv_path.write_text("id_lector,id_libro\na,b\n", encoding="utf-8")
    with track_command(
        "submit",
        PipelineConfig(run_note="primera entrega", artifacts_dir=tmp_path),
        tune=False,
    ):
        first = active_run_name()
        first_id = mlflow.active_run().info.run_id
        archive_submission(
            csv_path,
            origin="saved",
            hyperparameters_run_name="20260923-evaluate-tune-abcd",
            hyperparameters=hyperparameters,
            note="primera entrega",
            seed=42,
        )

    csv_path.write_text("id_lector,id_libro\na,c\n", encoding="utf-8")
    with track_command(
        "submit",
        PipelineConfig(run_note="segunda entrega", artifacts_dir=tmp_path),
        tune=False,
    ):
        second = active_run_name()
        archive_submission(
            csv_path,
            origin="default",
            hyperparameters_run_name="",
            hyperparameters=hyperparameters,
            note="segunda entrega",
            seed=7,
        )

    assert first != second
    first_dir = tmp_path / "submissions" / first
    second_dir = tmp_path / "submissions" / second
    assert (first_dir / "submission.csv").read_text(encoding="utf-8").endswith("a,b\n")
    assert (second_dir / "submission.csv").read_text(encoding="utf-8").endswith("a,c\n")
    manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mlflow_run_id"] == first_id
    assert manifest["git_commit"] == "abc123"
    assert manifest["git_dirty"] is False
    assert manifest["hyperparameters_origin"] == "saved"
    assert manifest["hyperparameters_run_name"] == "20260923-evaluate-tune-abcd"
    assert manifest["hyperparameters"] == hyperparameters
    assert manifest["note"] == "primera entrega"
    assert manifest["seed"] == 42
    assert manifest["csv_sha256"]
    sidecar = json.loads((tmp_path / "submission.csv.run.json").read_text(encoding="utf-8"))
    assert sidecar["run_name"] == second
    assert sidecar["hyperparameters_origin"] == "default"
    stored = mlflow.tracking.MlflowClient().get_run(first_id)
    assert stored.data.tags["hyperparameters_run_name"] == "20260923-evaluate-tune-abcd"
    names = [
        item.path
        for item in mlflow.tracking.MlflowClient().list_artifacts(first_id, f"submissions/{first}")
    ]
    assert f"submissions/{first}/submission.csv" in names
    assert f"submissions/{first}/manifest.json" in names


@dataclass
class _Trial:
    number: int
    value: float | None
    params: dict[str, float]


def test_optuna_study_is_one_child_run(tmp_path, monkeypatch) -> None:
    from quelibroleo.tracking import optuna_callback, track_study

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    with track_command(
        "evaluate",
        PipelineConfig(run_note="study anidado"),
        tune=True,
    ):
        parent = mlflow.active_run().info.run_id
        run_name = mlflow.active_run().info.run_name
        study = type("Study", (), {"study_name": f"{run_name}__fusion-m200"})()
        callback = optuna_callback("fusion-m200")
        with track_study("fusion-m200"):
            callback(study, _Trial(0, 0.2, {"weight__svd": 1.0}))
            callback(study, _Trial(1, 0.42, {"weight__svd": 1.2}))

    runs = mlflow.tracking.MlflowClient().search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("quelibroleo").experiment_id]
    )
    client = mlflow.tracking.MlflowClient()
    nested = [run for run in runs if run.info.run_id != parent]
    assert len(nested) == 1
    assert nested[0].info.run_name == f"{run_name}__fusion-m200"
    assert nested[0].data.metrics["objective"] == 0.42
    assert nested[0].data.tags["mlflow.parentRunId"] == parent
    history = client.get_metric_history(nested[0].info.run_id, "objective")
    assert [(point.step, point.value) for point in history] == [(0, 0.2), (1, 0.42)]
