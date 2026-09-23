"""Registro de métricas sin correr el pipeline completo."""

from dataclasses import dataclass

import mlflow

from quelibroleo.config import PipelineConfig
from quelibroleo.pipeline import EvaluationReport
from quelibroleo.tracking import log_evaluation, track_command


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
    with track_command("evaluate", PipelineConfig(), tune=False):
        log_evaluation(report)
        run_id = mlflow.active_run().info.run_id

    stored = mlflow.tracking.MlflowClient().get_run(run_id)
    assert stored.data.metrics["model_ndcg_at_20"] == 0.2
    assert stored.data.metrics["ndcg_lift_vs_popularity"] == 0.1
    assert stored.data.metrics["paired_delta.mean"] == 0.1
    assert stored.data.params["tune"] == "False"
    assert stored.data.params["command"] == "evaluate"
    assert "Primera corrida" in stored.data.tags["mlflow.note.content"]
    assert stored.data.tags["git_commit"]


@dataclass
class _Trial:
    number: int
    value: float | None
    params: dict[str, float]


def test_optuna_callback_nests_trial(tmp_path, monkeypatch) -> None:
    from quelibroleo.tracking import optuna_callback

    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    study = type("Study", (), {"study_name": "fusion"})()
    with track_command("evaluate", PipelineConfig(), tune=True):
        parent = mlflow.active_run().info.run_id
        optuna_callback("fusion-m200")(study, _Trial(3, 0.42, {"weight__svd": 1.2}))

    runs = mlflow.tracking.MlflowClient().search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("quelibroleo").experiment_id]
    )
    client = mlflow.tracking.MlflowClient()
    nested = [run for run in runs if run.info.run_id != parent]
    assert len(nested) == 1
    assert nested[0].data.metrics["objective"] == 0.42
    assert nested[0].data.tags["mlflow.parentRunId"] == parent
    history = client.get_metric_history(parent, "fusion-m200.objective")
    assert [(point.step, point.value) for point in history] == [(3, 0.42)]
