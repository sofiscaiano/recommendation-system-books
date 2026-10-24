from __future__ import annotations

import json

import pytest

from quelibroleo.config import PipelineConfig
from quelibroleo.tuning import load_tuned_hyperparameters, resolve_saved_or_default


def _payload() -> dict[str, object]:
    return {
        "run_name": "20260923-evaluate-tune",
        "rrf_cutoff": 1000,
        "source_weights": {"svd": 1.4, "popularidad": 0.8},
        "ranker": {"learning_rate": 0.05, "num_leaves": 33, "max_depth": 10},
    }


def test_missing_file_uses_fixed_config(tmp_path) -> None:
    chosen, origin = resolve_saved_or_default(
        PipelineConfig(artifacts_dir=tmp_path)
    )

    assert origin == "default"
    assert chosen.rrf_cutoff == 200
    assert chosen.ranker == {}
    assert chosen.weights["popularidad"] == 0.5


def test_saved_file_is_reused(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    chosen, origin = resolve_saved_or_default(
        PipelineConfig(artifacts_dir=tmp_path)
    )

    assert origin == "saved"
    assert chosen.rrf_cutoff == 1000
    assert chosen.weights == {"svd": 1.4, "popularidad": 0.8}
    assert chosen.ranker["num_leaves"] == 33
    assert chosen.run_name == "20260923-evaluate-tune"
    assert chosen.label_gain == "linear"
    assert load_tuned_hyperparameters(path).rrf_cutoff == 1000


def test_saved_label_gain_is_kept(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    payload = _payload()
    payload["label_gain"] = "exponential"
    path.write_text(json.dumps(payload), encoding="utf-8")

    chosen = load_tuned_hyperparameters(path)

    assert chosen.label_gain == "exponential"


def test_saved_label_gain_must_be_known(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    payload = _payload()
    payload["label_gain"] = "identity"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inválidos"):
        load_tuned_hyperparameters(path)


def test_quota_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    payload = _payload()
    payload["source_quotas"] = {"svd": 20, "popularidad": 70}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cuotas"):
        load_tuned_hyperparameters(path)


def test_invalid_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="inválidos"):
        load_tuned_hyperparameters(path)
