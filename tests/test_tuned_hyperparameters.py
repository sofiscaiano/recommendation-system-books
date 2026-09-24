from __future__ import annotations

import json

import pytest

from quelibroleo.config import PipelineConfig
from quelibroleo.tuning import load_tuned_hyperparameters, resolve_saved_or_default


def _payload() -> dict[str, object]:
    return {
        "run_name": "20260923-evaluate-tune",
        "candidate_budget": 1000,
        "source_weights": {"svd": 1.4, "popularidad": 0.8},
        "source_quotas": {"svd": 20, "popularidad": 70},
        "ranker": {"learning_rate": 0.05, "num_leaves": 33, "max_depth": 10},
    }


def test_missing_file_uses_fixed_config(tmp_path) -> None:
    chosen, origin = resolve_saved_or_default(
        PipelineConfig(artifacts_dir=tmp_path)
    )

    assert origin == "default"
    assert chosen.budget == 200
    assert chosen.ranker == {}
    assert chosen.weights["popularidad"] == 0.5


def test_saved_file_is_reused(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    chosen, origin = resolve_saved_or_default(
        PipelineConfig(artifacts_dir=tmp_path)
    )

    assert origin == "saved"
    assert chosen.budget == 1000
    assert chosen.weights == {"svd": 1.4, "popularidad": 0.8}
    assert chosen.quotas == {"svd": 20, "popularidad": 70}
    assert chosen.ranker["num_leaves"] == 33
    assert chosen.run_name == "20260923-evaluate-tune"
    assert load_tuned_hyperparameters(path).budget == 1000


def test_invalid_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="inválidos"):
        load_tuned_hyperparameters(path)
