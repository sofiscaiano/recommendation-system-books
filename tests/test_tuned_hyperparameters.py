from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from quelibroleo.config import PipelineConfig
from quelibroleo.data import USER_COL
from quelibroleo.tuning import (
    load_tuned_hyperparameters,
    record_tuned_hyperparameters,
    require_tuned_hyperparameters,
    restrict_to_example_users,
)
from quelibroleo.two_stage.columns import RankerTable


def _payload() -> dict[str, object]:
    return {
        "run_name": "20260923-evaluate-tune",
        "rrf_cutoff": 1000,
        "source_weights": {"svd": 1.4, "popularidad": 0.8},
        "ranker": {"learning_rate": 0.05, "num_leaves": 33, "max_depth": 10},
    }


def _matching_payload(config: PipelineConfig, **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_name": "20260923-evaluate-tune",
        "rrf_cutoff": 1000,
        "source_weights": {name: 1.0 for name in config.retrieval.enabled_sources()},
        "ranker": {"learning_rate": 0.05, "num_leaves": 33, "max_depth": 10},
        "label_gain": config.ranker.label_gain,
        "tune_on_example_users": config.tune_on_example_users,
    }
    payload.update(overrides)
    return payload


def test_missing_file_asks_for_tune(tmp_path) -> None:
    with pytest.raises(ValueError, match="evaluate --tune"):
        require_tuned_hyperparameters(PipelineConfig(artifacts_dir=tmp_path))


def test_saved_file_is_reused(tmp_path) -> None:
    config = PipelineConfig(artifacts_dir=tmp_path)
    path = tmp_path / "tuned_hyperparameters.json"
    path.write_text(json.dumps(_matching_payload(config)), encoding="utf-8")

    chosen = require_tuned_hyperparameters(config)

    assert chosen.rrf_cutoff == 1000
    assert set(chosen.weights) == set(config.retrieval.enabled_sources())
    assert chosen.ranker["num_leaves"] == 33
    assert chosen.run_name == "20260923-evaluate-tune"
    assert chosen.label_gain == config.ranker.label_gain
    assert chosen.tune_on_example_users is config.tune_on_example_users
    assert load_tuned_hyperparameters(path).rrf_cutoff == 1000
    history = json.loads((tmp_path / "tuning_history.json").read_text(encoding="utf-8"))
    assert len(history) == 1


def test_saved_label_gain_is_kept(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    payload = _payload()
    payload["label_gain"] = "exponential"
    path.write_text(json.dumps(payload), encoding="utf-8")

    chosen = load_tuned_hyperparameters(path)

    assert chosen.label_gain == "exponential"


def test_saved_example_population_is_kept(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    payload = _payload()
    payload["tune_on_example_users"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    chosen = load_tuned_hyperparameters(path)

    assert chosen.tune_on_example_users is True


def test_saved_example_population_must_be_boolean(tmp_path) -> None:
    path = tmp_path / "tuned_hyperparameters.json"
    payload = _payload()
    payload["tune_on_example_users"] = "si"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inválidos"):
        load_tuned_hyperparameters(path)


def test_legacy_file_with_another_population_is_not_reused(tmp_path) -> None:
    config = PipelineConfig(artifacts_dir=tmp_path)
    payload = _matching_payload(
        config,
        tune_on_example_users=not config.tune_on_example_users,
    )
    (tmp_path / "tuned_hyperparameters.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="esta config"):
        require_tuned_hyperparameters(config)


def test_history_reuses_the_matching_config_not_the_latest(tmp_path) -> None:
    base = PipelineConfig(artifacts_dir=tmp_path)
    other = replace(base, ranker=replace(base.ranker, label_gain="linear"))
    weights = {name: 1.0 for name in base.retrieval.enabled_sources()}
    ranker = {"learning_rate": 0.05, "num_leaves": 31}
    record_tuned_hyperparameters(base, rrf_cutoff=100, weights=weights, ranker=ranker)
    record_tuned_hyperparameters(other, rrf_cutoff=900, weights=weights, ranker=ranker)

    assert require_tuned_hyperparameters(base).rrf_cutoff == 100
    assert require_tuned_hyperparameters(other).rrf_cutoff == 900
    assert (
        require_tuned_hyperparameters(replace(base, run_note="otra")).rrf_cutoff == 100
    )
    with pytest.raises(ValueError, match="esta config"):
        require_tuned_hyperparameters(replace(base, optuna_trials=80))
    with pytest.raises(ValueError, match="esta config"):
        require_tuned_hyperparameters(
            replace(base, retrieval=replace(base.retrieval, rrf_k=40.0))
        )


def test_saved_ease_parameters_change_the_two_stage_config(tmp_path) -> None:
    config = PipelineConfig(artifacts_dir=tmp_path)
    weights = {name: 1.0 for name in config.retrieval.enabled_sources()}
    ranker = {"learning_rate": 0.05, "num_leaves": 31}
    record_tuned_hyperparameters(config, rrf_cutoff=100, weights=weights, ranker=ranker)

    assert require_tuned_hyperparameters(config).rrf_cutoff == 100

    (tmp_path / "ease_hyperparameters.json").write_text(
        json.dumps(
            {"reg_lambda": 12.0, "min_interactions": 30, "weight_mode": "rating"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="esta config"):
        require_tuned_hyperparameters(config)

    off = replace(config, retrieval=replace(config.retrieval, ease_source=False))
    record_tuned_hyperparameters(
        off,
        rrf_cutoff=40,
        weights={name: 1.0 for name in off.retrieval.enabled_sources()},
        ranker=ranker,
    )

    assert require_tuned_hyperparameters(off).rrf_cutoff == 40


def test_saved_source_parameters_change_the_two_stage_config(tmp_path) -> None:
    config = PipelineConfig(artifacts_dir=tmp_path)
    weights = {name: 1.0 for name in config.retrieval.enabled_sources()}
    ranker = {"learning_rate": 0.05, "num_leaves": 31}
    record_tuned_hyperparameters(config, rrf_cutoff=100, weights=weights, ranker=ranker)

    (tmp_path / "als_hyperparameters.json").write_text(
        json.dumps(
            {
                "factors": 32,
                "iterations": 8,
                "regularization": 0.2,
                "half_life_days": 90,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="esta config"):
        require_tuned_hyperparameters(config)

    (tmp_path / "two_tower_hyperparameters.json").write_text(
        json.dumps(
            {
                "embedding_dim": 32,
                "learning_rate": 0.0004,
                "temperature": 0.08,
                "min_item_reads": 3,
            }
        ),
        encoding="utf-8",
    )
    off = replace(
        config,
        retrieval=replace(config.retrieval, als_source=False, two_tower_source=False),
    )
    record_tuned_hyperparameters(
        off,
        rrf_cutoff=40,
        weights={name: 1.0 for name in off.retrieval.enabled_sources()},
        ranker=ranker,
    )

    assert require_tuned_hyperparameters(off).rrf_cutoff == 40


def test_restrict_to_example_users_keeps_delivery_readers_in_the_truth() -> None:
    table = RankerTable(
        frame=pd.DataFrame({USER_COL: ["a", "b", "a"], "score_x": [1.0, 2.0, 3.0]}),
        feature_columns=("score_x",),
    )
    target = pd.DataFrame({USER_COL: ["a", "b", "c"], "id_libro": ["1", "2", "3"]})

    kept, truth = restrict_to_example_users(table, target, {"a", "c"})

    assert list(kept.frame[USER_COL]) == ["a", "a"]
    assert kept.feature_columns == ("score_x",)
    assert set(truth[USER_COL]) == {"a", "c"}


def test_restrict_to_example_users_rejects_an_empty_fold() -> None:
    table = RankerTable(
        frame=pd.DataFrame({USER_COL: ["b"], "score_x": [1.0]}),
        feature_columns=("score_x",),
    )
    target = pd.DataFrame({USER_COL: ["b"], "id_libro": ["1"]})

    with pytest.raises(ValueError, match="ejemplo.csv"):
        restrict_to_example_users(table, target, {"a"})


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
