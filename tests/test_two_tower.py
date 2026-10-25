"""Two-tower como modelo único, sin tocar el tuneo de two_stage ni el de EASE."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quelibroleo.config import (
    PipelineConfig,
    SearchRange,
    SplitConfig,
    TwoTowerConfig,
    TwoTowerSearch,
)
from quelibroleo.data import DataQualityReport
from quelibroleo.models import build_model
from quelibroleo.split_data import SplitData
from quelibroleo.two_tower import load_two_tower_hyperparameters


def _frame(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["id_lector", "id_libro", "fecha", "rating"],
    ).assign(fecha=lambda frame: pd.to_datetime(frame["fecha"]))


def _split(frame: pd.DataFrame) -> SplitData:
    users = tuple(sorted(frame["id_lector"].unique()))
    return SplitData(
        holdout=True,
        books=pd.DataFrame({"id_libro": sorted(frame["id_libro"].unique())}),
        readers=pd.DataFrame({"id_lector": list(users)}),
        history=frame,
        test=frame.iloc[0:0].copy(),
        evaluation_users=users,
        submit_users=list(users),
        invalid_interactions=frame.iloc[0:0].copy(),
        quality=DataQualityReport(0, 0, 0, 0, 0, 0, 0, 0),
    )


def _config(tmp_path, **overrides: object) -> PipelineConfig:
    values = {
        "embedding_dim": 8,
        "learning_rate": 0.05,
        "n_negatives": 4,
        "temperature": 0.2,
        "max_epochs": 25,
        "patience": 5,
        "batch_size": 32,
    }
    values.update(overrides)
    return PipelineConfig(artifacts_dir=tmp_path, two_tower=TwoTowerConfig(**values))


def test_neighbor_outranks_a_rare_cooccurrence_and_skips_seen_books(tmp_path) -> None:
    rows = [("u1", "a", "2020-01-01", 8), ("u1", "b", "2020-01-02", 8)]
    for index, user in enumerate(("u2", "u3", "u4"), start=3):
        rows.append((user, "a", f"2020-02-0{index}", 8))
        rows.append((user, "b", f"2020-03-0{index}", 8))
    rows.append(("u5", "a", "2020-04-01", 8))
    rows.append(("u5", "c", "2020-04-02", 8))
    rows.append(("u6", "a", "2020-05-01", 9))
    frame = _frame(rows)
    model = build_model("two_tower", _config(tmp_path))
    model.fit(_split(frame), tune=False)

    ranking = model.recommend(["u6"])

    assert ranking["id_libro"].tolist()[0] == "b"
    assert "a" not in ranking["id_libro"].tolist()


def test_reader_without_history_follows_popularity(tmp_path) -> None:
    frame = _frame(
        [
            ("u1", "popular", "2020-01-01", 8),
            ("u2", "popular", "2020-01-02", 8),
            ("u1", "rare", "2020-01-03", 8),
        ]
    )
    model = build_model("two_tower", _config(tmp_path, max_epochs=1))
    model.fit(_split(frame), tune=False)

    ranking = model.recommend(["nuevo"])

    assert ranking["id_libro"].tolist() == ["popular", "rare"]


def test_invalid_two_tower_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "two_tower_hyperparameters.json"
    path.write_text(json.dumps({"embedding_dim": 8}), encoding="utf-8")

    with pytest.raises(ValueError, match="inválidos"):
        load_two_tower_hyperparameters(path)


def test_tune_writes_its_own_file(tmp_path) -> None:
    rows = []
    for user, start in (("u1", 0), ("u2", 2)):
        for offset in range(10):
            rows.append(
                (user, f"b{start + offset}", f"2020-01-{offset + 1:02d}", 6 + offset % 4)
            )
    frame = _frame(rows)
    two_stage = tmp_path / "tuned_hyperparameters.json"
    ease = tmp_path / "ease_hyperparameters.json"
    two_stage.write_text("{}\n", encoding="utf-8")
    ease.write_text("{}\n", encoding="utf-8")
    config = PipelineConfig(
        artifacts_dir=tmp_path,
        model="two_tower",
        optuna_trials=2,
        random_seed=0,
        split=SplitConfig(rolling_folds=2, rolling_horizon=2, min_history=1),
        two_tower=TwoTowerConfig(
            embedding_dim=4,
            learning_rate=0.05,
            n_negatives=2,
            temperature=0.2,
            max_epochs=1,
            patience=1,
            batch_size=16,
            search=TwoTowerSearch(
                embedding_dim=SearchRange(4, 8, integer=True),
                learning_rate=SearchRange(0.01, 0.05, log=True),
                n_negatives=SearchRange(2, 4, integer=True),
                temperature=SearchRange(0.1, 0.3, log=True),
            ),
        ),
    )
    model = build_model("two_tower", config)
    model.fit(_split(frame), tune=True)

    saved = json.loads((tmp_path / "two_tower_hyperparameters.json").read_text(encoding="utf-8"))
    assert two_stage.read_text(encoding="utf-8") == "{}\n"
    assert ease.read_text(encoding="utf-8") == "{}\n"
    assert saved["embedding_dim"] in {4, 5, 6, 7, 8}
    assert saved["epochs"] == 1
    assert model.hyperparameters_origin == "tune"
    assert model.hyperparameters["model"] == "two_tower"

    reused = build_model("two_tower", config)
    reused.fit(_split(frame), tune=False)
    assert reused.hyperparameters_origin == "saved"
    assert reused.hyperparameters["embedding_dim"] == saved["embedding_dim"]
    seen = set(frame.loc[frame["id_lector"] == "u1", "id_libro"])
    ranking = reused.recommend(["u1"])
    assert set(ranking["id_libro"]).isdisjoint(seen)


def test_recommend_requires_fit() -> None:
    model = build_model("two_tower", PipelineConfig())

    with pytest.raises(RuntimeError, match="entrenarse"):
        model.recommend(["u1"])
