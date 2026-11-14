from __future__ import annotations

import json

import pandas as pd
import pytest

from quelibroleo import two_tower as two_tower_module
from quelibroleo.config import (
    AlsConfig,
    AlsSearch,
    BprConfig,
    BprSearch,
    ItemCosineConfig,
    ItemCosineSearch,
    PipelineConfig,
    SearchRange,
    SplitConfig,
    TfidfConfig,
    TfidfSearch,
    TwoTowerConfig,
    TwoTowerSearch,
)
from quelibroleo.data import DataQualityReport
from quelibroleo.models import build_model
from quelibroleo.source_models import (
    load_source_hyperparameters,
    load_two_tower_hyperparameters,
)
from quelibroleo.split_data import SplitData


def _frame(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["id_lector", "id_libro", "fecha", "rating"],
    ).assign(fecha=lambda frame: pd.to_datetime(frame["fecha"]))


def _books(frame: pd.DataFrame) -> pd.DataFrame:
    items = sorted(frame["id_libro"].unique())
    return pd.DataFrame(
        {
            "id_libro": items,
            "titulo": [f"titulo {item}" for item in items],
            "autor": [f"autor {index % 3}" for index, _item in enumerate(items)],
            "genero": ["novela"] * len(items),
            "editorial": ["editorial"] * len(items),
            "resumen": ["un resumen corto"] * len(items),
        }
    )


def _split(frame: pd.DataFrame) -> SplitData:
    users = tuple(sorted(frame["id_lector"].unique()))
    return SplitData(
        holdout=True,
        books=_books(frame),
        readers=pd.DataFrame({"id_lector": list(users)}),
        history=frame,
        test=frame.iloc[0:0].copy(),
        evaluation_users=users,
        submit_users=list(users),
        invalid_interactions=frame.iloc[0:0].copy(),
        quality=DataQualityReport(0, 0, 0, 0, 0, 0, 0, 0),
    )


def _history() -> pd.DataFrame:
    rows = []
    for user, start in (("u1", 0), ("u2", 2), ("u3", 4)):
        for offset in range(10):
            rows.append(
                (
                    user,
                    f"b{start + offset}",
                    f"2020-01-{offset + 1:02d}",
                    6 + offset % 4,
                )
            )
    return _frame(rows)


def _config(tmp_path, model: str) -> PipelineConfig:
    narrow = {
        "item_cosine": {
            "item_cosine": ItemCosineConfig(
                half_life_days=40,
                search=ItemCosineSearch(half_life_days=SearchRange(30, 50)),
            )
        },
        "als": {
            "als": AlsConfig(
                factors=4,
                iterations=2,
                regularization=0.05,
                half_life_days=40,
                search=AlsSearch(
                    factors=SearchRange(2, 4, integer=True),
                    iterations=SearchRange(1, 2, integer=True),
                    regularization=SearchRange(0.01, 0.05),
                    half_life_days=SearchRange(30, 50),
                ),
            )
        },
        "bpr": {
            "bpr": BprConfig(
                factors=4,
                iterations=2,
                learning_rate=0.05,
                regularization=0.05,
                half_life_days=40,
                search=BprSearch(
                    factors=SearchRange(2, 4, integer=True),
                    iterations=SearchRange(1, 2, integer=True),
                    learning_rate=SearchRange(0.01, 0.05),
                    regularization=SearchRange(0.01, 0.05),
                    half_life_days=SearchRange(30, 50),
                ),
            )
        },
        "tfidf": {
            "tfidf": TfidfConfig(
                max_features=16,
                half_life_days=40,
                search=TfidfSearch(
                    max_features=SearchRange(8, 16, integer=True),
                    half_life_days=SearchRange(30, 50),
                ),
            )
        },
    }
    return PipelineConfig(
        artifacts_dir=tmp_path,
        model=model,
        optuna_trials=2,
        random_seed=0,
        tune_on_example_users=False,
        split=SplitConfig(rolling_folds=2, rolling_horizon=2, min_history=1),
        **narrow[model],
    )


@pytest.mark.parametrize("model", ["item_cosine", "als", "bpr", "tfidf"])
def test_tune_writes_only_that_source_file(tmp_path, model: str) -> None:
    two_stage = tmp_path / "tuned_hyperparameters.json"
    other = tmp_path / "ease_hyperparameters.json"
    two_stage.write_text("{}\n", encoding="utf-8")
    other.write_text("{}\n", encoding="utf-8")
    frame = _history()
    fitted = build_model(model, _config(tmp_path, model))
    fitted.fit(_split(frame), tune=True)

    saved = json.loads(
        (tmp_path / f"{model}_hyperparameters.json").read_text(encoding="utf-8")
    )
    assert two_stage.read_text(encoding="utf-8") == "{}\n"
    assert other.read_text(encoding="utf-8") == "{}\n"
    assert saved
    assert fitted.hyperparameters_origin == "tune"
    assert fitted.hyperparameters["model"] == model

    reused = build_model(model, _config(tmp_path, model))
    reused.fit(_split(frame), tune=False)
    assert reused.hyperparameters_origin == "saved"
    ranking = reused.recommend(["u1"])
    seen = set(frame.loc[frame["id_lector"] == "u1", "id_libro"])
    assert ranking["id_libro"].tolist()
    assert set(ranking["id_libro"]).isdisjoint(seen)
    assert len(ranking) <= 20


def test_reader_without_history_follows_popularity(tmp_path) -> None:
    frame = _frame(
        [
            ("u1", "popular", "2020-01-01", 8),
            ("u2", "popular", "2020-01-02", 8),
            ("u1", "rare", "2020-01-03", 8),
        ]
    )
    model = build_model("item_cosine", PipelineConfig(artifacts_dir=tmp_path))
    model.fit(_split(frame), tune=False)

    ranking = model.recommend(["nuevo"])

    assert ranking["id_libro"].tolist() == ["popular", "rare"]


def test_recommend_requires_fit() -> None:
    model = build_model("als", PipelineConfig())

    with pytest.raises(RuntimeError, match="entrenarse"):
        model.recommend(["u1"])


def test_invalid_source_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "als_hyperparameters.json"
    path.write_text(json.dumps({"factors": 4}), encoding="utf-8")

    with pytest.raises(ValueError, match="inválidos"):
        load_source_hyperparameters(path, "als")


def test_invalid_two_tower_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "two_tower_hyperparameters.json"
    path.write_text(json.dumps({"embedding_dim": 32}), encoding="utf-8")

    with pytest.raises(ValueError, match="inválidos"):
        load_two_tower_hyperparameters(path)


def test_two_tower_tune_writes_its_file_and_keeps_two_stage(
    tmp_path, monkeypatch
) -> None:
    rows = [
        (
            f"u{user}",
            f"b{(user + offset) % 8}",
            pd.Timestamp("2020-01-01") + pd.Timedelta(days=offset),
            1 + offset % 10,
        )
        for user in range(6)
        for offset in range(8)
    ]
    frame = pd.DataFrame(rows, columns=["id_lector", "id_libro", "fecha", "rating"])
    two_stage = tmp_path / "tuned_hyperparameters.json"
    two_stage.write_text("{}\n", encoding="utf-8")
    config = PipelineConfig(
        artifacts_dir=tmp_path,
        model="two_tower",
        optuna_trials=1,
        random_seed=0,
        tune_on_example_users=False,
        split=SplitConfig(rolling_folds=2, rolling_horizon=2, min_history=2),
        two_tower=TwoTowerConfig(
            embedding_dim=4,
            batch_size=8,
            max_epochs=1,
            patience=1,
            min_item_reads=1,
            search=TwoTowerSearch(
                embedding_dim=SearchRange(4, 8, integer=True),
                learning_rate=SearchRange(1e-3, 2e-3),
                temperature=SearchRange(0.05, 0.08),
                min_item_reads=SearchRange(1, 2, integer=True),
            ),
        ),
    )
    model = build_model("two_tower", config)
    calls: list[bool] = []
    real_train = two_tower_module.train

    def counted(*args, **kwargs):
        calls.append(kwargs.get("validation") is not None)
        return real_train(*args, **kwargs)

    monkeypatch.setattr(two_tower_module, "train", counted)
    model.fit(_split(frame), tune=True)

    saved = json.loads(
        (tmp_path / "two_tower_hyperparameters.json").read_text(encoding="utf-8")
    )
    assert two_stage.read_text(encoding="utf-8") == "{}\n"
    assert 4 <= saved["embedding_dim"] <= 8
    assert saved["min_item_reads"] in {1, 2}
    assert model.hyperparameters_origin == "tune"
    assert "search" not in model.hyperparameters
    assert calls == [True, False]
    assert model.hyperparameters["epochs"] == 1

    reused = build_model("two_tower", config)
    reused.fit(_split(frame), tune=False)
    assert reused.hyperparameters_origin == "saved"
    assert reused.hyperparameters["embedding_dim"] == saved["embedding_dim"]
