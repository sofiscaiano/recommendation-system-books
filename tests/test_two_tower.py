"""Two-tower con producto y la red única, sin Optuna."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from quelibroleo.config import PipelineConfig, RatingMlpConfig, TwoTowerConfig
from quelibroleo.data import DataQualityReport
from quelibroleo.models import build_model
from quelibroleo.split_data import SplitData


@pytest.fixture(autouse=True)
def _cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "quelibroleo.neural.pick_device",
        lambda: torch.device("cpu"),
    )


def _frame(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["id_lector", "id_libro", "fecha", "rating"],
    ).assign(fecha=lambda frame: pd.to_datetime(frame["fecha"]))


def _split(
    frame: pd.DataFrame,
    books: pd.DataFrame | None = None,
    *,
    holdout: bool = True,
) -> SplitData:
    users = tuple(sorted(frame["id_lector"].unique()))
    if books is None:
        books = pd.DataFrame({"id_libro": sorted(frame["id_libro"].unique())})
    return SplitData(
        holdout=holdout,
        books=books,
        readers=pd.DataFrame({"id_lector": list(users)}),
        history=frame,
        test=frame.iloc[0:0].copy(),
        evaluation_users=users,
        submit_users=list(users),
        invalid_interactions=frame.iloc[0:0].copy(),
        quality=DataQualityReport(0, 0, 0, 0, 0, 0, 0, 0),
    )


def _config(tmp_path, model: str, **overrides: object) -> PipelineConfig:
    values = {
        "embedding_dim": 8,
        "learning_rate": 0.05,
        "max_epochs": 40,
        "finetune_epochs": 0,
        "batch_size": 8,
    }
    values.update(overrides)
    if model == "rating_mlp":
        block = RatingMlpConfig(hidden_1=8, hidden_2=4, **values)
        return PipelineConfig(artifacts_dir=tmp_path, model=model, rating_mlp=block)
    block = TwoTowerConfig(hidden_dim=8, output_dim=4, **values)
    return PipelineConfig(artifacts_dir=tmp_path, model=model, two_tower=block)


def _neighbor_frame() -> pd.DataFrame:
    rows = [("u1", "a", "2020-01-01", 9), ("u1", "b", "2020-01-02", 9)]
    for index, user in enumerate(("u2", "u3", "u4", "u7", "u8"), start=3):
        day = index % 9
        rows.append((user, "a", f"2020-02-0{day}", 9))
        rows.append((user, "b", f"2020-03-0{day}", 9))
    rows.append(("u5", "a", "2020-04-01", 9))
    rows.append(("u5", "c", "2020-04-02", 9))
    rows.append(("u6", "a", "2020-05-01", 9))
    return _frame(rows)


@pytest.mark.parametrize("model", ["two_tower", "rating_mlp"])
def test_neighbor_outranks_a_rare_cooccurrence_and_skips_seen_books(tmp_path, model: str) -> None:
    fitted = build_model(model, _config(tmp_path, model))
    fitted.fit(_split(_neighbor_frame()), tune=False)

    ranking = fitted.recommend(["u6"])

    assert ranking["id_libro"].tolist()[0] == "b"
    assert "a" not in ranking["id_libro"].tolist()


@pytest.mark.parametrize("model", ["two_tower", "rating_mlp"])
def test_reader_without_history_follows_popularity(tmp_path, model: str) -> None:
    frame = _frame(
        [
            ("u1", "popular", "2020-01-01", 8),
            ("u2", "popular", "2020-01-02", 8),
            ("u1", "rare", "2020-01-03", 8),
        ]
    )
    fitted = build_model(model, _config(tmp_path, model, max_epochs=1))
    fitted.fit(_split(frame), tune=False)

    ranking = fitted.recommend(["nuevo"])

    assert ranking["id_libro"].tolist() == ["popular", "rare"]


@pytest.mark.parametrize("model", ["two_tower", "rating_mlp"])
def test_tune_does_not_write_a_study_or_touch_other_files(tmp_path, model: str) -> None:
    stale = tmp_path / "two_tower_hyperparameters.json"
    two_stage = tmp_path / "tuned_hyperparameters.json"
    ease = tmp_path / "ease_hyperparameters.json"
    stale.write_text("{}\n", encoding="utf-8")
    two_stage.write_text("{}\n", encoding="utf-8")
    ease.write_text("{}\n", encoding="utf-8")
    frame = _neighbor_frame()
    config = _config(tmp_path, model, max_epochs=1)
    fitted = build_model(model, config)
    fitted.fit(_split(frame), tune=True)

    assert stale.read_text(encoding="utf-8") == "{}\n"
    assert two_stage.read_text(encoding="utf-8") == "{}\n"
    assert ease.read_text(encoding="utf-8") == "{}\n"
    assert not (tmp_path / "rating_mlp_hyperparameters.json").exists()
    assert fitted.hyperparameters_origin == "config"
    assert fitted.hyperparameters["model"] == model
    seen = set(frame.loc[frame["id_lector"] == "u1", "id_libro"])
    ranking = fitted.recommend(["u1"])
    assert set(ranking["id_libro"]).isdisjoint(seen)


@pytest.mark.parametrize("model", ["two_tower", "rating_mlp"])
def test_recommend_requires_fit(model: str) -> None:
    fitted = build_model(model, PipelineConfig())

    with pytest.raises(RuntimeError, match="entrenarse"):
        fitted.recommend(["u1"])


def test_author_and_genre_get_their_own_rows(tmp_path) -> None:
    frame = _frame(
        [
            ("u1", "a", "2020-01-01", 8),
            ("u1", "b", "2020-01-02", 7),
        ]
    )
    books = pd.DataFrame(
        {
            "id_libro": ["a", "b"],
            "autor": ["Autora", ""],
            "genero": ["Novela negra, intriga, terror", ""],
        }
    )
    fitted = build_model("rating_mlp", _config(tmp_path, "rating_mlp", max_epochs=1))
    fitted.fit(_split(frame, books), tune=False)

    assert fitted.hyperparameters["n_authors"] == 2
    assert fitted.hyperparameters["n_genres"] == 2


def test_submit_trains_without_the_holdout_and_then_adds_it() -> None:
    rows = [("u1", f"b{offset}", f"2020-01-{offset + 1:02d}", 8) for offset in range(25)]
    frame = _frame(rows)

    from quelibroleo.neural import preholdout

    earlier = preholdout(frame, 20)

    assert earlier is not None
    assert len(earlier) == 5
    assert set(earlier["id_libro"]) == {f"b{offset}" for offset in range(5)}


def test_submit_fit_keeps_training_after_the_holdout_joins(tmp_path) -> None:
    rows = [("u1", f"b{offset}", f"2020-01-{offset + 1:02d}", 8) for offset in range(25)]
    frame = _frame(rows)
    config = _config(tmp_path, "two_tower", max_epochs=1, finetune_epochs=1)
    fitted = build_model("two_tower", config)
    fitted.fit(_split(frame, holdout=False), tune=False)

    seen = set(frame["id_libro"])
    ranking = fitted.recommend(["u1"])
    assert set(ranking["id_libro"]).isdisjoint(seen)
    assert fitted.hyperparameters["finetune_epochs"] == 1
