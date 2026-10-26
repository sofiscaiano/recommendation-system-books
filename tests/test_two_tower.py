"""Two-tower con producto y la red única, sin Optuna."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from quelibroleo.config import (
    PipelineConfig,
    RatingMlpConfig,
    SplitConfig,
    TwoTowerConfig,
)
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


def test_submit_resumes_saved_weights_and_keeps_known_books(tmp_path, monkeypatch) -> None:
    rows = [("u1", f"b{offset}", f"2020-01-{offset + 1:02d}", 8) for offset in range(25)]
    frame = _frame(rows)
    calls: list[str] = []
    real = __import__("quelibroleo.neural", fromlist=["_run_epochs"])._run_epochs

    def spy(*args, **kwargs):
        calls.append(kwargs["phase"])
        return real(*args, **kwargs)

    monkeypatch.setattr("quelibroleo.neural._run_epochs", spy)
    config = _config(tmp_path, "two_tower", max_epochs=1, finetune_epochs=0)
    first = build_model("two_tower", config)
    first.fit(_split(frame.iloc[:5].reset_index(drop=True), holdout=True), tune=False)
    assert calls == ["época"]
    assert (tmp_path / "two_tower.pt").is_file()
    kept = first._net.book.weight.detach().cpu().clone()
    known = first._catalog.item_index["b0"]

    calls.clear()
    config = _config(tmp_path, "two_tower", max_epochs=1, finetune_epochs=0)
    second = build_model("two_tower", config)
    second.fit(_split(frame, holdout=False), tune=False)

    assert calls == []
    resumed = second._net.book.weight.detach().cpu()
    assert torch.allclose(resumed[second._catalog.item_index["b0"]], kept[known])
    assert second.hyperparameters["n_items"] == 25
    assert "b24" in second._catalog.item_index


def test_preholdout_drops_the_last_twenty() -> None:
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


def test_last_reads_keep_short_users_and_the_recent_tail() -> None:
    from quelibroleo.neural import last_reads_for_validation

    rows = [("u1", f"b{offset}", f"2020-01-{offset + 1:02d}", 8) for offset in range(8)]
    rows += [("u2", f"c{offset}", f"2020-02-{offset + 1:02d}", 6) for offset in range(5)]
    rows += [("u3", "solo", "2020-03-01", 7)]
    train, val = last_reads_for_validation(_frame(rows), 5)

    assert set(val["id_lector"]) == {"u1"}
    assert set(val["id_libro"]) == {f"b{offset}" for offset in range(3, 8)}
    assert set(train["id_libro"]) >= {f"c{offset}" for offset in range(5)} | {"solo", "b0", "b1", "b2"}
    assert set(train["id_libro"]).isdisjoint(set(val["id_libro"]))


def test_validation_rmse_is_on_the_rating_scale() -> None:
    from quelibroleo.neural import _validation_loss

    class _Half(torch.nn.Module):
        def forward(self, users: torch.Tensor, _items: torch.Tensor) -> torch.Tensor:
            return torch.full((users.shape[0],), 0.5)

    rows = (
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
        np.array([1.0, 0.0], dtype=np.float32),
    )

    rmse = _validation_loss(_Half(), rows, batch_size=1, device=torch.device("cpu"))

    assert rmse == pytest.approx(4.5)


def test_validation_split_rejects_a_non_positive_k() -> None:
    with pytest.raises(ValueError, match="validation_k"):
        SplitConfig(validation_k=0)


def test_run_epochs_restores_the_lowest_validation_loss(monkeypatch) -> None:
    from quelibroleo.neural import TrainSchedule, _run_epochs

    net = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    schedule = TrainSchedule(0.01, 4, 3, 0)
    rows = (
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([0.5], dtype=np.float32),
    )
    state = {"epoch": 0}
    planned = {1: 0.4, 2: 0.1, 3: 0.3}

    def fake_epoch(module, *_args, **_kwargs):
        state["epoch"] += 1
        with torch.no_grad():
            module.bias.fill_(state["epoch"])
        return 0.2

    def fake_val(module, *_args, **_kwargs):
        return planned[int(module.bias.detach())]

    monkeypatch.setattr("quelibroleo.neural._epoch", fake_epoch)
    monkeypatch.setattr("quelibroleo.neural._validation_loss", fake_val)

    restored = _run_epochs(
        net,
        rows,
        optimizer,
        schedule,
        np.random.default_rng(0),
        torch.device("cpu"),
        epochs=3,
        model_name="rating_mlp",
        phase="época",
        val_rows=rows,
    )

    assert restored
    assert int(net.bias.detach()) == 2


def test_main_pass_holds_out_last_five_and_finetune_sees_the_tail(tmp_path, monkeypatch) -> None:
    rows = [("u1", f"b{offset}", f"2020-01-{offset + 1:02d}", 8) for offset in range(30)]
    frame = _frame(rows)
    seen: list[tuple[str, int, int]] = []
    real = __import__("quelibroleo.neural", fromlist=["_run_epochs"])._run_epochs

    def spy(*args, **kwargs):
        val_rows = kwargs.get("val_rows")
        seen.append((kwargs["phase"], len(args[1][0]), 0 if val_rows is None else len(val_rows[0])))
        return real(*args, **kwargs)

    monkeypatch.setattr("quelibroleo.neural._run_epochs", spy)
    config = _config(tmp_path, "rating_mlp", max_epochs=1, finetune_epochs=1)
    fitted = build_model("rating_mlp", config)
    fitted.fit(_split(frame, holdout=False), tune=False)

    assert seen == [("época", 5, 5), ("continuación", 30, 0)]
