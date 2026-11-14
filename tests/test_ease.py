from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from quelibroleo.config import (
    EaseConfig,
    EaseSearch,
    PipelineConfig,
    SearchRange,
    SplitConfig,
)
from quelibroleo.data import DataQualityReport
from quelibroleo.ease import (
    interaction_matrix,
    load_ease_hyperparameters,
    score_users,
    spectrum_from_matrix,
)
from quelibroleo.models import build_model
from quelibroleo.pipeline import clear_foreign_diagnostics
from quelibroleo.split_data import SplitData


def _frame(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["id_lector", "id_libro", "fecha", "rating"],
    ).assign(fecha=lambda frame: pd.to_datetime(frame["fecha"]))


def _books(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"id_libro": sorted(frame["id_libro"].unique())})


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


def _closed_form(matrix: sparse.csr_matrix, reg_lambda: float) -> np.ndarray:
    gram = (matrix.T @ matrix).toarray()
    gram = (gram + gram.T) * 0.5
    precision = np.linalg.inv(gram + reg_lambda * np.eye(gram.shape[0]))
    weights = np.eye(gram.shape[0]) - precision / np.diag(precision)
    return matrix.toarray() @ weights


def test_scores_match_the_closed_form() -> None:
    matrix = sparse.csr_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 0, 0, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.float64,
        )
    )
    item_ids = np.array(["a", "b", "c", "d"], dtype=object)
    spectrum = spectrum_from_matrix(matrix, item_ids)

    scores = score_users(matrix, spectrum, reg_lambda=5.0)

    assert scores == pytest.approx(_closed_form(matrix, 5.0), rel=1e-6, abs=1e-6)


def test_neighbor_outranks_a_rare_cooccurrence_and_skips_seen_books(tmp_path) -> None:
    rows = [("u1", "a", "2020-01-01", 8), ("u1", "b", "2020-01-02", 8)]
    for index, user in enumerate(("u2", "u3", "u4"), start=3):
        rows.append((user, "a", f"2020-02-0{index}", 8))
        rows.append((user, "b", f"2020-03-0{index}", 8))
    rows.append(("u5", "a", "2020-04-01", 8))
    rows.append(("u5", "c", "2020-04-02", 8))
    rows.append(("u6", "a", "2020-05-01", 9))
    frame = _frame(rows)
    config = PipelineConfig(
        artifacts_dir=tmp_path,
        ease=EaseConfig(reg_lambda=1.0, min_interactions=1),
    )
    model = build_model("ease", config)
    model.fit(_split(frame), tune=False)

    ranking = model.recommend(["u6"])
    top = ranking["id_libro"].tolist()

    assert top[0] == "b"
    assert "a" not in top
    assert model._spectrum is not None
    assert set(model._spectrum.item_ids) == {"a", "b", "c"}


def test_reader_without_history_follows_popularity(tmp_path) -> None:
    frame = _frame(
        [
            ("u1", "popular", "2020-01-01", 8),
            ("u2", "popular", "2020-01-02", 8),
            ("u1", "rare", "2020-01-03", 8),
        ]
    )
    config = PipelineConfig(
        artifacts_dir=tmp_path,
        ease=EaseConfig(reg_lambda=10.0, min_interactions=1),
    )
    model = build_model("ease", config)
    model.fit(_split(frame), tune=False)

    ranking = model.recommend(["nuevo"])

    assert ranking["id_libro"].tolist() == ["popular", "rare"]


def test_min_interactions_drops_rare_books() -> None:
    frame = _frame(
        [
            ("u1", "frequent", "2020-01-01", 8),
            ("u2", "frequent", "2020-01-02", 8),
            ("u1", "rare", "2020-01-03", 5),
        ]
    )

    _, items = interaction_matrix(frame, min_interactions=2)

    assert items.tolist() == ["frequent"]


def test_recent_read_outranks_an_equally_rated_old_one(tmp_path) -> None:
    rows = []
    for user, day in (("u2", 1), ("u3", 2), ("u4", 3)):
        rows.append((user, "old", f"2010-01-0{day}", 8))
        rows.append((user, "from_old", f"2010-02-0{day}", 8))
        rows.append((user, "recent", f"2024-01-0{day}", 8))
        rows.append((user, "from_recent", f"2024-02-0{day}", 8))
    rows.append(("u1", "old", "2010-06-01", 8))
    rows.append(("u1", "recent", "2024-06-01", 8))
    frame = _frame(rows)
    config = PipelineConfig(
        artifacts_dir=tmp_path,
        ease=EaseConfig(reg_lambda=1.0, min_interactions=1),
    )
    model = build_model("ease", config)
    model.fit(_split(frame))
    ranking = model.recommend(["u1"])

    assert ranking["id_libro"].tolist()[0] == "from_recent"


def test_gram_uses_the_rating_and_not_the_date() -> None:
    early = _frame([("u1", "a", "2010-01-01", 10), ("u1", "b", "2010-01-02", 2)])
    late = _frame([("u1", "a", "2024-01-01", 10), ("u1", "b", "2024-06-01", 2)])

    early_matrix, _ = interaction_matrix(early, min_interactions=1)
    late_matrix, _ = interaction_matrix(late, min_interactions=1)

    assert early_matrix.toarray() == pytest.approx(late_matrix.toarray())
    assert early_matrix.sum() == pytest.approx(12)


def test_dense_item_cap_rejects_an_oversized_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("quelibroleo.ease.MAX_ITEMS", 1)
    frame = _frame(
        [
            ("u1", "a", "2020-01-01", 8),
            ("u1", "b", "2020-01-02", 8),
        ]
    )

    with pytest.raises(ValueError, match="densa"):
        interaction_matrix(frame, min_interactions=1)


def test_invalid_ease_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "ease_hyperparameters.json"
    path.write_text(json.dumps({"reg_lambda": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="inválidos"):
        load_ease_hyperparameters(path)


def test_tune_writes_ease_file_and_keeps_two_stage_file(tmp_path) -> None:
    rows = []
    for user, start in (("u1", 0), ("u2", 2)):
        for offset in range(10):
            rows.append(
                (
                    user,
                    f"b{start + offset}",
                    f"2020-01-{offset + 1:02d}",
                    6 + offset % 4,
                )
            )
    frame = _frame(rows)
    two_stage = tmp_path / "tuned_hyperparameters.json"
    two_stage.write_text("{}\n", encoding="utf-8")
    config = PipelineConfig(
        artifacts_dir=tmp_path,
        model="ease",
        optuna_trials=3,
        random_seed=0,
        tune_on_example_users=False,
        split=SplitConfig(rolling_folds=2, rolling_horizon=2, min_history=1),
        ease=EaseConfig(
            reg_lambda=10.0,
            min_interactions=1,
            search=EaseSearch(
                reg_lambda=SearchRange(1.0, 20.0, log=True),
                min_interactions=SearchRange(1, 2, integer=True),
            ),
        ),
    )
    model = build_model("ease", config)
    model.fit(_split(frame), tune=True)

    saved = json.loads(
        (tmp_path / "ease_hyperparameters.json").read_text(encoding="utf-8")
    )
    assert two_stage.read_text(encoding="utf-8") == "{}\n"
    assert saved["min_interactions"] in {1, 2}
    assert saved["reg_lambda"] > 0
    assert "weight_mode" not in saved
    assert model.hyperparameters_origin == "tune"
    assert model.hyperparameters["model"] == "ease"

    reused = build_model("ease", config)
    reused.fit(_split(frame), tune=False)
    assert reused.hyperparameters_origin == "saved"
    assert reused.hyperparameters["reg_lambda"] == saved["reg_lambda"]
    ranking = reused.recommend(["u1"])
    seen = set(frame.loc[frame["id_lector"] == "u1", "id_libro"])
    assert ranking["id_libro"].tolist()
    assert set(ranking["id_libro"]).isdisjoint(seen)


def test_tune_on_example_users_requires_those_readers(tmp_path) -> None:
    rows = []
    for user, start in (("u1", 0), ("u2", 2)):
        for offset in range(10):
            rows.append(
                (
                    user,
                    f"b{start + offset}",
                    f"2020-01-{offset + 1:02d}",
                    6 + offset % 4,
                )
            )
    config = PipelineConfig(
        artifacts_dir=tmp_path,
        model="ease",
        optuna_trials=1,
        split=SplitConfig(rolling_folds=2, rolling_horizon=2, min_history=1),
        ease=EaseConfig(reg_lambda=1.0, min_interactions=1),
    )
    model = build_model("ease", config)

    with pytest.raises(ValueError, match="ejemplo.csv"):
        model.fit(_split(_frame(rows)), tune=True)


def test_recommend_requires_fit() -> None:
    model = build_model("ease", PipelineConfig())

    with pytest.raises(RuntimeError, match="entrenarse"):
        model.recommend(["u1"])


def test_foreign_diagnostics_keep_the_two_stage_file(tmp_path) -> None:
    tuned = tmp_path / "tuned_hyperparameters.json"
    folds = tmp_path / "retrieval_folds.csv"
    tuned.write_text("{}", encoding="utf-8")
    folds.write_text("fold\n", encoding="utf-8")

    clear_foreign_diagnostics(tmp_path, "ease")

    assert tuned.is_file()
    assert not folds.exists()
    folds.write_text("fold\n", encoding="utf-8")
    clear_foreign_diagnostics(tmp_path, "two_stage")
    assert folds.is_file()
