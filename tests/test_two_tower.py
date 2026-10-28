"""Two-tower: ejemplos sin futuro, pérdida in-batch y contrato de fit/recommend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from quelibroleo.config import PipelineConfig, SplitConfig, TwoTowerConfig
from quelibroleo.data import DataQualityReport
from quelibroleo.models import build_model
from quelibroleo.split_data import SplitData
from quelibroleo.two_tower import (
    TwoTowerNet,
    build_sequences,
    build_vocabulary,
    history_bags,
    in_batch_loss,
    sample_examples,
)


def _history(n_users: int = 12, n_reads: int = 12) -> pd.DataFrame:
    rows = [
        (f"u{user}", f"b{(user + offset) % 15}", pd.Timestamp("2020-01-01") + pd.Timedelta(days=offset), 1 + offset % 10)
        for user in range(n_users)
        for offset in range(n_reads)
    ]
    return pd.DataFrame(rows, columns=["id_lector", "id_libro", "fecha", "rating"])


def _books(history: pd.DataFrame) -> pd.DataFrame:
    items = sorted(history["id_libro"].unique())
    return pd.DataFrame(
        {
            "id_libro": items,
            "autor": [f"a{index % 4}" for index in range(len(items))],
            "genero": ["" if index % 5 == 0 else "novela" for index in range(len(items))],
        }
    )


def test_examples_take_history_before_the_cut_and_target_inside_the_window() -> None:
    history = _history(n_users=3, n_reads=30)
    vocabulary = build_vocabulary(history, _books(history), min_reads=1)
    sequences = build_sequences(history, vocabulary)

    targets, starts, lengths = sample_examples(sequences, window=20, rng=np.random.default_rng(0))

    position_in_user = targets - starts
    assert (lengths >= 1).all()
    assert (lengths <= position_in_user).all()
    assert (position_in_user - lengths < 20).all()


def test_recent_bag_keeps_the_last_reads_before_the_cut() -> None:
    items = np.arange(10)

    full, recent = history_bags(items, np.array([0, 5]), np.array([4, 2]), recent=3)

    assert full[0].tolist() == [0, 1, 2, 3, 5, 6]
    assert recent[0].tolist() == [1, 2, 3, 5, 6]
    assert recent[1].tolist() == [0, 3]


def test_a_repeated_book_is_not_its_own_negative() -> None:
    history = _history(n_users=2, n_reads=5)
    vocabulary = build_vocabulary(history, _books(history), min_reads=1)
    net = TwoTowerNet(vocabulary, TwoTowerConfig(embedding_dim=8, dropout=0.0))
    bags = history_bags(np.array([0, 1]), np.array([0, 1]), np.array([1, 1]), recent=20)
    weights = torch.ones(2)

    repeated = in_batch_loss(net, bags, torch.tensor([3, 3]), weights, temperature=1.0)

    assert float(repeated.detach()) == 0.0


def test_fit_recommends_unseen_books_and_falls_back_to_popularity() -> None:
    history = _history()
    users = tuple(sorted(history["id_lector"].unique()))
    split = SplitData(
        holdout=True,
        books=_books(history),
        readers=pd.DataFrame({"id_lector": list(users)}),
        history=history,
        test=history.iloc[0:0].copy(),
        evaluation_users=users,
        submit_users=list(users),
        invalid_interactions=history.iloc[0:0].copy(),
        quality=DataQualityReport(0, 0, 0, 0, 0, 0, 0, 0),
    )
    config = PipelineConfig(
        split=SplitConfig(rolling_folds=2, rolling_horizon=2, min_history=2),
        two_tower=TwoTowerConfig(embedding_dim=8, batch_size=16, max_epochs=3, patience=1, min_item_reads=1),
    )
    model = build_model("two_tower", config)
    model.fit(split)

    ranking = model.recommend(["u0", "nuevo"])

    seen = set(history.loc[history["id_lector"] == "u0", "id_libro"])
    u0 = ranking[ranking["id_lector"] == "u0"]
    assert set(u0["id_libro"]).isdisjoint(seen)
    assert u0["rank"].tolist() == list(range(1, len(u0) + 1))
    assert len(ranking[ranking["id_lector"] == "nuevo"]) == 15
    assert 1 <= model.hyperparameters["epochs"] <= 3
