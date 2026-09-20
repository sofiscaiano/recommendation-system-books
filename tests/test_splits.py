from __future__ import annotations

import pandas as pd

from quelibroleo.splits import (
    assert_no_within_user_leakage,
    leave_last_k,
    rolling_origin_folds,
)


def interactions(users: int = 3, rows: int = 50) -> pd.DataFrame:
    records = []
    for user in range(users):
        for index in range(rows):
            records.append(
                {
                    "id_lector": f"u{user}",
                    "id_libro": f"b{index:03d}",
                    "fecha": pd.Timestamp("2020-01-01")
                    + pd.Timedelta(days=index // 2),
                    "rating": index % 10 + 1,
                }
            )
    records.append(
        {
            "id_lector": "short",
            "id_libro": "short-book",
            "fecha": pd.Timestamp("2020-01-01"),
            "rating": 8,
        }
    )
    return pd.DataFrame(records)


def test_leave_last_k_is_exact_and_deterministic() -> None:
    data = interactions()
    first = leave_last_k(data, k=20)
    second = leave_last_k(data.sample(frac=1.0, random_state=7), k=20)

    assert len(first.evaluation_users) == 3
    assert first.test.groupby("id_lector").size().eq(20).all()
    assert "short" in set(first.train["id_lector"])
    assert first.test[["id_lector", "id_libro"]].equals(
        second.test[["id_lector", "id_libro"]]
    )


def test_rolling_folds_only_use_each_query_prefix() -> None:
    folds = rolling_origin_folds(
        interactions(rows=70),
        n_folds=3,
        horizon=10,
        min_history=5,
    )
    assert len(folds) == 3
    for fold in folds:
        assert_no_within_user_leakage(fold)
        assert fold.target.groupby("id_lector").size().eq(10).all()
        for user in fold.query_users:
            train_items = set(
                fold.train.loc[fold.train["id_lector"] == user, "id_libro"]
            )
            target_items = set(
                fold.target.loc[fold.target["id_lector"] == user, "id_libro"]
            )
            assert train_items.isdisjoint(target_items)
