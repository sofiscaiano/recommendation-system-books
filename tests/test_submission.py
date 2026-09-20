from __future__ import annotations

import pandas as pd

from quelibroleo.baselines import PopularityRecommender
from quelibroleo.pipeline import _fill_submission


def test_submission_has_twenty_unique_unseen_books_per_user() -> None:
    interactions = pd.DataFrame(
        [
            {
                "id_lector": f"source-{index}",
                "id_libro": f"b{index:02d}",
                "rating": 8,
                "fecha": pd.Timestamp("2020-01-01"),
            }
            for index in range(25)
        ]
        + [
            {
                "id_lector": "known",
                "id_libro": "b00",
                "rating": 10,
                "fecha": pd.Timestamp("2020-01-02"),
            }
        ]
    )
    popularity = PopularityRecommender().fit(interactions)
    ranked = pd.DataFrame(
        {
            "id_lector": ["known"] * 3,
            "id_libro": ["b01", "b02", "b03"],
            "rank": [1, 2, 3],
        }
    )

    submission = _fill_submission(
        ranked,
        ["known", "cold"],
        interactions,
        popularity,
    )

    assert submission.groupby("id_lector").size().eq(20).all()
    assert not submission.duplicated(["id_lector", "id_libro"]).any()
    assert "b00" not in set(
        submission.loc[submission["id_lector"] == "known", "id_libro"]
    )
