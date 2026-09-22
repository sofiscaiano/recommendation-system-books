from __future__ import annotations

import pandas as pd

from quelibroleo.metrics import ndcg_at_k, retrieval_metrics, segment_metrics


def test_ndcg_penalizes_relevant_items_not_retrieved() -> None:
    truth = pd.DataFrame(
        {
            "id_lector": ["u", "u"],
            "id_libro": ["relevant-10", "relevant-8"],
            "rating": [10, 8],
        }
    )
    perfect = pd.DataFrame(
        {
            "id_lector": ["u", "u"],
            "id_libro": ["relevant-10", "relevant-8"],
            "rank": [1, 2],
        }
    )
    missing = pd.DataFrame(
        {
            "id_lector": ["u", "u"],
            "id_libro": ["relevant-10", "irrelevant"],
            "rank": [1, 2],
        }
    )

    assert ndcg_at_k(perfect, truth, k=2) == 1.0
    assert 0.0 < ndcg_at_k(missing, truth, k=2) < 1.0


def test_retrieval_metrics_report_binary_weighted_and_oracle() -> None:
    truth = pd.DataFrame(
        {
            "id_lector": ["u", "u"],
            "id_libro": ["high", "low"],
            "rating": [10, 2],
        }
    )
    candidates = pd.DataFrame(
        {
            "id_lector": ["u"],
            "id_libro": ["high"],
            "rrf_score": [1.0],
        }
    )
    metrics = retrieval_metrics(candidates, truth)

    assert metrics.recall == 0.5
    assert metrics.weighted_recall == 10 / 12
    assert 0 < metrics.oracle_ndcg < 1


def test_segment_metrics_include_retrieval_coverage_and_tail() -> None:
    train = pd.DataFrame(
        {
            "id_lector": (
                ["u0"]
                + ["u1"] * 2
                + ["u2"] * 6
                + ["u3"] * 12
            ),
            "id_libro": [f"seen-{i}" for i in range(21)],
            "fecha": pd.to_datetime(["2020-01-01"] * 21),
            "rating": [8] * 21,
        }
    )
    truth = pd.DataFrame(
        {
            "id_lector": ["u0", "u0", "u1", "u2", "u3", "u3"],
            "id_libro": [
                "hit",
                "miss-light",
                "hit",
                "hit",
                "hit",
                "miss-heavy",
            ],
            "rating": [10, 8, 9, 9, 10, 8],
        }
    )
    recommendations = pd.DataFrame(
        {
            "id_lector": ["u0", "u0", "u1", "u2", "u3", "u3"],
            "id_libro": ["hit", "other", "hit", "hit", "hit", "other"],
            "rank": [1, 2, 1, 1, 1, 2],
        }
    )
    candidates = pd.DataFrame(
        {
            "id_lector": ["u0", "u0", "u1", "u2", "u3"],
            "id_libro": ["hit", "miss-light", "hit", "hit", "hit"],
        }
    )
    eligible = set(train["id_libro"]) | {"hit", "miss-light", "miss-heavy", "other"}

    report = segment_metrics(
        recommendations,
        candidates,
        truth,
        train,
        eligible,
        k=2,
    )

    assert report["activity_q1"]["retrieval_recall"] == 1.0
    assert report["activity_q4"]["retrieval_recall"] == 0.5
    assert "long_tail_exposure_at_20" in report["activity_q1"]
    assert "catalog_coverage_at_20" in report["activity_q1"]
    assert report["single_date_history"]["users"] >= 2
