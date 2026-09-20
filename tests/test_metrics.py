from __future__ import annotations

import pandas as pd

from quelibroleo.metrics import ndcg_at_k, retrieval_metrics


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
