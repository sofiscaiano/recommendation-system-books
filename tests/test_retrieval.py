from __future__ import annotations

import pandas as pd

from quelibroleo.baselines import PopularityRecommender
from quelibroleo.retrieval import CandidateFusion, ItemCosineRetriever


def tiny_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.DataFrame(
        [
            ("u1", "b1", "2020-01-01", 10),
            ("u1", "b2", "2020-01-02", 8),
            ("u2", "b1", "2020-01-01", 9),
            ("u2", "b3", "2020-01-02", 7),
            ("u3", "b2", "2020-01-01", 8),
            ("u3", "b3", "2020-01-02", 9),
            ("u3", "b4", "2020-01-03", 6),
        ],
        columns=["id_lector", "id_libro", "fecha", "rating"],
    )
    interactions["fecha"] = pd.to_datetime(interactions["fecha"])
    catalog = pd.DataFrame(
        {
            "id_libro": ["b1", "b2", "b3", "b4"],
            "titulo": ["Uno", "Dos", "Tres", "Cuatro"],
            "autor": ["A", "A", "B", "C"],
            "genero": ["G", "G", "H", "H"],
            "editorial": ["E"] * 4,
            "resumen": ["texto libro"] * 4,
            "anio_edicion_num": [2020] * 4,
        }
    )
    return interactions, catalog


def test_popularity_and_item_cosine_exclude_seen_items() -> None:
    interactions, catalog = tiny_data()
    popularity = PopularityRecommender().fit(
        interactions, set(catalog["id_libro"])
    )
    popular = popularity.recommend(["u1"], interactions, 3)
    assert set(popular["id_libro"]).isdisjoint({"b1", "b2"})

    cosine = ItemCosineRetriever("occurrence").fit(interactions, catalog)
    candidates = cosine.recommend(["u1"], interactions, 3)
    assert not candidates.empty
    assert set(candidates["id_libro"]).isdisjoint({"b1", "b2"})
    assert candidates["rank"].is_monotonic_increasing


def test_candidate_fusion_respects_budget_and_deduplicates() -> None:
    rows = []
    for source in ("a", "b"):
        for rank, item in enumerate(("i1", "i2", "i3", "i4"), start=1):
            rows.append(
                {
                    "id_lector": "u",
                    "id_libro": item,
                    "source": source,
                    "score": 5 - rank,
                    "rank": rank,
                }
            )
    candidates = pd.DataFrame(rows)
    fusion = CandidateFusion(
        weights={"a": 1.0, "b": 2.0},
        quotas={"a": 2, "b": 2},
    ).fuse(candidates, budget=3)

    assert len(fusion.candidates) == 3
    assert fusion.candidates["id_libro"].is_unique
    assert fusion.candidates["retrieval_rank"].tolist() == [1, 2, 3]
