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


def _two_source_candidates() -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def test_candidate_fusion_respects_cutoff_and_deduplicates() -> None:
    fusion = CandidateFusion(weights={"a": 1.0, "b": 2.0})
    fused = fusion.fuse(_two_source_candidates(), cutoff=3)

    assert fusion.weights["a"] + fusion.weights["b"] == 1
    assert len(fused.candidates) == 3
    assert fused.candidates["id_libro"].is_unique
    assert fused.candidates["retrieval_rank"].tolist() == [1, 2, 3]
    assert fused.candidates["id_libro"].tolist() == ["i1", "i2", "i3"]


def test_rrf_cutoff_is_a_prefix_of_the_same_ranking() -> None:
    fusion = CandidateFusion(weights={"a": 1.0, "b": 2.0})
    candidates = _two_source_candidates()
    short = fusion.fuse(candidates, cutoff=2).candidates
    long = fusion.fuse(candidates, cutoff=4).candidates

    assert short["id_libro"].tolist() == long["id_libro"].tolist()[:2]


def test_weight_scale_does_not_change_the_fused_ranking() -> None:
    candidates = _two_source_candidates()
    base = CandidateFusion(weights={"a": 1.0, "b": 2.0}).fuse(candidates, 3)
    scaled = CandidateFusion(weights={"a": 10.0, "b": 20.0}).fuse(candidates, 3)

    assert scaled.candidates["id_libro"].tolist() == base.candidates["id_libro"].tolist()


def test_rrf_cutoff_follows_fused_score_not_source_slots() -> None:
    rows = [
        {"id_lector": "u", "id_libro": "only_a", "source": "a", "score": 9, "rank": 1},
        {"id_lector": "u", "id_libro": "both", "source": "a", "score": 1, "rank": 2},
        {"id_lector": "u", "id_libro": "both", "source": "b", "score": 1, "rank": 2},
    ]
    fused = CandidateFusion(weights={"a": 1.0, "b": 1.0}).fuse(
        pd.DataFrame(rows), cutoff=1
    )

    assert fused.candidates["id_libro"].tolist() == ["both"]
