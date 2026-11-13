from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from quelibroleo.baselines import PopularityRecommender
from quelibroleo.config import PipelineConfig, RetrievalConfig
from quelibroleo.ease import EaseParameters
from quelibroleo.two_stage.retrieval import (
    CandidateFusion,
    EaseRetriever,
    ImplicitMFRetriever,
    ItemCosineRetriever,
    build_default_retrievers,
    per_source_depth,
    retrievers_from_config,
)


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


def test_bpr_uses_new_reads_without_refitting() -> None:
    interactions, catalog = tiny_data()
    retriever = ImplicitMFRetriever("bpr", factors=8, iterations=5, random_state=0)
    retriever.fit(interactions, catalog)
    user = interactions.loc[interactions["id_lector"] == "u1"]
    baseline = retriever.recommend(["u1"], user, n=2)
    same = retriever.recommend_with_history("u1", user, n=2)

    assert same["id_libro"].tolist() == baseline["id_libro"].tolist()
    assert same["score"].tolist() == baseline["score"].tolist()

    extra = user.iloc[:1].copy()
    extra["id_libro"] = "b4"
    extra["fecha"] = user["fecha"].max() + pd.Timedelta(days=1)
    extra["rating"] = 10
    extended = pd.concat([user, extra], ignore_index=True)
    moved = retriever.recommend_with_history("u1", extended, n=2)

    assert set(moved["id_libro"]).isdisjoint({"b1", "b2", "b4"})
    assert moved["id_libro"].tolist() == ["b3"]

    newbie = user.copy()
    newbie["id_lector"] = "nuevo"
    cold = retriever.recommend_with_history("nuevo", newbie, n=2)
    assert set(cold["id_libro"]).isdisjoint(set(newbie["id_libro"]))
    assert len(cold) == 2


def test_popularity_and_item_cosine_exclude_seen_items() -> None:
    interactions, catalog = tiny_data()
    popularity = PopularityRecommender().fit(
        interactions, set(catalog["id_libro"])
    )
    popular = popularity.recommend(["u1"], interactions, 3)
    assert set(popular["id_libro"]).isdisjoint({"b1", "b2"})

    cosine = ItemCosineRetriever().fit(interactions, catalog)
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


def test_default_retrievers_keep_sources_that_change_the_candidate_set() -> None:
    names = [retriever.source_name for retriever in build_default_retrievers()]

    assert names == ["popularidad", "item_cosine", "als", "bpr", "tfidf"]


def test_source_flags_choose_which_retrievers_are_built(tmp_path) -> None:
    retrieval = replace(
        RetrievalConfig(),
        als_source=False,
        bpr_source=False,
        two_tower_source=True,
        ease_source=False,
    )
    config = replace(PipelineConfig(artifacts_dir=tmp_path), retrieval=retrieval)
    names = [retriever.source_name for retriever in retrievers_from_config(config)]

    assert names == ["popularidad", "item_cosine", "tfidf", "two_tower"]


def test_ease_retriever_ranks_neighbors_and_does_not_pad_with_popularity() -> None:
    interactions, catalog = tiny_data()
    source = EaseRetriever(
        EaseParameters(reg_lambda=1.0, min_interactions=1),
        half_life_days=365.0,
    ).fit(interactions, catalog)

    candidates = source.recommend(["u1", "nuevo"], interactions, n=10)

    assert list(candidates.columns) == [
        "id_lector",
        "id_libro",
        "source",
        "score",
        "rank",
    ]
    assert set(candidates["id_lector"]) == {"u1"}
    assert candidates["id_libro"].tolist() == ["b3", "b4"]
    assert candidates["rank"].tolist() == [1, 2]
    assert set(candidates["id_libro"]).isdisjoint({"b1", "b2"})
    assert (candidates["source"] == "ease").all()


def test_ease_source_reuses_saved_parameters(tmp_path) -> None:
    (tmp_path / "ease_hyperparameters.json").write_text(
        '{"reg_lambda": 12.0, "min_interactions": 30, "weight_mode": "rating"}\n',
        encoding="utf-8",
    )
    config = PipelineConfig(artifacts_dir=tmp_path)
    ease = next(
        retriever
        for retriever in retrievers_from_config(config)
        if retriever.source_name == "ease"
    )

    assert isinstance(ease, EaseRetriever)
    assert ease.parameters.reg_lambda == 12.0
    assert ease.parameters.min_interactions == 30


def test_each_source_reuses_its_own_saved_parameters(tmp_path) -> None:
    (tmp_path / "als_hyperparameters.json").write_text(
        json.dumps(
            {
                "factors": 32,
                "iterations": 7,
                "regularization": 0.2,
                "half_life_days": 90,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "item_cosine_hyperparameters.json").write_text(
        json.dumps({"half_life_days": 40}),
        encoding="utf-8",
    )
    (tmp_path / "two_tower_hyperparameters.json").write_text(
        json.dumps(
            {
                "embedding_dim": 48,
                "learning_rate": 0.0004,
                "temperature": 0.08,
                "min_item_reads": 4,
            }
        ),
        encoding="utf-8",
    )
    config = PipelineConfig(artifacts_dir=tmp_path)
    by_name = {retriever.source_name: retriever for retriever in retrievers_from_config(config)}

    assert by_name["als"].factors == 32
    assert by_name["als"].iterations == 7
    assert by_name["als"].regularization == 0.2
    assert by_name["als"].half_life_days == 90
    assert by_name["bpr"].factors == 64
    assert by_name["bpr"].learning_rate == 0.01
    assert by_name["bpr"].half_life_days == 365
    assert by_name["item_cosine"].half_life_days == 40
    assert by_name["tfidf"].max_features == 60_000
    assert by_name["two_tower"].config.two_tower.embedding_dim == 48
    assert by_name["two_tower"].config.two_tower.min_item_reads == 4
    assert by_name["two_tower"].config.two_tower.dropout == config.two_tower.dropout


def test_retrieval_config_requires_one_enabled_source() -> None:
    with pytest.raises(ValueError, match="al menos una fuente"):
        replace(
            RetrievalConfig(),
            popularidad_source=False,
            item_cosine_source=False,
            als_source=False,
            bpr_source=False,
            tfidf_source=False,
            two_tower_source=False,
            ease_source=False,
        )


def test_per_source_depth_keeps_the_tail_out_of_the_fused_head() -> None:
    cutoff = 200
    k = 60.0
    n_sources = 5
    depth = per_source_depth(cutoff, rrf_k=k, n_sources=n_sources)

    assert depth == 1240
    assert per_source_depth(1000, rrf_k=k, n_sources=n_sources) == 5240
    marginal = (1.0 / n_sources) / (k + cutoff)
    tail = 1.0 / (k + depth + 1)
    assert marginal > tail


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
