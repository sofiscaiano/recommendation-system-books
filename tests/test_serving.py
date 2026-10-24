from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from quelibroleo.config import PipelineConfig, RankerConfig
from quelibroleo.data import ITEM_COL, USER_COL, observed_catalog
from quelibroleo.serving import (
    ServingModel,
    _isbn,
    _year,
    build_request_history,
    export_serving_artifact,
    recommend_user,
    unknown_book_ids,
)
from quelibroleo.two_stage.columns import ColumnBuilder, attach_relevance
from quelibroleo.two_stage.ranker import LambdaRanker
from quelibroleo.two_stage.retrieval import (
    CandidateFusion,
    fit_retrievers,
    per_source_depth,
    retrievers_from_config,
)


def _catalog(books: list[str]) -> pd.DataFrame:
    authors = ["A", "B", "C"]
    return pd.DataFrame(
        {
            "id_libro": books,
            "titulo": [f"Libro {book}" for book in books],
            "autor": [authors[index % 3] for index in range(len(books))],
            "genero": ["G" if index % 2 == 0 else "H" for index in range(len(books))],
            "editorial": ["E"] * len(books),
            "resumen": [
                "novela de autor alfa" if index % 3 == 0 else "ensayo breve"
                for index in range(len(books))
            ],
            "anio_edicion_num": [2000 + (index % 10) for index in range(len(books))],
            "img_src": [f"https://example.test/{book}.jpg" for book in books],
        }
    )


def _interactions(books: list[str]) -> pd.DataFrame:
    rows = []
    for index, book in enumerate(books):
        if index < 4:
            user = "u1"
            rating = 10
        elif index < 8:
            user = "u2"
            rating = 9
        else:
            user = "u3" if index % 2 == 0 else "u4"
            rating = 7
        day = (index % 27) + 1
        rows.append((user, book, f"2020-01-{day:02d}", rating))
    frame = pd.DataFrame(rows, columns=["id_lector", "id_libro", "fecha", "rating"])
    frame["fecha"] = pd.to_datetime(frame["fecha"])
    return frame


def _fit_ranker(
    model: ServingModel,
    interactions: pd.DataFrame,
) -> LambdaRanker:
    users = tuple(interactions[USER_COL].unique())
    depth = per_source_depth(
        model.rrf_cutoff,
        rrf_k=model.rrf_k,
        n_sources=len(model.retrievers),
    )
    frames = [
        retriever.recommend(users, interactions, depth)
        for retriever in model.retrievers
    ]
    frames = [frame for frame in frames if not frame.empty]
    candidates = pd.concat(frames, ignore_index=True)
    fusion = CandidateFusion(weights=model.weights, rrf_k=model.rrf_k)
    fused = fusion.fuse(candidates, model.rrf_cutoff)
    features = model.feature_builder.transform(
        fused.candidates,
        fused.source_candidates,
    )
    target = (
        fused.candidates.groupby(USER_COL, sort=False)
        .head(1)[[USER_COL, ITEM_COL]]
        .copy()
    )
    target["rating"] = 8
    labeled = attach_relevance(features, target)
    return LambdaRanker(
        RankerConfig(
            n_estimators=15,
            early_stopping_rounds=5,
            min_child_samples=1,
            num_leaves=7,
        )
    ).fit(labeled, labeled)


@pytest.fixture(scope="module")
def fitted() -> tuple[ServingModel, pd.DataFrame, list[str]]:
    books = [f"b{index:02d}" for index in range(30)]
    interactions = _interactions(books)
    catalog = _catalog(books)
    observed = observed_catalog(interactions, catalog)
    config = PipelineConfig()
    retrievers = fit_retrievers(retrievers_from_config(config), interactions, observed)
    builder = ColumnBuilder().fit(interactions, observed)
    model = ServingModel(
        ranker=LambdaRanker(),
        retrievers=retrievers,
        feature_builder=builder,
        weights=dict(config.retrieval.source_weights),
        rrf_cutoff=50,
        rrf_k=config.retrieval.rrf_k,
    )
    model.ranker = _fit_ranker(model, interactions)
    return model, interactions, books


def test_export_serving_requires_trained_ranker(tmp_path: Path) -> None:
    config = PipelineConfig(
        db_path=tmp_path / "missing.db",
        artifacts_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="submit"):
        export_serving_artifact(config)


def test_known_user_gets_twenty_unseen_books(
    fitted: tuple[ServingModel, pd.DataFrame, list[str]],
) -> None:
    model, interactions, _books = fitted
    history = interactions.loc[interactions[USER_COL] == "u1"]
    seen = set(history[ITEM_COL])
    before = model.feature_builder.user_stats_.copy()
    recommended = recommend_user(model, "u1", history)

    assert len(recommended) == 20
    assert len(set(recommended)) == 20
    assert set(recommended).isdisjoint(seen)
    assert model.feature_builder.user_stats_.equals(before)


def test_unknown_user_without_ratings_uses_popular_catalog(
    fitted: tuple[ServingModel, pd.DataFrame, list[str]],
) -> None:
    model, _interactions, books = fitted
    empty = interactions_frame()
    recommended = recommend_user(model, "nuevo", empty)

    assert len(recommended) == 20
    assert set(recommended) <= set(books)


def test_unknown_user_ratings_are_excluded_and_ranker_accepts_missing_sources(
    fitted: tuple[ServingModel, pd.DataFrame, list[str]],
) -> None:
    model, _interactions, _books = fitted
    history = build_request_history(
        interactions_frame(),
        "nuevo",
        [("b00", 10), ("b01", 9)],
    )
    recommended = recommend_user(model, "nuevo", history)

    assert len(recommended) == 20
    assert set(recommended).isdisjoint({"b00", "b01"})


def test_session_rating_replaces_stored_book_and_drops_it_from_the_list(
    fitted: tuple[ServingModel, pd.DataFrame, list[str]],
) -> None:
    model, interactions, _books = fitted
    stored = interactions.loc[interactions[USER_COL] == "u1"].copy()
    original_rating = int(stored.loc[stored[ITEM_COL] == "b00", "rating"].iloc[0])
    history = build_request_history(stored, "u1", [("b00", 1), ("b10", 8)])

    replaced = history.loc[history[ITEM_COL] == "b00"]
    assert len(replaced) == 1
    assert int(replaced["rating"].iloc[0]) == 1
    assert original_rating != 1
    assert (
        history.loc[history[ITEM_COL] == "b10", "fecha"].iloc[0]
        == history["fecha"].max()
    )
    recommended = recommend_user(model, "u1", history)
    assert set(recommended).isdisjoint({"b00", "b10"})


def test_serving_model_roundtrip(
    fitted: tuple[ServingModel, pd.DataFrame, list[str]],
    tmp_path: Path,
) -> None:
    model, interactions, _books = fitted
    path = tmp_path / "serving.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    history = interactions.loc[interactions[USER_COL] == "u1"]
    assert recommend_user(restored, "u1", history) == recommend_user(
        model, "u1", history
    )


def test_book_card_formats_year_and_isbn() -> None:
    assert _year(2006) == "2006"
    assert _year(2006.0) == "2006"
    assert _year(None) == ""
    assert _isbn(9788433967305.0) == "9788433967305"
    assert _isbn("978-84-339-6730-5") == "978-84-339-6730-5"


def test_unknown_book_ids_keep_first_seen_order() -> None:
    assert unknown_book_ids(["b1", "missing", "b1", "other"], {"b1"}) == [
        "missing",
        "other",
    ]


def interactions_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=["id_lector", "id_libro", "fecha", "rating"])
    frame["fecha"] = pd.to_datetime(frame["fecha"])
    return frame
