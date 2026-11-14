from __future__ import annotations

from math import log1p

import pandas as pd
import pytest

from quelibroleo.config import RankerConfig
from quelibroleo.two_stage.columns import ColumnBuilder, attach_relevance
from quelibroleo.two_stage.ranker import LambdaRanker


def test_label_gain_is_configurable_and_reaches_lightgbm() -> None:
    linear = LambdaRanker(RankerConfig(label_gain="linear"))._parameters()
    exponential = LambdaRanker(
        RankerConfig(label_gain="exponential"),
        overrides={"learning_rate": 0.05, "early_stopping_rounds": 30},
    )._parameters()

    assert linear["label_gain"] == list(range(11))
    assert exponential["label_gain"] == [2**rating - 1 for rating in range(11)]
    assert exponential["learning_rate"] == 0.05
    assert exponential["objective"] == "lambdarank"
    with pytest.raises(ValueError):
        RankerConfig(label_gain="identity")


def test_publication_age_is_years_since_the_users_last_read() -> None:
    train = pd.DataFrame(
        [("u1", "b1", "2020-06-01", 8)],
        columns=["id_lector", "id_libro", "fecha", "rating"],
    )
    train["fecha"] = pd.to_datetime(train["fecha"])
    catalog = pd.DataFrame(
        {
            "id_libro": ["b1", "b_old", "b_new"],
            "autor": ["A", "B", "C"],
            "genero": ["G", "H", "I"],
            "anio_edicion_num": [2010, 2015, 2024],
        }
    )
    fused = pd.DataFrame(
        [
            ("u1", "b_old", 0.4, 1),
            ("u1", "b_new", 0.2, 2),
        ],
        columns=["id_lector", "id_libro", "rrf_score", "retrieval_rank"],
    )
    source = fused.rename(columns={"rrf_score": "score", "retrieval_rank": "rank"})
    source["source"] = "mock"

    features = ColumnBuilder().fit(train, catalog).transform(fused, source)

    ages = features.frame.set_index("id_libro")["publication_age"]
    assert ages["b_old"] == 5
    assert ages["b_new"] == -4


def _cell(frame: pd.DataFrame, user: str, book: str, column: str) -> object:
    rows = frame[(frame["id_lector"] == user) & (frame["id_libro"] == book)]
    assert len(rows) == 1
    return rows.iloc[0][column]


def _expect(
    frame: pd.DataFrame,
    user: str,
    book: str,
    column: str,
    expected: float | None,
) -> None:
    actual = _cell(frame, user, book, column)
    if expected is None:
        assert pd.isna(actual), (user, book, column, actual)
        return
    assert actual == pytest.approx(expected), (user, book, column, actual, expected)


def test_new_ranker_columns_match_hand_calculation() -> None:
    train = pd.DataFrame(
        [
            ("u1", "bA1", "2010-01-01", 8),
            ("u1", "bB1", "2010-01-05", 6),
            ("u1", "bA2", "2010-01-15", 9),
            ("u2", "bA1", "2019-03-01", 5),
            ("u2", "bB1", "2019-03-01", 7),
            ("u2", "bC1", "2019-04-01", 10),
            ("u3", "bD1", "2021-05-01", 4),
        ],
        columns=["id_lector", "id_libro", "fecha", "rating"],
    )
    train["fecha"] = pd.to_datetime(train["fecha"])
    catalog = pd.DataFrame(
        {
            "id_libro": ["bA1", "bA2", "bB1", "bC1", "bD1", "b_new", "b_blank"],
            "autor": ["Ana", "Ana", "Bea", "Ciro", "Dina", "Ana", ""],
            "genero": ["Novela", "Novela", "Poesia", "Poesia", "Ensayo", "Novela", ""],
            "anio_edicion_num": [1995, 2005, 2012, 2020, 1980, 2024, 2001],
        }
    )
    readers = pd.DataFrame(
        {
            "id_lector": ["u1", "u2", "u3"],
            "genero": ["Mujer", "Hombre", "-"],
            "nacimiento": ["1980", "2000", ""],
        }
    )
    fused = pd.DataFrame(
        [
            ("u1", "bC1", 0.5, 1),
            ("u1", "b_new", 0.4, 2),
            ("u1", "b_blank", 0.3, 3),
            ("u1", "bA1", 0.2, 4),
            ("u2", "bA2", 0.5, 1),
            ("u3", "bA1", 0.5, 1),
        ],
        columns=["id_lector", "id_libro", "rrf_score", "retrieval_rank"],
    )
    source = fused.rename(columns={"rrf_score": "score", "retrieval_rank": "rank"})
    source["source"] = "mock"

    table = (
        ColumnBuilder(recent_n=2).fit(train, catalog, readers).transform(fused, source)
    )
    frame = table.frame

    added = {
        "user_mean_days_between_reads",
        "user_first_year",
        "user_last_year",
        "user_gender",
        "user_age_first",
        "user_age_last",
        "edition_vs_user_mean_year",
        "author_mean_rating",
        "item_author_rating_ratio",
        "user_author_share",
        "user_genre_share",
        "item_author_share",
        "item_genre_share",
        "user_author_diversity",
        "user_genre_diversity",
        "user_decade_diversity",
        "author_transition_window",
        "genre_transition_window",
        "item_transition_window",
        "days_since_author",
        "author_high_rating_share",
        "user_median_item_popularity",
        "item_popularity_vs_user",
        "author_repeat_rate",
    }
    assert added <= set(table.feature_columns)
    assert {
        "user_mean_read_year",
        "author_read_count",
        "genre_read_count",
    }.isdisjoint(table.feature_columns)

    ana_mean = (8 + 9 + 5) / 3
    _expect(frame, "u1", "bC1", "user_mean_days_between_reads", 7)
    _expect(frame, "u1", "bC1", "user_first_year", 2010)
    _expect(frame, "u1", "bC1", "user_last_year", 2010)
    _expect(frame, "u1", "bC1", "user_gender", 1)
    _expect(frame, "u1", "bC1", "user_age_first", 30)
    _expect(frame, "u1", "bC1", "user_age_last", 30)
    _expect(frame, "u1", "bC1", "user_author_diversity", 2 / 3)
    _expect(frame, "u1", "bC1", "user_genre_diversity", 2 / 3)
    _expect(frame, "u1", "bC1", "user_decade_diversity", 1)
    _expect(frame, "u1", "bC1", "edition_vs_user_mean_year", 10)
    _expect(frame, "u1", "bC1", "publication_age", -10)
    _expect(frame, "u1", "bC1", "user_author_share", 0)
    _expect(frame, "u1", "bC1", "user_genre_share", 1 / 3)
    _expect(frame, "u1", "bC1", "recent_author_share", 0)
    _expect(frame, "u1", "bC1", "recent_genre_share", 0.5)
    _expect(frame, "u1", "bC1", "author_mean_rating", 10)
    _expect(frame, "u1", "bC1", "item_author_rating_ratio", 1)
    _expect(frame, "u1", "bC1", "item_author_share", 1)
    _expect(frame, "u1", "bC1", "item_genre_share", 1 / 3)
    _expect(frame, "u1", "bC1", "author_transition_prob", 0.5)
    _expect(frame, "u1", "bC1", "author_transition_window", 0.5)
    _expect(frame, "u1", "bC1", "genre_transition_prob", 1)
    _expect(frame, "u1", "bC1", "genre_transition_window", 5 / 6)
    _expect(frame, "u1", "bC1", "item_transition_prob", None)
    _expect(frame, "u1", "bC1", "item_transition_window", 7 / 24)
    _expect(frame, "u1", "bC1", "days_since_author", None)
    _expect(frame, "u1", "bC1", "author_high_rating_share", None)
    _expect(frame, "u1", "bC1", "user_median_item_popularity", 2)
    _expect(frame, "u1", "bC1", "item_popularity_vs_user", log1p(1) - log1p(2))
    _expect(frame, "u1", "bC1", "author_repeat_rate", 0.5)

    _expect(frame, "u1", "b_new", "edition_vs_user_mean_year", 14)
    _expect(frame, "u1", "b_new", "user_author_share", 2 / 3)
    _expect(frame, "u1", "b_new", "user_genre_share", 2 / 3)
    _expect(frame, "u1", "b_new", "recent_author_share", 0.5)
    _expect(frame, "u1", "b_new", "author_mean_rating", ana_mean)
    _expect(frame, "u1", "b_new", "item_author_rating_ratio", None)
    _expect(frame, "u1", "b_new", "item_author_share", None)
    _expect(frame, "u1", "b_new", "author_transition_prob", 0.5)
    _expect(frame, "u1", "b_new", "author_transition_window", 0.5)
    _expect(frame, "u1", "b_new", "genre_transition_window", 0.5)
    _expect(frame, "u1", "b_new", "item_transition_window", 0)
    _expect(frame, "u1", "b_new", "days_since_author", 0)
    _expect(frame, "u1", "b_new", "author_high_rating_share", 1)
    _expect(frame, "u1", "b_new", "user_median_item_popularity", 2)
    _expect(frame, "u1", "b_new", "item_popularity_vs_user", None)
    _expect(frame, "u1", "b_new", "author_repeat_rate", 0.5)

    _expect(frame, "u1", "b_blank", "user_author_share", 0)
    _expect(frame, "u1", "b_blank", "item_author_share", None)
    _expect(frame, "u1", "b_blank", "item_genre_share", None)
    _expect(frame, "u1", "b_blank", "author_mean_rating", None)
    _expect(frame, "u1", "b_blank", "days_since_author", None)
    _expect(frame, "u1", "b_blank", "author_high_rating_share", None)
    _expect(frame, "u1", "b_blank", "item_popularity_vs_user", None)

    _expect(frame, "u1", "bA1", "author_mean_rating", ana_mean)
    _expect(frame, "u1", "bA1", "item_author_rating_ratio", 6.5 / ana_mean)
    _expect(frame, "u1", "bA1", "item_author_share", 2 / 3)
    _expect(frame, "u1", "bA1", "item_genre_share", 2 / 3)
    _expect(frame, "u1", "bA1", "user_author_share", 2 / 3)
    _expect(frame, "u1", "bA1", "days_since_author", 0)
    _expect(frame, "u1", "bA1", "author_high_rating_share", 1)
    _expect(frame, "u1", "bA1", "item_popularity_vs_user", 0)

    _expect(frame, "u2", "bA2", "user_mean_days_between_reads", 15.5)
    _expect(frame, "u2", "bA2", "user_first_year", 2019)
    _expect(frame, "u2", "bA2", "user_last_year", 2019)
    _expect(frame, "u2", "bA2", "user_gender", 0)
    _expect(frame, "u2", "bA2", "user_age_first", 19)
    _expect(frame, "u2", "bA2", "user_age_last", 19)
    _expect(frame, "u2", "bA2", "user_author_diversity", 1)
    _expect(frame, "u2", "bA2", "user_genre_diversity", 2 / 3)
    _expect(frame, "u2", "bA2", "user_decade_diversity", 1)
    _expect(frame, "u2", "bA2", "edition_vs_user_mean_year", -14)
    _expect(frame, "u2", "bA2", "user_author_share", 1 / 3)
    _expect(frame, "u2", "bA2", "user_genre_share", 1 / 3)
    _expect(frame, "u2", "bA2", "item_author_rating_ratio", 9 / ana_mean)
    _expect(frame, "u2", "bA2", "item_author_share", 1 / 3)
    _expect(frame, "u2", "bA2", "item_genre_share", 1 / 3)
    _expect(frame, "u2", "bA2", "author_transition_prob", None)
    _expect(frame, "u2", "bA2", "author_transition_window", 7 / 24)
    _expect(frame, "u2", "bA2", "genre_transition_prob", 0.5)
    _expect(frame, "u2", "bA2", "genre_transition_window", 0.5)
    _expect(frame, "u2", "bA2", "item_transition_window", 7 / 24)
    _expect(frame, "u2", "bA2", "days_since_author", 31)
    _expect(frame, "u2", "bA2", "author_high_rating_share", 0)
    _expect(frame, "u2", "bA2", "user_median_item_popularity", 2)
    _expect(frame, "u2", "bA2", "item_popularity_vs_user", log1p(1) - log1p(2))
    _expect(frame, "u2", "bA2", "author_repeat_rate", 0)

    _expect(frame, "u3", "bA1", "user_mean_days_between_reads", None)
    _expect(frame, "u3", "bA1", "user_first_year", 2021)
    _expect(frame, "u3", "bA1", "user_last_year", 2021)
    _expect(frame, "u3", "bA1", "user_gender", None)
    _expect(frame, "u3", "bA1", "user_age_first", None)
    _expect(frame, "u3", "bA1", "user_age_last", None)
    _expect(frame, "u3", "bA1", "user_author_diversity", 1)
    _expect(frame, "u3", "bA1", "user_genre_diversity", 1)
    _expect(frame, "u3", "bA1", "user_decade_diversity", 1)
    _expect(frame, "u3", "bA1", "edition_vs_user_mean_year", 1995 - 2021)
    _expect(frame, "u3", "bA1", "user_author_share", 0)
    _expect(frame, "u3", "bA1", "author_transition_window", 0)
    _expect(frame, "u3", "bA1", "genre_transition_window", 0)
    _expect(frame, "u3", "bA1", "recent_author_share", 0)
    _expect(frame, "u3", "bA1", "days_since_author", None)
    _expect(frame, "u3", "bA1", "author_high_rating_share", None)
    _expect(frame, "u3", "bA1", "user_median_item_popularity", 1)
    _expect(frame, "u3", "bA1", "item_popularity_vs_user", log1p(2) - log1p(1))
    _expect(frame, "u3", "bA1", "author_repeat_rate", None)


def test_transition_window_renormalizes_when_history_has_two_reads() -> None:
    train = pd.DataFrame(
        [
            ("u_long", "bA", "2010-01-01", 5),
            ("u_long", "bB", "2010-02-01", 6),
            ("u_long", "bA", "2010-03-01", 7),
            ("u_short", "bA", "2011-01-01", 8),
            ("u_short", "bB", "2011-01-21", 9),
        ],
        columns=["id_lector", "id_libro", "fecha", "rating"],
    )
    train["fecha"] = pd.to_datetime(train["fecha"])
    catalog = pd.DataFrame(
        {
            "id_libro": ["bA", "bB"],
            "autor": ["A", "B"],
            "genero": ["G", "G"],
            "anio_edicion_num": [2000, 2001],
        }
    )
    readers = pd.DataFrame(
        {
            "id_lector": ["u_short"],
            "genero": ["Mujer"],
            "nacimiento": ["1990"],
        }
    )
    fused = pd.DataFrame(
        [
            ("u_short", "bA", 0.4, 1),
            ("u_short", "bB", 0.2, 2),
        ],
        columns=["id_lector", "id_libro", "rrf_score", "retrieval_rank"],
    )
    source = fused.rename(columns={"rrf_score": "score", "retrieval_rank": "rank"})
    source["source"] = "mock"
    frame = ColumnBuilder().fit(train, catalog, readers).transform(fused, source).frame

    _expect(frame, "u_short", "bA", "author_transition_window", 7 / 9)
    _expect(frame, "u_short", "bB", "author_transition_window", 4 / 9)
    _expect(frame, "u_short", "bA", "author_transition_prob", 1)
    _expect(frame, "u_short", "bA", "user_mean_days_between_reads", 20)
    _expect(frame, "u_short", "bA", "user_age_first", 21)
    _expect(frame, "u_short", "bA", "user_age_last", 21)
    _expect(frame, "u_short", "bA", "user_author_diversity", 1)

    _expect(frame, "u_short", "bA", "user_decade_diversity", 0.5)


def test_item_transition_stops_after_five_reads() -> None:
    hit = ["b0", "f1", "f2", "f3", "f4", "bZ"]
    miss = ["b0", "f1", "f2", "f3", "f4", "f5", "bZ"]
    rows = [
        ("u_hit", book, f"2010-01-{index + 1:02d}", 8) for index, book in enumerate(hit)
    ]
    rows += [
        ("u_miss", book, f"2011-01-{index + 1:02d}", 8)
        for index, book in enumerate(miss)
    ]
    rows.append(("u_now", "b0", "2012-01-01", 8))
    train = pd.DataFrame(rows, columns=["id_lector", "id_libro", "fecha", "rating"])
    train["fecha"] = pd.to_datetime(train["fecha"])
    books = sorted({book for _, book, _, _ in rows} | {"bZ"})
    catalog = pd.DataFrame(
        {
            "id_libro": books,
            "autor": ["A"] * len(books),
            "genero": ["G"] * len(books),
            "anio_edicion_num": [2000] * len(books),
        }
    )
    fused = pd.DataFrame(
        [("u_now", "bZ", 0.5, 1)],
        columns=["id_lector", "id_libro", "rrf_score", "retrieval_rank"],
    )
    source = fused.rename(columns={"rrf_score": "score", "retrieval_rank": "rank"})
    source["source"] = "mock"
    frame = ColumnBuilder().fit(train, catalog).transform(fused, source).frame

    _expect(frame, "u_now", "bZ", "item_transition_prob", 0.5)
    _expect(frame, "u_now", "bZ", "item_transition_window", 0.5)


def test_features_labels_and_lambdarank_roundtrip() -> None:
    train = pd.DataFrame(
        [
            ("u1", "b1", "2020-01-01", 8),
            ("u1", "b2", "2020-01-02", 9),
            ("u2", "b3", "2020-01-01", 7),
            ("u2", "b4", "2020-01-03", 10),
            ("u3", "b1", "2020-01-01", 6),
            ("u3", "b4", "2020-01-04", 8),
        ],
        columns=["id_lector", "id_libro", "fecha", "rating"],
    )
    train["fecha"] = pd.to_datetime(train["fecha"])
    catalog = pd.DataFrame(
        {
            "id_libro": ["b1", "b2", "b3", "b4"],
            "autor": ["A", "A", "B", "C"],
            "genero": ["G", "G", "H", "H"],
            "editorial": ["E"] * 4,
            "anio_edicion_num": [2010, 2011, 2012, 2013],
        }
    )
    fused = pd.DataFrame(
        [
            ("u1", "b3", 0.3, 1),
            ("u1", "b4", 0.2, 2),
            ("u2", "b1", 0.4, 1),
            ("u2", "b2", 0.1, 2),
            ("u3", "b2", 0.3, 1),
            ("u3", "b3", 0.2, 2),
        ],
        columns=["id_lector", "id_libro", "rrf_score", "retrieval_rank"],
    )
    source = fused.rename(columns={"rrf_score": "score", "retrieval_rank": "rank"})
    source["source"] = "mock"
    target = pd.DataFrame(
        [
            ("u1", "b4", 10),
            ("u2", "b1", 8),
            ("u3", "b3", 9),
        ],
        columns=["id_lector", "id_libro", "rating"],
    )

    features = ColumnBuilder(recent_n=2).fit(train, catalog).transform(fused, source)
    labeled = attach_relevance(features, target, query_suffix="__test")

    assert labeled.frame["relevance"].tolist() == [0, 10, 8, 0, 0, 9]
    assert labeled.frame["item_count"].notna().all()
    assert "recent_genre_share" in labeled.feature_columns

    ranker = LambdaRanker(
        RankerConfig(
            n_estimators=20,
            early_stopping_rounds=5,
            min_child_samples=1,
            num_leaves=7,
        )
    ).fit(labeled, labeled)
    predictions = ranker.predict(features)

    assert len(predictions) == len(fused)
    assert predictions.groupby("id_lector")["rank"].min().eq(1).all()
