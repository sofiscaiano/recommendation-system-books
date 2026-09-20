from __future__ import annotations

import pandas as pd

from quelibroleo.config import RankerConfig
from quelibroleo.features import FeatureBuilder, attach_relevance
from quelibroleo.ranker import LambdaRanker


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
    source = fused.rename(
        columns={"rrf_score": "score", "retrieval_rank": "rank"}
    )
    source["source"] = "mock"
    target = pd.DataFrame(
        [
            ("u1", "b4", 10),
            ("u2", "b1", 8),
            ("u3", "b3", 9),
        ],
        columns=["id_lector", "id_libro", "rating"],
    )

    features = FeatureBuilder(recent_n=2).fit(train, catalog).transform(
        fused, source
    )
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
