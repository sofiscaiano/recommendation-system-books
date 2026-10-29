"""El diagnóstico atribuye la caída del NDCG a retrieval o a ranking."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostico import cascade, summarize, user_metrics


def test_a_missed_book_is_retrieval_and_a_buried_book_is_ranking() -> None:
    books = pd.DataFrame(
        {
            "id_libro": ["a", "b", "c"],
            "autor": ["X", "X", "Y"],
            "genero": ["G", "G", "H"],
        }
    )
    history = pd.DataFrame({"id_lector": ["u"], "id_libro": ["a"]})
    test = pd.DataFrame(
        {"id_lector": ["u", "u"], "id_libro": ["b", "c"], "rating": [10, 9]}
    )
    recommendations = pd.DataFrame(
        {"id_lector": ["u", "u"], "id_libro": ["otro", "b"], "rank": [1, 2]}
    )
    hits = pd.DataFrame(
        {
            "id_lector": ["u", "u"],
            "id_libro": ["b", "c"],
            "rating": [10, 9],
            "in_retrieval": [1, 0],
        }
    )

    frame = user_metrics(recommendations, test, history, books, {"u"}, hits)
    row = frame.iloc[0]

    assert row["poblacion"] == "ejemplo.csv"
    assert row["autores"] == "repite la mayoría"
    assert row["generos"] == "repite la mayoría"
    assert row["recall"] == 0.5
    assert row["en_top"] == 1
    assert row["oracle"] > row["ndcg"]
    assert summarize(frame, "autores").iloc[0]["falla"] == "retrieval"
    labels = cascade(frame)["grupo"].tolist()
    assert labels[0] == "hold-out"
    assert labels[1].strip() == "ejemplo.csv"
    assert any(label.strip().startswith("géneros") for label in labels)
