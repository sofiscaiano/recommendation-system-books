from __future__ import annotations

import pandas as pd
import pytest

from quelibroleo.config import PipelineConfig
from quelibroleo.models import MODEL_NAMES, PopularityModel, build_model
from quelibroleo.split_data import validation_folds


def _history() -> pd.DataFrame:
    rows = []
    for user in ("u1", "u2"):
        for index in range(40):
            rows.append(
                {
                    "id_lector": user,
                    "id_libro": f"{user}-b{index}",
                    "fecha": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index),
                    "rating": 8,
                }
            )
    return pd.DataFrame(rows)


def test_validation_folds_do_not_depend_on_the_model() -> None:
    history = _history()
    config = PipelineConfig()
    first = validation_folds(history, config)
    second = validation_folds(history, config)

    assert first
    assert [fold.fold for fold in first] == [fold.fold for fold in second]
    assert [fold.query_users for fold in first] == [fold.query_users for fold in second]
    assert first[-1].target.equals(second[-1].target)
    assert len(first[-1].target) == config.split.rolling_horizon * 2


def test_build_model_exposes_the_same_names() -> None:
    config = PipelineConfig()
    assert isinstance(build_model("popularity", config), PopularityModel)
    assert build_model("two_stage", config).name == "two_stage"
    assert build_model("ease", config).name == "ease"
    assert build_model("two_tower", config).name == "two_tower"
    assert build_model("rating_mlp", config).name == "rating_mlp"
    assert "two_stage" in MODEL_NAMES
    assert "ease" in MODEL_NAMES
    assert "two_tower" in MODEL_NAMES
    assert "rating_mlp" in MODEL_NAMES
    with pytest.raises(ValueError, match="desconocido"):
        build_model("otro", config)
