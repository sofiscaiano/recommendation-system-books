"""La red única queda registrada como modelo propio."""

from __future__ import annotations

from quelibroleo.config import PipelineConfig
from quelibroleo.models import MODEL_NAMES, build_model


def test_rating_mlp_is_selectable() -> None:
    assert "rating_mlp" in MODEL_NAMES
    assert build_model("rating_mlp", PipelineConfig()).name == "rating_mlp"
