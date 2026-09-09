from pathlib import Path

import pytest

from dibo.schemas import load_graph
from dibo.selector import union_neighbor_actions

CONFIGS = Path(__file__).parents[2] / "configs"


def test_known_metric_unions_are_complete_and_canonical() -> None:
    graph = load_graph(CONFIGS / "graph.yaml")

    assert union_neighbor_actions(("m01", "m02"), graph) == ("A01", "A02", "A05", "A07")
    assert union_neighbor_actions(("m03", "m06"), graph) == (
        "A02",
        "A03",
        "A04",
        "A05",
        "A06",
        "A08",
    )
    assert union_neighbor_actions(("m01", "m03", "m06"), graph) == (
        "A01",
        "A02",
        "A03",
        "A04",
        "A05",
        "A06",
        "A07",
        "A08",
    )


def test_union_keeps_a05_and_a08_without_runtime_masks() -> None:
    graph = load_graph(CONFIGS / "graph.yaml")

    assert "A05" in union_neighbor_actions(("m01",), graph)
    assert "A08" in union_neighbor_actions(("m03",), graph)


def test_union_rejects_empty_duplicate_and_unknown_metrics() -> None:
    graph = load_graph(CONFIGS / "graph.yaml")

    with pytest.raises(ValueError, match="at least one"):
        union_neighbor_actions((), graph)
    with pytest.raises(ValueError, match="duplicates"):
        union_neighbor_actions(("m01", "m01"), graph)
    with pytest.raises(ValueError, match="unknown"):
        union_neighbor_actions(("m99",), graph)
