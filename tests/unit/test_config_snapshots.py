from __future__ import annotations

from pathlib import Path

from dibo.schemas import load_actions, load_graph, load_parameters

CONFIGS = Path(__file__).parents[2] / "configs"

EXPECTED_ANCHORS = {
    "A01": ("p06",),
    "A02": ("p03",),
    "A03": ("p04",),
    "A04": ("p12",),
    "A05": ("p13",),
    "A06": ("p15",),
    "A07": ("p05",),
    "A08": ("p01", "p14"),
}

EXPECTED_W = {
    "A01": (0.30, 0.15, -0.85, -0.45, -0.35, 1.00, 0.35, 0.30, -0.25, -0.20, -0.15, 0.30, 0.10, -0.05, 0.15),
    "A02": (0.15, -0.20, 1.00, 0.85, 0.15, 0.45, 0.10, -0.25, 0.35, 0.20, 0.10, 0.40, 0.20, -0.10, -0.35),
    "A03": (0.15, -0.10, 0.30, 1.00, 0.10, 0.25, 0.05, -0.20, 0.35, 0.25, 0.30, 0.70, 0.05, -0.10, -0.25),
    "A04": (0.05, 0.10, 0.15, -0.55, -0.10, 0.15, 0.05, -0.05, 0.85, -0.70, -0.65, 1.00, 0.10, -0.05, -0.15),
    "A05": (-0.05, -0.03, 0.25, 0.20, -0.30, 0.40, 0.05, -0.05, 0.10, 0.05, -0.10, 0.15, 1.00, -0.03, -0.10),
    "A06": (-0.10, -0.10, 0.30, 0.20, 0.05, -0.20, -0.05, -0.10, 0.10, 0.05, 0.05, 0.15, 0.05, -0.10, -1.00),
    "A07": (0.05, 0.05, 0.35, 0.15, -1.00, 0.30, 0.10, 0.05, 0.10, 0.05, -0.05, 0.10, 0.30, -0.03, -0.05),
    "A08": (1.00, 0.45, 0.20, 0.35, 0.05, 0.25, 0.10, -0.25, 0.10, 0.05, 0.05, 0.15, 0.05, -1.00, -0.30),
}

EXPECTED_NEIGHBORS = {
    "m01": ("A01", "A02", "A05", "A07"),
    "m02": ("A01", "A02", "A07"),
    "m03": ("A02", "A03", "A04", "A06", "A08"),
    "m04": ("A02", "A03", "A04", "A05"),
    "m05": ("A03", "A07", "A08"),
    "m06": ("A03", "A04", "A05", "A06", "A08"),
}


def test_parameter_snapshot() -> None:
    catalog = load_parameters(CONFIGS / "parameters.yaml")
    by_id = {item.parameter_id: item for item in catalog.parameters}

    assert len(by_id) == 15
    assert by_id["p03"].initial == 128
    assert by_id["p04"].initial == "PROMPT_P95_OR_8192"
    assert by_id["p06"].action_scale == 0.04
    assert by_id["p08"].initial == 2.0
    assert by_id["p11"].alignment == 256
    assert by_id["p13"].initial == "FROM_REQUEST_PREFIX_PROFILE"


def test_action_matrix_and_anchor_snapshot() -> None:
    bundle = load_actions(CONFIGS / "actions_v1.yaml")

    assert {action.action_id: action.direction_vector for action in bundle.actions} == EXPECTED_W
    assert {action.action_id: action.anchor_parameter_ids for action in bundle.actions} == EXPECTED_ANCHORS


def test_graph_snapshot_and_reverse_neighbors() -> None:
    graph = load_graph(CONFIGS / "graph.yaml")

    assert sum(sum(row) for row in graph.adjacency) == 24
    assert {metric: graph.neighbors(metric) for metric in graph.metric_order} == EXPECTED_NEIGHBORS
