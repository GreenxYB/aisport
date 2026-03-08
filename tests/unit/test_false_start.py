import pathlib
import sys

import numpy as np

# Ensure src on path for tests
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge.app.core.state import NodeState, NodePhase  # noqa: E402
from edge.app.services.algorithms.violation import ViolationAlgo  # noqa: E402


def _make_keypoints(
    left_knee_y: float,
    left_ankle_y: float,
    right_knee_y: float,
    right_ankle_y: float,
    score: float = 0.9,
):
    kps = [(0.0, 0.0, 0.0) for _ in range(17)]
    kps[13] = (98.0, left_knee_y, score)  # left knee
    kps[14] = (118.0, right_knee_y, score)  # right knee
    kps[15] = (100.0, left_ankle_y, score)  # left ankle
    kps[16] = (120.0, right_ankle_y, score)  # right ankle
    return kps


def _build_algo(monkeypatch, state: NodeState) -> ViolationAlgo:
    monkeypatch.setattr(ViolationAlgo, "_load_model", lambda self: None)
    algo = ViolationAlgo(state)
    algo._names = ["person"]
    return algo


def test_false_start_when_toe_proxy_crosses_line(monkeypatch):
    state = NodeState(node_id=1, session_id="RUN_TEST", phase=NodePhase.MONITORING)
    state.expected_start_time = 10_000
    state.config = {"ready_ts": 9_000, "false_start_check": True, "lane_count": 1}
    algo = _build_algo(monkeypatch, state)

    frame = np.zeros((640, 1280, 3), dtype=np.uint8)
    boxes = [{"bbox": [0, 0, 100, 200], "class_id": 0, "score": 0.9}]
    # With scale=0.45 and line_y=480:
    # left toe_y ~= 470 + 0.45 * (470 - 440) = 483.5 (crossed)
    kps = [_make_keypoints(440.0, 470.0, 430.0, 460.0)]

    events = algo.process_frame_logic(frame, [1], boxes, kps, 9_500)
    assert events, "Expected false start event"
    assert events[0]["event"] == "FALSE_START"
    assert events[0].get("toe_proxy_points")


def test_no_false_start_when_toe_proxy_before_line(monkeypatch):
    state = NodeState(node_id=1, session_id="RUN_TEST", phase=NodePhase.MONITORING)
    state.expected_start_time = 10_000
    state.config = {"ready_ts": 9_000, "false_start_check": True, "lane_count": 1}
    algo = _build_algo(monkeypatch, state)

    frame = np.zeros((640, 1280, 3), dtype=np.uint8)
    boxes = [{"bbox": [0, 0, 100, 200], "class_id": 0, "score": 0.9}]
    # left toe_y ~= 450 + 0.45 * (450 - 440) = 454.5 (not crossed)
    kps = [_make_keypoints(440.0, 450.0, 430.0, 445.0)]

    events = algo.process_frame_logic(frame, [1], boxes, kps, 9_500)
    assert events == []
