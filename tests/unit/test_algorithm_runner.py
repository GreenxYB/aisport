import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge.app.core.config import get_settings  # noqa: E402
from edge.app.core.state import NodePhase, NodeState  # noqa: E402
from edge.app.services.algorithms.runner import AlgorithmRunner  # noqa: E402


class DummyTrackerResult:
    def __init__(self, result, keypoints):
        self.result = result
        self.keypoints = keypoints


def test_start_node_pipeline_runs_face_binding_and_false_start(monkeypatch):
    monkeypatch.setenv("NODE_ROLE", "START")
    monkeypatch.setenv("FACE_REPORT_INTERVAL_SEC", "0")
    get_settings.cache_clear()

    state = NodeState(
        node_id=1,
        session_id="RUN_TEST_001",
        phase=NodePhase.MONITORING,
        expected_start_time=10_000,
        bindings=[{"lane": 1, "student_id": "S101"}],
    )
    state.config["tracking_active"] = True

    runner = AlgorithmRunner(state)
    runner.face.process = lambda frame, ts_ms: [{"msg_type": "ID_REPORT", "data": [{"student_id": "S101"}]}]
    runner.finish.process_detections = lambda **kwargs: None

    called = {}

    def fake_violation(frame, track_ids, boxes, kps, current_time):
        called["track_ids"] = track_ids
        called["boxes"] = boxes
        called["kps"] = kps
        return [{"msg_type": "VIOLATION_EVENT", "data": [{"event": "FALSE_START", "lane": 1}]}]

    runner.violation.process_frame_logic = fake_violation

    tracker_result = DummyTrackerResult(
        result=np.array([[10, 20, 30, 40, 7, 0.95, 0]], dtype=float),
        keypoints=[[[1, 2, 0.9]]],
    )
    events = runner.process_pipeline_result(
        np.zeros((640, 1280, 3), dtype=np.uint8),
        tracker_result,
        9_000,
    )

    msg_types = {item["msg_type"] for item in events}
    assert "ID_REPORT" in msg_types
    assert "VIOLATION_EVENT" in msg_types
    assert called["track_ids"] == [7]
    assert called["boxes"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]

    get_settings.cache_clear()


def test_finish_node_pipeline_only_emits_finish_report(monkeypatch):
    monkeypatch.setenv("NODE_ROLE", "FINISH")
    get_settings.cache_clear()

    state = NodeState(
        node_id=2,
        session_id="RUN_TEST_002",
        phase=NodePhase.MONITORING,
        expected_start_time=1_000,
    )
    state.config["tracking_active"] = True

    runner = AlgorithmRunner(state)

    def unexpected_face(*args, **kwargs):
        raise AssertionError("finish node should not call face binding")

    def unexpected_violation(*args, **kwargs):
        raise AssertionError("finish node should not call violation logic")

    runner.face.process = unexpected_face
    runner.violation.process_frame_logic = unexpected_violation
    runner.finish.process_detections = lambda **kwargs: {
        "msg_type": "FINISH_REPORT",
        "data": [{"lane": 1, "finish_ts": 2_500, "rank": 1}],
    }

    tracker_result = DummyTrackerResult(
        result=np.array([[10, 20, 30, 40, 12, 0.88, 0]], dtype=float),
        keypoints=[[[1, 2, 0.9]]],
    )
    events = runner.process_pipeline_result(
        np.zeros((640, 1280, 3), dtype=np.uint8),
        tracker_result,
        2_000,
    )

    assert len(events) == 1
    assert events[0]["msg_type"] == "FINISH_REPORT"
    assert events[0]["data"][0]["rank"] == 1

    get_settings.cache_clear()
