import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge.app.core.state import NodeState, NodePhase  # noqa: E402
from edge.app.services.algorithms.finish_line import FinishLineAlgo  # noqa: E402
from edge.app.services.algorithms.race_line import load_line_definition  # noqa: E402
from edge.app.services.algorithms.violation import ViolationAlgo  # noqa: E402


def _make_keypoints(left_ankle, right_ankle, score: float = 0.9):
    kps = [(0.0, 0.0, 0.0) for _ in range(17)]
    kps[15] = (float(left_ankle[0]), float(left_ankle[1]), score)
    kps[16] = (float(right_ankle[0]), float(right_ankle[1]), score)
    return kps


def test_line_file_scales_to_runtime_size(tmp_path):
    line_file = tmp_path / "start_line.json"
    line_file.write_text(
        '{"frame_width":1280,"frame_height":640,"p1":[0,480],"p2":[1279,480]}',
        encoding="utf-8",
    )
    line = load_line_definition(
        frame_width=640,
        frame_height=320,
        line_file=str(line_file),
        fallback_y=480,
        line_name="start_line",
    )
    assert line["p1"] == [0, 240]
    assert line["p2"] == [640 - 1, 240]


def test_false_start_uses_ankle_crossing(monkeypatch):
    monkeypatch.setenv("START_LINE_Y", "480")
    state = NodeState(node_id=1, session_id="RUN_TEST", phase=NodePhase.MONITORING)
    state.expected_start_time = 10_000
    state.config = {"ready_ts": 9_000, "false_start_check": True, "lane_count": 1}
    monkeypatch.setattr(ViolationAlgo, "_load_model", lambda self: None)
    algo = ViolationAlgo(state)
    algo._names = ["person"]

    frame = np.zeros((640, 1280, 3), dtype=np.uint8)
    boxes = [{"bbox": [80, 100, 140, 520], "class_id": 0, "score": 0.9}]
    kps = [_make_keypoints((100, 490), (120, 470))]

    events = algo.process_frame_logic(frame, [1], boxes, kps, 9_500)
    assert events
    item = events[0]["data"][0]
    assert item["event"] == "FALSE_START"
    assert item["ankle_points"][0]["ankle"][1] == 490.0


def test_finish_line_uses_ankle_crossing_timestamp(monkeypatch):
    monkeypatch.setenv("FINISH_LINE_Y", "520")
    state = NodeState(node_id=2, session_id="RUN_FINISH", phase=NodePhase.MONITORING)
    state.expected_start_time = 1_000
    algo = FinishLineAlgo(state)

    dets = [{"bbox": [80, 100, 140, 520], "keypoints": _make_keypoints((100, 510), (120, 505))}]
    assert algo.process_detections(dets, [7], 1_500, frame_shape=(640, 1280)) is None

    dets = [{"bbox": [80, 100, 140, 540], "keypoints": _make_keypoints((100, 530), (120, 525))}]
    report = algo.process_detections(dets, [7], 1_650, frame_shape=(640, 1280))
    assert report is not None
    assert report["data"][0]["finish_ts"] == 1650
