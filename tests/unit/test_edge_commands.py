import pathlib
import sys
import numpy as np
import pytest

from fastapi.testclient import TestClient

# Ensure src on path for tests
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge.app.main import create_app  # noqa: E402
from edge.app.routers.commands import get_handler  # noqa: E402
from edge.app.core.state import NodeState  # noqa: E402


def reset_state():
    handler = get_handler()
    handler.state = NodeState(node_id=handler.settings.node_id)
    return handler


def build_client():
    reset_state()
    app = create_app()
    return TestClient(app)


def test_init_success():
    client = build_client()
    resp = client.post(
        "/commands",
        json={
            "cmd": "CMD_INIT",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
            "task_mode": "IDENTITY_BINDING",
            "config": {"project_type": "200m", "lane_count": 8, "sync_time": 123},
        },
    )
    assert resp.status_code == 200
    status = client.get("/status").json()
    assert status["phase"] == "BINDING"


def test_start_without_init_rejected():
    client = build_client()
    resp = client.post(
        "/commands",
        json={
            "cmd": "CMD_START_MONITOR",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
            "config": {"expected_start_time": 1000},
        },
    )
    assert resp.status_code == 409


def test_session_mismatch_rejected():
    client = build_client()
    client.post(
        "/commands",
        json={
            "cmd": "CMD_INIT",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
            "config": {"project_type": "200m", "lane_count": 8},
        },
    )
    resp = client.post(
        "/commands",
        json={
            "cmd": "CMD_START_MONITOR",
            "session_id": "RUN_OTHER",
            "node_id": 1,
            "config": {"expected_start_time": 1000},
        },
    )
    assert resp.status_code == 409


def test_unknown_command_rejected():
    client = build_client()
    resp = client.post(
        "/commands",
        json={
            "cmd": "CMD_UNKNOWN",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
        },
    )
    assert resp.status_code == 400


def test_full_workflow():
    """测试完整的工作流程"""
    client = build_client()
    
    # 1. 初始化
    resp = client.post("/commands", json={
        "cmd": "CMD_INIT",
        "session_id": "RUN_TEST_001",
        "node_id": 1,
        "config": {"lane_count": 8},
    })
    assert resp.status_code == 200
    
    # 2. 绑定同步
    resp = client.post("/commands", json={
        "cmd": "CMD_BINDING_SYNC",
        "session_id": "RUN_TEST_001",
        "node_id": 1,
        "config": {"bindings": [{"lane": 1, "athlete_id": "A001"}]},
    })
    assert resp.status_code == 200
    
    # 3. 开始监控
    resp = client.post("/commands", json={
        "cmd": "CMD_START_MONITOR",
        "session_id": "RUN_TEST_001",
        "node_id": 1,
    })
    assert resp.status_code == 200
    status = client.get("/status").json()
    assert status["phase"] == "MONITORING"
    
    # 4. 心跳
    resp = client.post("/commands", json={
        "cmd": "CMD_HEARTBEAT",
        "session_id": "RUN_TEST_001",
        "node_id": 1,
    })
    assert resp.status_code == 200
    
    # 5. 停止
    resp = client.post("/commands", json={
        "cmd": "CMD_STOP",
        "session_id": "RUN_TEST_001",
        "node_id": 1,
        "config": {"reason": "测试结束"},
    })
    assert resp.status_code == 200
    status = client.get("/status").json()
    assert status["phase"] == "STOPPED"


def test_reset_round_keeps_bindings_and_returns_to_binding():
    client = build_client()

    client.post(
        "/commands",
        json={
            "cmd": "CMD_INIT",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
            "task_mode": "TRACK_RACE",
            "config": {"lane_count": 8},
        },
    )
    client.post(
        "/commands",
        json={
            "cmd": "CMD_BINDING_SYNC",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
            "config": {"bindings": [{"lane": 1, "student_id": "S101"}]},
        },
    )
    client.post(
        "/commands",
        json={
            "cmd": "CMD_START_MONITOR",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
            "config": {"expected_start_time": 1000},
        },
    )

    resp = client.post(
        "/commands",
        json={
            "cmd": "CMD_RESET_ROUND",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
            "config": {"reason": "FALSE_START"},
        },
    )
    assert resp.status_code == 200

    status = client.get("/status").json()
    assert status["phase"] == "BINDING"
    assert status["bindings"] == [{"lane": 1, "student_id": "S101"}]
    assert status["expected_start_time"] is None


def test_status_binding_ready_requires_real_face_confirmation():
    client = build_client()
    client.post(
        "/commands",
        json={
            "cmd": "CMD_INIT",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
            "config": {"lane_count": 8},
        },
    )
    client.post(
        "/commands",
        json={
            "cmd": "CMD_BINDING_SYNC",
            "session_id": "RUN_TEST_001",
            "node_id": 1,
            "config": {
                "bindings": [
                    {"lane": 1, "student_id": "S101"},
                    {"lane": 2, "student_id": "S102"},
                ]
            },
        },
    )

    handler = get_handler()
    status = handler.build_status_report().model_dump()
    assert status["data"]["binding_required"] is True
    assert status["data"]["binding_ready"] is False
    assert status["data"]["binding_target_count"] == 2
    assert status["data"]["binding_confirmed_count"] == 0

    handler.state.binding_confirmed_students = ["S101", "S102"]
    handler.state.binding_confirmed_lanes = [1, 2]
    handler.state.binding_assignments = [
        {"lane": 1, "student_id": "S101"},
        {"lane": 2, "student_id": "S102"},
    ]
    handler.state.binding_confirmed_at_ms = 1234567890
    status = handler.build_status_report().model_dump()
    assert status["data"]["binding_ready"] is True
    assert status["data"]["binding_confirmed_count"] == 2
    assert status["data"]["binding_pending_count"] == 0


def test_preview_snapshot_uses_pipeline_cache():
    client = build_client()
    handler = get_handler()
    handler.pipeline.snapshot_jpeg = lambda: b"fake-jpeg"
    handler.pipeline.last_encode_error = lambda: None

    resp = client.get("/preview/snapshot")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"fake-jpeg"


def test_results_boolean_mask_keeps_lengths_aligned():
    Results = pytest.importorskip("edge.app.services.pipeline").Results
    frame = [[0]]
    result = Results(
        orig_img=frame,
        confs=[0.9, 0.8, 0.7],
        boxes=[[1, 1, 2, 2], [3, 3, 4, 4], [5, 5, 6, 6]],
        cls=[0, 0, 0],
        keypoints=[["a"], ["b"], ["c"]],
    )

    subset = result[np.array([True, False, True])]

    assert len(subset.confs) == 2
    assert len(subset.boxes) == 2
    assert len(subset.cls) == 2
    assert len(subset.keypoints) == 2
    assert subset.keypoints == [["a"], ["c"]]
