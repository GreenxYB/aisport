import pathlib
import sys

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
