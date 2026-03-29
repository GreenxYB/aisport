import pathlib
import sys

from fastapi.testclient import TestClient

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cloud.app.main import create_app  # noqa: E402
from cloud.app.services.node_connection_manager import get_node_manager  # noqa: E402
from cloud.app.routers.sessions import get_service  # noqa: E402
from cloud.app.services.orchestrator import get_orchestrator  # noqa: E402


def build_client():
    if hasattr(get_node_manager, "_manager"):
        delattr(get_node_manager, "_manager")
    if hasattr(get_service, "_svc"):
        delattr(get_service, "_svc")
    if hasattr(get_orchestrator, "_orchestrator"):
        delattr(get_orchestrator, "_orchestrator")
    return TestClient(create_app())


def test_node_websocket_register_dispatch_and_status_roundtrip():
    client = build_client()

    with client.websocket_connect("/nodes/ws") as websocket:
        websocket.send_json(
            {
                "node_id": 1,
                "node_role": "START",
                "site_id": "lab-a",
                "capabilities": ["camera", "speaker"],
            }
        )
        ack = websocket.receive_json()
        assert ack == {"type": "CONNECTED", "node_id": 1}

        resp = client.post(
            "/nodes/1/dispatch",
            json={
                "cmd": "CMD_INIT",
                "session_id": "RUN_TEST_001",
                "node_id": 1,
                "task_mode": "TRACK_RACE",
                "config": {"lane_count": 8},
            },
        )
        assert resp.status_code == 200
        pushed = websocket.receive_json()
        assert pushed["cmd"] == "CMD_INIT"
        assert pushed["session_id"] == "RUN_TEST_001"

        websocket.send_json(
            {
                "msg_type": "NODE_STATUS",
                "node_id": 1,
                "session_id": "RUN_TEST_001",
                "timestamp": 1234567890,
                "data": {
                    "phase": "BINDING",
                    "binding_ready": True,
                    "camera_ready": True,
                },
            }
        )
        websocket.send_json(
            {
                "msg_type": "COMMAND_ACK",
                "node_id": 1,
                "session_id": "RUN_TEST_001",
                "timestamp": 1234567891,
                "cmd": "CMD_INIT",
                "status": "accepted",
                "phase": "BINDING",
            }
        )

        online = client.get("/nodes/online")
        assert online.status_code == 200
        nodes = online.json()["nodes"]
        assert len(nodes) == 1
        assert nodes[0]["node_id"] == 1
        assert nodes[0]["online"] is True
        assert nodes[0]["last_status"]["data"]["phase"] == "BINDING"
        assert nodes[0]["last_ack"]["cmd"] == "CMD_INIT"


def test_dispatch_offline_node_rejected():
    client = build_client()
    resp = client.post(
        "/nodes/99/dispatch",
        json={
            "cmd": "CMD_HEARTBEAT",
            "session_id": "RUN_TEST_001",
            "node_id": 99,
            "config": {},
        },
    )
    assert resp.status_code == 409


def test_session_readiness_and_start_dispatch():
    client = build_client()

    create = client.post(
        "/sessions",
        json={
            "project_type": "200m",
            "lane_count": 8,
            "start_node_id": 1,
            "finish_node_id": 2,
            "tracking_node_ids": [3],
            "auto_start": False,
            "sync_time_ms": 1738416000000,
        },
    )
    assert create.status_code == 200
    session_id = create.json()["session_id"]

    sockets = []
    for node_id, role in [(1, "START"), (2, "FINISH"), (3, "MID")]:
        ws = client.websocket_connect("/nodes/ws")
        ws.__enter__()
        ws.send_json(
            {
                "node_id": node_id,
                "node_role": role,
                "site_id": "lab-a",
                "capabilities": ["camera"],
            }
        )
        _ = ws.receive_json()
        ws.send_json(
            {
                "msg_type": "NODE_STATUS",
                "node_id": node_id,
                "session_id": session_id,
                "timestamp": 1234567890 + node_id,
                "data": {
                    "phase": "BINDING",
                    "binding_ready": True,
                    "camera_ready": True,
                },
            }
        )
        sockets.append(ws)

    readiness = client.get(f"/sessions/{session_id}/readiness")
    assert readiness.status_code == 200
    assert readiness.json()["all_ready"] is True

    start = client.post(
        f"/sessions/{session_id}/commands/start-monitor",
        json={
            "expected_start_time": 1773000005000,
            "countdown_seconds": 3,
            "tracking_active": True,
            "audio_plan": "START_321_GO",
        },
    )
    assert start.status_code == 200

    received = []
    for ws in sockets:
        received.append(ws.receive_json())
        ws.__exit__(None, None, None)

    assert {item["node_id"] for item in received} == {1, 2, 3}
    assert all(item["cmd"] == "CMD_START_MONITOR" for item in received)


def test_session_auto_orchestrates_init_binding_and_start():
    with build_client() as client:
        sockets = []
        for node_id, role in [(1, "START"), (2, "FINISH")]:
            ws = client.websocket_connect("/nodes/ws")
            ws.__enter__()
            ws.send_json(
                {
                    "node_id": node_id,
                    "node_role": role,
                    "site_id": "lab-a",
                    "capabilities": ["camera"],
                }
            )
            ack = ws.receive_json()
            assert ack == {"type": "CONNECTED", "node_id": node_id}
            sockets.append(ws)

        create = client.post(
            "/sessions",
            json={
                "project_type": "200m",
                "lane_count": 8,
                "start_node_id": 1,
                "finish_node_id": 2,
                "tracking_node_ids": [],
                "bindings": [{"lane": 1, "student_id": "S101", "feature_id": "F001"}],
                "auto_start": True,
                "start_delay_ms": 3000,
                "countdown_seconds": 3,
                "audio_plan": "START_321_GO",
                "tracking_active": True,
                "sync_time_ms": 1738416000000,
            },
        )
        assert create.status_code == 200
        session_id = create.json()["session_id"]

        for expected_node_id, ws in zip([1, 2], sockets):
            init_cmd = ws.receive_json()
            binding_cmd = ws.receive_json()
            assert init_cmd["cmd"] == "CMD_INIT"
            assert init_cmd["node_id"] == expected_node_id
            assert init_cmd["session_id"] == session_id
            assert binding_cmd["cmd"] == "CMD_BINDING_SYNC"
            assert binding_cmd["node_id"] == expected_node_id
            assert binding_cmd["config"]["bindings"][0]["student_id"] == "S101"

            ws.send_json(
                {
                    "msg_type": "NODE_STATUS",
                    "node_id": expected_node_id,
                    "session_id": session_id,
                    "timestamp": 1234567890 + expected_node_id,
                    "data": {
                        "phase": "BINDING",
                        "binding_ready": True,
                        "camera_ready": True,
                    },
                }
            )

        for expected_node_id, ws in zip([1, 2], sockets):
            start_cmd = ws.receive_json()
            assert start_cmd["cmd"] == "CMD_START_MONITOR"
            assert start_cmd["node_id"] == expected_node_id
            assert start_cmd["session_id"] == session_id
            assert start_cmd["config"]["countdown_seconds"] == 3
            assert start_cmd["config"]["audio_plan"] == "START_321_GO"
            assert start_cmd["config"]["tracking_active"] is True
            assert isinstance(start_cmd["config"]["expected_start_time"], int)
            ws.__exit__(None, None, None)
