import pathlib
import sys
import time

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


def test_session_auto_orchestration_waits_for_start_binding_only():
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
            _ = ws.receive_json()
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
            },
        )
        assert create.status_code == 200
        session_id = create.json()["session_id"]

        for ws in sockets:
            _ = ws.receive_json()  # CMD_INIT
            _ = ws.receive_json()  # CMD_BINDING_SYNC

        sockets[0].send_json(
            {
                "msg_type": "NODE_STATUS",
                "node_id": 1,
                "session_id": session_id,
                "timestamp": 1234567891,
                "data": {
                    "phase": "BINDING",
                    "binding_ready": False,
                    "camera_ready": True,
                    "node_role": "START",
                },
            }
        )
        sockets[1].send_json(
            {
                "msg_type": "NODE_STATUS",
                "node_id": 2,
                "session_id": session_id,
                "timestamp": 1234567892,
                "data": {
                    "phase": "BINDING",
                    "binding_ready": False,
                    "camera_ready": True,
                    "node_role": "FINISH",
                },
            }
        )

        readiness = client.get(f"/sessions/{session_id}/readiness")
        assert readiness.status_code == 200
        assert readiness.json()["all_ready"] is False

        sockets[0].send_json(
            {
                "msg_type": "NODE_STATUS",
                "node_id": 1,
                "session_id": session_id,
                "timestamp": 1234567893,
                "data": {
                    "phase": "BINDING",
                    "binding_ready": True,
                    "camera_ready": True,
                    "node_role": "START",
                },
            }
        )

        start_cmd_1 = sockets[0].receive_json()
        start_cmd_2 = sockets[1].receive_json()
        assert start_cmd_1["cmd"] == "CMD_START_MONITOR"
        assert start_cmd_2["cmd"] == "CMD_START_MONITOR"

        for ws in sockets:
            ws.__exit__(None, None, None)


def test_session_results_aggregate_finish_and_false_start():
    client = build_client()

    create = client.post(
        "/sessions",
        json={
            "project_type": "200m",
            "lane_count": 8,
            "start_node_id": 1,
            "finish_node_id": 2,
            "tracking_node_ids": [],
            "bindings": [
                {"lane": 1, "student_id": "S101", "feature_id": "F001"},
                {"lane": 2, "student_id": "S102", "feature_id": "F002"},
            ],
            "auto_start": False,
        },
    )
    assert create.status_code == 200
    session_id = create.json()["session_id"]

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

    for ws in sockets:
        _ = ws.receive_json()

    violation = client.post(
        "/nodes/reports/violation",
        json={
            "msg_type": "VIOLATION_EVENT",
            "node_id": 1,
            "session_id": session_id,
            "timestamp": 1773000004000,
            "data": [
                {
                    "event": "FALSE_START",
                    "lane": 1,
                    "track_id": 11,
                }
            ],
        },
    )
    assert violation.status_code == 200

    finish = client.post(
        "/nodes/reports/finish",
        json={
            "msg_type": "FINISH_REPORT",
            "node_id": 2,
            "session_id": session_id,
            "timestamp": 1773000013000,
            "data": [
                {"lane": 1, "track_id": 11, "rank": 1, "finish_ts": 1773000012000},
                {"lane": 2, "track_id": 22, "rank": 2, "finish_ts": 1773000012400},
            ],
        },
    )
    assert finish.status_code == 200

    results = client.get(f"/sessions/{session_id}/results")
    assert results.status_code == 200
    payload = results.json()
    assert payload["expected_start_time"] == 1773000005000
    assert len(payload["results"]) == 2
    lane1 = next(item for item in payload["results"] if item["lane"] == 1)
    lane2 = next(item for item in payload["results"] if item["lane"] == 2)
    assert lane1["student_id"] == "S101"
    assert lane1["false_start"] is True
    assert lane1["elapsed_ms"] == 7000
    assert lane1["rank"] == 1
    assert lane2["student_id"] == "S102"
    assert lane2["false_start"] is False
    assert lane2["elapsed_ms"] == 7400

    for ws in sockets:
        ws.__exit__(None, None, None)


def test_session_diagnostics_exposes_workflow_and_node_status():
    with build_client() as client:
        ws = client.websocket_connect("/nodes/ws")
        ws.__enter__()
        ws.send_json(
            {
                "node_id": 1,
                "node_role": "START",
                "site_id": "lab-a",
                "capabilities": ["camera"],
            }
        )
        _ = ws.receive_json()

        create = client.post(
            "/sessions",
            json={
                "project_type": "200m",
                "lane_count": 8,
                "start_node_id": 1,
                "finish_node_id": 1,
                "tracking_node_ids": [],
                "bindings": [{"lane": 1, "student_id": "S101"}],
                "auto_start": False,
            },
        )
        session_id = create.json()["session_id"]

        _ = ws.receive_json()
        _ = ws.receive_json()
        ws.send_json(
            {
                "msg_type": "NODE_STATUS",
                "node_id": 1,
                "session_id": session_id,
                "timestamp": 1234567890,
                "data": {
                    "phase": "BINDING",
                    "session_stage": "WAIT_BINDING",
                    "binding_ready": False,
                    "binding_required": True,
                    "binding_target_count": 1,
                    "binding_confirmed_count": 0,
                    "binding_pending_count": 1,
                    "binding_pending_students": ["S101"],
                    "lane_layout_status": {
                        "source": "auto",
                        "calibrated": False,
                        "warning": "lane layout missing; using equal-width fallback",
                        "file": None,
                    },
                    "camera_ready": True,
                    "node_role": "START",
                },
            }
        )

        diagnostics = client.get(f"/sessions/{session_id}/diagnostics")
        assert diagnostics.status_code == 200
        payload = diagnostics.json()
        assert payload["session"]["session_id"] == session_id
        assert payload["workflow"]["init_sent_to"] == [1]
        assert payload["workflow"]["binding_sent_to"] == [1]
        assert payload["readiness"]["all_ready"] is False
        assert payload["nodes"][0]["last_status"]["data"]["session_stage"] == "WAIT_BINDING"
        assert payload["results"]["report_counts"]["finishes"] == 0
        assert payload["warnings"][0]["type"] == "LANE_LAYOUT"

        ws.__exit__(None, None, None)


def test_session_binding_timeout_marks_session_terminal():
    with build_client() as client:
        ws = client.websocket_connect("/nodes/ws")
        ws.__enter__()
        ws.send_json(
            {
                "node_id": 1,
                "node_role": "START",
                "site_id": "lab-a",
                "capabilities": ["camera"],
            }
        )
        _ = ws.receive_json()

        create = client.post(
            "/sessions",
            json={
                "project_type": "200m",
                "lane_count": 8,
                "start_node_id": 1,
                "finish_node_id": 1,
                "tracking_node_ids": [],
                "bindings": [{"lane": 1}],
                "auto_start": True,
                "binding_timeout_sec": 1,
            },
        )
        session_id = create.json()["session_id"]

        _ = ws.receive_json()  # init
        _ = ws.receive_json()  # binding
        ws.send_json(
            {
                "msg_type": "NODE_STATUS",
                "node_id": 1,
                "session_id": session_id,
                "timestamp": 1234567890,
                "data": {
                    "phase": "BINDING",
                    "binding_ready": False,
                    "camera_ready": True,
                    "node_role": "START",
                },
            }
        )

        time.sleep(1.7)
        stop_cmd = ws.receive_json()
        assert stop_cmd["cmd"] == "CMD_STOP"
        assert stop_cmd["config"]["reason"] == "BINDING_TIMEOUT"

        diagnostics = client.get(f"/sessions/{session_id}/diagnostics")
        payload = diagnostics.json()
        assert payload["session"]["status"] == "BINDING_TIMEOUT"
        assert payload["session"]["terminal_reason"] == "No valid binding completed before timeout"
        assert payload["workflow"]["stop_sent"] is True

        results = client.get(f"/sessions/{session_id}/results").json()
        lane1 = next(item for item in results["results"] if item["lane"] == 1)
        assert lane1["result_status"] == "UNBOUND"

        ws.__exit__(None, None, None)


def test_session_race_timeout_marks_missing_lane_dnf():
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
            _ = ws.receive_json()
            sockets.append(ws)

        create = client.post(
            "/sessions",
            json={
                "project_type": "200m",
                "lane_count": 8,
                "start_node_id": 1,
                "finish_node_id": 2,
                "tracking_node_ids": [],
                "bindings": [{"lane": 1}, {"lane": 2}],
                "auto_start": True,
                "binding_timeout_sec": 5,
                "start_delay_ms": 1000,
                "race_timeout_sec": 1,
            },
        )
        session_id = create.json()["session_id"]

        for node_id, ws in zip([1, 2], sockets):
            _ = ws.receive_json()  # init
            _ = ws.receive_json()  # binding
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
                        "node_role": "START" if node_id == 1 else "FINISH",
                    },
                }
            )

        for ws in sockets:
            start_cmd = ws.receive_json()
            assert start_cmd["cmd"] == "CMD_START_MONITOR"

        finish = client.post(
            "/nodes/reports/finish",
            json={
                "msg_type": "FINISH_REPORT",
                "node_id": 2,
                "session_id": session_id,
                "timestamp": 1773000013000,
                "data": [{"lane": 1, "track_id": 11, "rank": 1, "finish_ts": 1773000012000}],
            },
        )
        assert finish.status_code == 200

        time.sleep(2.2)
        for ws in sockets:
            stop_cmd = ws.receive_json()
            assert stop_cmd["cmd"] == "CMD_STOP"
            assert stop_cmd["config"]["reason"] == "RACE_TIMEOUT"

        diagnostics = client.get(f"/sessions/{session_id}/diagnostics")
        payload = diagnostics.json()
        assert payload["session"]["status"] == "RACE_TIMEOUT"
        assert payload["workflow"]["stop_sent"] is True

        results = client.get(f"/sessions/{session_id}/results").json()
        lane1 = next(item for item in results["results"] if item["lane"] == 1)
        lane2 = next(item for item in results["results"] if item["lane"] == 2)
        assert lane1["result_status"] == "OK"
        assert lane2["result_status"] == "DNF"

        for ws in sockets:
            ws.__exit__(None, None, None)
