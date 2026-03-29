import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import WebSocket

from common.protocol import CommandPayload, FinishReport, NodeConnectPayload, NodeStatusReport, ViolationReport


class NodeConnectionManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._connections: dict[int, WebSocket] = {}
        self._meta: dict[int, dict[str, Any]] = {}
        self._status: dict[int, dict[str, Any]] = {}
        self._last_ack: dict[int, dict[str, Any]] = {}
        self._last_id_report: dict[int, dict[str, Any]] = {}
        self._last_violation: dict[int, dict[str, Any]] = {}
        self._last_finish: dict[int, dict[str, Any]] = {}
        self._id_reports_by_session: dict[str, list[dict[str, Any]]] = {}
        self._violations_by_session: dict[str, list[dict[str, Any]]] = {}
        self._finishes_by_session: dict[str, list[dict[str, Any]]] = {}

    async def register(self, websocket: WebSocket, payload: NodeConnectPayload) -> None:
        async with self._lock:
            self._connections[payload.node_id] = websocket
            self._meta[payload.node_id] = {
                "node_id": payload.node_id,
                "node_role": payload.node_role,
                "site_id": payload.site_id,
                "capabilities": payload.capabilities,
                "connected_at": self._now_iso(),
                "online": True,
            }

    async def unregister(self, node_id: int) -> None:
        async with self._lock:
            self._connections.pop(node_id, None)
            meta = self._meta.get(node_id)
            if meta is not None:
                meta["online"] = False
                meta["disconnected_at"] = self._now_iso()

    async def update_status(self, report: NodeStatusReport) -> None:
        async with self._lock:
            self._status[report.node_id] = {
                "session_id": report.session_id,
                "timestamp": report.timestamp,
                "data": report.data,
                "updated_at": self._now_iso(),
            }

    async def record_violation(self, report: ViolationReport) -> None:
        async with self._lock:
            payload = report.model_dump()
            self._last_violation[report.node_id] = payload
            self._violations_by_session.setdefault(report.session_id, []).append(payload)

    async def record_finish(self, report: FinishReport) -> None:
        async with self._lock:
            payload = report.model_dump()
            self._last_finish[report.node_id] = payload
            self._finishes_by_session.setdefault(report.session_id, []).append(payload)

    async def record_ack(self, ack: dict[str, Any]) -> None:
        async with self._lock:
            self._last_ack[int(ack["node_id"])] = ack

    async def record_id_report(self, report: dict[str, Any]) -> None:
        async with self._lock:
            self._last_id_report[int(report["node_id"])] = report
            session_id = str(report.get("session_id") or "")
            if session_id:
                self._id_reports_by_session.setdefault(session_id, []).append(report)

    async def send_command(self, node_id: int, payload: CommandPayload) -> bool:
        async with self._lock:
            websocket = self._connections.get(node_id)
        if websocket is None:
            return False
        await websocket.send_json(payload.model_dump())
        return True

    async def list_online(self) -> list[dict[str, Any]]:
        async with self._lock:
            rows = []
            for node_id, meta in self._meta.items():
                rows.append(
                    {
                        **meta,
                        "last_status": self._status.get(node_id),
                        "last_ack": self._last_ack.get(node_id),
                        "last_id_report": self._last_id_report.get(node_id),
                        "last_violation": self._last_violation.get(node_id),
                        "last_finish": self._last_finish.get(node_id),
                    }
                )
            return sorted(rows, key=lambda item: item["node_id"])

    async def get_session_reports(self, session_id: str) -> dict[str, list[dict[str, Any]]]:
        async with self._lock:
            return {
                "id_reports": list(self._id_reports_by_session.get(session_id, [])),
                "violations": list(self._violations_by_session.get(session_id, [])),
                "finishes": list(self._finishes_by_session.get(session_id, [])),
            }

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


def get_node_manager() -> NodeConnectionManager:
    if not hasattr(get_node_manager, "_manager"):
        get_node_manager._manager = NodeConnectionManager()
    return get_node_manager._manager  # type: ignore[attr-defined]
