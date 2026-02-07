import time
from typing import Callable

from fastapi import HTTPException

from common.protocol import CommandPayload
from ..core.config import get_settings
from ..core.state import NodePhase, NodeState


class CommandHandler:
    def __init__(self):
        self.settings = get_settings()
        self.state = NodeState(node_id=self.settings.node_id)
        self._dispatch_map: dict[str, Callable[[CommandPayload], None]] = {
            "CMD_INIT": self._handle_init,
            "CMD_BINDING_SYNC": self._handle_binding_sync,
            "CMD_START_MONITOR": self._handle_start_monitor,
            "CMD_STOP": self._handle_stop,
            "CMD_HEARTBEAT": self._handle_heartbeat,
        }

    def handle(self, payload: CommandPayload) -> None:
        handler = self._dispatch_map.get(payload.cmd, self._handle_unknown)
        handler(payload)

    # --- command handlers ---
    def _handle_init(self, payload: CommandPayload) -> None:
        self.state.session_id = payload.session_id
        self.state.phase = NodePhase.BINDING
        self.state.config = payload.config or {}
        self.state.bindings = []
        self.state.expected_start_time = None
        self.state.stop_reason = None
        self._touch(payload.cmd)

    def _handle_binding_sync(self, payload: CommandPayload) -> None:
        self._ensure_same_session(payload)
        bindings = payload.config.get("bindings") if payload.config else None
        self.state.bindings = bindings or []
        self.state.phase = NodePhase.BINDING
        self._touch(payload.cmd)

    def _handle_start_monitor(self, payload: CommandPayload) -> None:
        self._ensure_same_session(payload)
        self.state.phase = NodePhase.MONITORING
        self.state.expected_start_time = (payload.config or {}).get("expected_start_time")
        self._touch(payload.cmd)

    def _handle_stop(self, payload: CommandPayload) -> None:
        self._ensure_same_session(payload)
        self.state.phase = NodePhase.STOPPED
        self.state.stop_reason = (payload.config or {}).get("reason")
        self._touch(payload.cmd)

    def _handle_heartbeat(self, payload: CommandPayload) -> None:
        self._ensure_same_session(payload, allow_empty=True)
        self._touch(payload.cmd)

    def _handle_unknown(self, payload: CommandPayload) -> None:
        self._touch(payload.cmd)

    # --- helpers ---
    def _ensure_same_session(self, payload: CommandPayload, allow_empty: bool = False) -> None:
        if allow_empty and not self.state.session_id:
            return
        if self.state.session_id and self.state.session_id != payload.session_id:
            raise HTTPException(status_code=409, detail="Session mismatch on node")

    def _touch(self, cmd: str) -> None:
        self.state.last_command = cmd
        self.state.last_updated_ms = int(time.time() * 1000)

    def snapshot(self) -> dict:
        return self.state.model_dump()
