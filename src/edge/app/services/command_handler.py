import json
import logging
import time
from pathlib import Path
from typing import Callable

from fastapi import HTTPException

from common.protocol import CommandPayload
from ..core.config import get_settings
from ..core.state import NodePhase, NodeState
from .camera import CaptureManager


class CommandHandler:
    def __init__(self):
        self.settings = get_settings()
        self.state_file = Path(__file__).resolve().parents[4] / "logs" / "state.json"
        self.state = self._load_state()
        self.logger = logging.getLogger("edge.command")
        self.capture = CaptureManager(on_frame=self._on_frame)
        if self.settings.auto_start_capture:
            self.capture.start()
            self.capture.start_display()
            self.state.capture_running = True
        self.allowed_cmds = {
            "CMD_INIT",
            "CMD_BINDING_SYNC",
            "CMD_START_MONITOR",
            "CMD_STOP",
            "CMD_HEARTBEAT",
        }
        self._dispatch_map: dict[str, Callable[[CommandPayload], None]] = {
            "CMD_INIT": self._handle_init,
            "CMD_BINDING_SYNC": self._handle_binding_sync,
            "CMD_START_MONITOR": self._handle_start_monitor,
            "CMD_STOP": self._handle_stop,
            "CMD_HEARTBEAT": self._handle_heartbeat,
        }

    def handle(self, payload: CommandPayload) -> None:
        started = time.time()
        if payload.cmd not in self.allowed_cmds:
            self.logger.warning("Unknown command %s", payload.cmd)
            raise HTTPException(status_code=400, detail="Unsupported command")
        handler = self._dispatch_map[payload.cmd]
        self.logger.info(
            "recv cmd=%s session=%s node=%s config=%s",
            payload.cmd,
            payload.session_id,
            payload.node_id,
            payload.config,
        )
        self.logger.info("payload_json=%s", payload.model_dump())
        handler(payload)
        self._persist_state()
        elapsed_ms = int((time.time() - started) * 1000)
        self.logger.info(
            "state updated phase=%s last_cmd=%s elapsed_ms=%s",
            self.state.phase,
            self.state.last_command,
            elapsed_ms,
        )

    # --- command handlers ---
    def _handle_init(self, payload: CommandPayload) -> None:
        # allow re-init to reset state; when session changes, reset everything
        self.state.session_id = payload.session_id
        self.state.phase = NodePhase.BINDING
        self.state.config = payload.config or {}
        self.state.bindings = []
        self.state.expected_start_time = None
        self.state.stop_reason = None
        self._touch(payload.cmd)

    def _handle_binding_sync(self, payload: CommandPayload) -> None:
        self._ensure_same_session(payload)
        self._ensure_phase([NodePhase.BINDING])
        bindings = payload.config.get("bindings") if payload.config else None
        self.state.bindings = bindings or []
        self.state.phase = NodePhase.BINDING
        self._touch(payload.cmd)

    def _handle_start_monitor(self, payload: CommandPayload) -> None:
        self._ensure_same_session(payload)
        self._ensure_phase([NodePhase.BINDING])
        self.state.phase = NodePhase.MONITORING
        self.state.expected_start_time = (payload.config or {}).get("expected_start_time")
        # simulate activation result
        self.state.config["tracking_active"] = (payload.config or {}).get("tracking_active", True)
        if not self.capture._running.is_set():
            self.capture.start()
            self.capture.start_display()
        self.state.capture_running = True
        self._touch(payload.cmd)

    def _handle_stop(self, payload: CommandPayload) -> None:
        self._ensure_same_session(payload)
        self.state.phase = NodePhase.STOPPED
        self.state.stop_reason = (payload.config or {}).get("reason")
        # simulate cleanup
        self.state.config["tracking_active"] = False
        self.capture.stop()
        self.state.capture_running = False
        self._touch(payload.cmd)

    def _handle_heartbeat(self, payload: CommandPayload) -> None:
        self._ensure_same_session(payload, allow_empty=True)
        self._touch(payload.cmd)

    # --- helpers ---
    def _ensure_same_session(self, payload: CommandPayload, allow_empty: bool = False) -> None:
        if allow_empty and not self.state.session_id:
            return
        if self.state.session_id and self.state.session_id != payload.session_id:
            raise HTTPException(status_code=409, detail="Session mismatch on node")

    def _ensure_phase(self, allowed: list[NodePhase]) -> None:
        if self.state.phase not in allowed:
            raise HTTPException(
                status_code=409,
                detail=f"Invalid phase {self.state.phase} for command; allowed: {[p.value for p in allowed]}",
            )

    def _touch(self, cmd: str) -> None:
        self.state.last_command = cmd
        self.state.last_updated_ms = int(time.time() * 1000)

    def snapshot(self) -> dict:
        return self.state.model_dump()

    def _on_frame(self, frame, ts_ms: float) -> None:
        # Update capture stats; algorithm placeholder can be inserted here
        prev_ts = self.state.last_frame_ts
        if prev_ts:
            delta = ts_ms - prev_ts
            if delta > 0:
                self.state.capture_fps_est = round(1000.0 / delta, 2)
        self.state.last_frame_ts = int(ts_ms)

    # --- persistence ---
    def _persist_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("w", encoding="utf-8") as f:
            json.dump(self.state.model_dump(), f, ensure_ascii=False, indent=2)

    def _load_state(self) -> NodeState:
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return NodeState(**data)
        except FileNotFoundError:
            return NodeState(node_id=self.settings.node_id)
        except Exception as exc:  # corrupted state
            logging.getLogger("edge.command").warning("Failed to load state.json: %s", exc)
            return NodeState(node_id=self.settings.node_id)
