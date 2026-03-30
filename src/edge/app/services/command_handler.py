import json
import logging
import time
from pathlib import Path
from typing import Callable

from fastapi import HTTPException

from common.protocol import CommandPayload, NodeStatusReport
from ..core.config import get_settings
from ..core.state import NodePhase, NodeState
from .algorithms.lane_layout import binding_target_lanes, inspect_lane_layout
from .algorithms.race_line import inspect_line_definition
from .event_simulator import EventSimulator
from .algorithms import AlgorithmRunner
from .publisher import NullPublisher

try:
    from .pipeline import EdgePipeline
except Exception as exc:  # pragma: no cover
    EdgePipeline = None
    PIPELINE_IMPORT_ERROR = exc
else:
    PIPELINE_IMPORT_ERROR = None


class _NoopPipeline:
    def __init__(self):
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def snapshot_jpeg(self) -> bytes | None:
        return None

    def last_encode_error(self) -> str | None:
        return None


class CommandHandler:
    """
    Handle Cloud->Edge commands and maintain edge node runtime state.

    Main transitions:
    - CMD_INIT: reset round/session runtime
    - CMD_BINDING_SYNC: sync lane/student bindings
    - CMD_START_MONITOR: enter monitoring and start capture
    - CMD_STOP: stop current round
    - CMD_RESET_ROUND: reset to binding stage for rerun
    """

    def __init__(self):
        """Initialize command processor and runtime dependencies."""
        self.settings = get_settings()
        # Persisted runtime state snapshot for quick restart recovery.
        self.state_file = Path(__file__).resolve().parents[4] / "logs" / "state.json"
        self.state = self._load_state()
        self.logger = logging.getLogger("edge.command")
        self.publisher = NullPublisher()

        # Event generator for simulation/debug mode.
        self.event_sim = EventSimulator(self.state, publisher=self.publisher)
        # Domain logic runner (face binding / false start / finish line).
        self.algo = AlgorithmRunner(self.state, publisher=self.publisher)

        # Capture -> inference -> tracker -> business logic pipeline.
        if EdgePipeline is None:
            self.pipeline = _NoopPipeline()
            self.logger.warning("EdgePipeline unavailable, falling back to no-op pipeline: %s", PIPELINE_IMPORT_ERROR)
        else:
            self.pipeline = EdgePipeline(algo_runner=self.algo)

        if self.settings.auto_start_capture:
            self.pipeline.start()
            self.state.capture_running = self.pipeline.running
            self.logger.info("capture auto-start finished running=%s", self.pipeline.running)

        self.allowed_cmds = {
            "CMD_INIT",
            "CMD_BINDING_SYNC",
            "CMD_START_MONITOR",
            "CMD_STOP",
            "CMD_RESET_ROUND",
            "CMD_HEARTBEAT",
        }
        self._dispatch_map: dict[str, Callable[[CommandPayload], None]] = {
            "CMD_INIT": self._handle_init,
            "CMD_BINDING_SYNC": self._handle_binding_sync,
            "CMD_START_MONITOR": self._handle_start_monitor,
            "CMD_STOP": self._handle_stop,
            "CMD_RESET_ROUND": self._handle_reset_round,
            "CMD_HEARTBEAT": self._handle_heartbeat,
        }

    def handle(self, payload: CommandPayload) -> None:
        """Dispatch one command and persist updated edge state."""
        started = time.time()
        if payload.cmd not in self.allowed_cmds:
            self.logger.warning("unsupported command cmd=%s session=%s node=%s", payload.cmd, payload.session_id, payload.node_id)
            raise HTTPException(status_code=400, detail="unsupported command")

        handler = self._dispatch_map[payload.cmd]
        self.logger.info(
            "cmd=%s session=%s node=%s summary=%s",
            payload.cmd,
            payload.session_id,
            payload.node_id,
            self._summarize_command(payload),
        )

        handler(payload)
        self._persist_state()

        elapsed_ms = int((time.time() - started) * 1000)
        self.logger.info(
            "handled cmd=%s phase=%s elapsed_ms=%s",
            payload.cmd,
            self.state.phase,
            elapsed_ms,
        )

    def _handle_init(self, payload: CommandPayload) -> None:
        """Reset runtime state and enter binding phase."""
        self.state.session_id = payload.session_id
        self.state.phase = NodePhase.BINDING
        self.state.config = payload.config or {}
        self.state.bindings = []
        self.state.expected_start_time = None
        self.state.stop_reason = None
        self.state.events_generated = 0
        self.state.last_event_ts = None
        self.state.finish_reports_generated = 0
        self.state.last_finish_ts = None
        self.state.binding_confirmed_students = []
        self.state.binding_confirmed_lanes = []
        self.state.binding_assignments = []
        self.state.binding_confirmed_at_ms = None
        self.state.last_face_result = None
        self.state.last_face_ts = None
        self.algo.reset_binding_runtime()
        self.event_sim.stop()
        self._touch(payload.cmd)
        self.logger.info(
            "state reset done session=%s lane_count=%s phase=%s",
            self.state.session_id,
            self.state.config.get("lane_count"),
            self.state.phase.value,
        )

    def _handle_binding_sync(self, payload: CommandPayload) -> None:
        """Sync lane bindings before monitoring starts."""
        self._ensure_same_session(payload)
        self._ensure_phase([NodePhase.BINDING])

        bindings = payload.config.get("bindings") if payload.config else None
        self.state.bindings = bindings or []
        self.state.phase = NodePhase.BINDING
        self.algo.reset_binding_runtime()

        if self.settings.simulate_events:
            lane_count = int(self.state.config.get("lane_count", 1) or 1)
            self.event_sim.start(self.state.session_id or "UNKNOWN", lane_count)
        self._touch(payload.cmd)
        self.logger.info("binding synced count=%s session=%s", len(self.state.bindings), self.state.session_id)

    def _handle_start_monitor(self, payload: CommandPayload) -> None:
        """Switch to MONITORING phase and ensure pipeline is running."""
        self._ensure_same_session(payload)
        self._ensure_phase([NodePhase.BINDING])

        self.state.phase = NodePhase.MONITORING
        self.state.expected_start_time = (payload.config or {}).get(
            "expected_start_time"
        )
        self.state.config["ready_ts"] = int(time.time() * 1000)
        self.state.config["tracking_active"] = (payload.config or {}).get(
            "tracking_active", True
        )
        self.state.config["countdown_seconds"] = (payload.config or {}).get(
            "countdown_seconds", 3
        )

        if not self.pipeline.running:
            try:
                self.pipeline.start()
            except Exception as exc:
                self.state.capture_error = str(exc)
                self.logger.error("pipeline start failed session=%s error=%s", self.state.session_id, exc)
                raise HTTPException(status_code=503, detail="camera open failed")

        self.state.capture_running = self.pipeline.running
        self.state.capture_error = None

        if self.settings.simulate_events:
            lane_count = int(self.state.config.get("lane_count", 1) or 1)
            self.event_sim.start(self.state.session_id or "UNKNOWN", lane_count)
        self._touch(payload.cmd)
        self.logger.info(
            "monitor started session=%s expected_start=%s tracking_active=%s countdown=%s capture_running=%s",
            self.state.session_id,
            self.state.expected_start_time,
            self.state.config.get("tracking_active"),
            self.state.config.get("countdown_seconds"),
            self.state.capture_running,
        )

    def _handle_stop(self, payload: CommandPayload) -> None:
        """Stop current round and optionally stop capture."""
        self._ensure_same_session(payload)
        self.state.phase = NodePhase.STOPPED
        self.state.stop_reason = (payload.config or {}).get("reason")
        reason = str(self.state.stop_reason or "")
        stop_capture = bool(
            (payload.config or {}).get(
                "stop_capture",
                False if reason == "BINDING_TIMEOUT" else True,
            )
        )
        if reason == "BINDING_TIMEOUT":
            try:
                status = self.build_status_report().data
                self.logger.warning(
                    "binding timeout diagnostics session=%s stage=%s ready=%s required=%s "
                    "configured_lanes=%s observed_lanes=%s confirmed_lanes=%s pending_lanes=%s "
                    "last_face_ts=%s last_lane_obs_ts=%s camera_ready=%s",
                    self.state.session_id,
                    status.get("session_stage"),
                    status.get("binding_ready"),
                    status.get("binding_required"),
                    status.get("binding_configured_lanes", status.get("binding_target_lanes")),
                    status.get("binding_observed_lanes"),
                    status.get("binding_confirmed_lanes"),
                    status.get("binding_pending_lanes"),
                    status.get("last_face_ts"),
                    status.get("last_lane_observation_ts"),
                    status.get("camera_ready"),
                )
            except Exception as exc:
                self.logger.warning("binding timeout diagnostics failed: %s", exc)

        self.state.config["tracking_active"] = False
        if stop_capture:
            self.pipeline.stop()
            self.state.capture_running = False
            self.state.capture_error = None
        else:
            self.logger.info(
                "stop cmd=%s reason=%s stop_capture=%s -> keep capture and preview running",
                payload.cmd,
                reason or "-",
                stop_capture,
            )
            self.state.capture_running = self.pipeline.running

        self.event_sim.stop()
        self._touch(payload.cmd)
        self.logger.info(
            "monitor stopped session=%s reason=%s stop_capture=%s capture_running=%s",
            self.state.session_id,
            reason or "-",
            stop_capture,
            self.state.capture_running,
        )

    def _handle_reset_round(self, payload: CommandPayload) -> None:
        """Reset one round while keeping session and binding context."""
        self._ensure_same_session(payload)
        self._ensure_phase([NodePhase.BINDING, NodePhase.MONITORING, NodePhase.STOPPED])

        self.state.phase = NodePhase.BINDING
        self.state.expected_start_time = None
        self.state.stop_reason = None
        self.state.config["tracking_active"] = False
        self.state.binding_confirmed_students = []
        self.state.binding_confirmed_lanes = []
        self.state.binding_assignments = []
        self.state.binding_confirmed_at_ms = None
        self.state.last_face_result = None
        self.state.last_face_ts = None
        self.state.last_false_start_event = None
        self.state.last_false_start_ts = None
        self.state.last_toe_proxy_debug = None
        self.state.last_toe_proxy_ts = None
        self.algo.reset_binding_runtime()

        self.event_sim.stop()
        self._touch(payload.cmd)
        self.logger.info("round reset session=%s phase=%s", self.state.session_id, self.state.phase.value)

    def _handle_heartbeat(self, payload: CommandPayload) -> None:
        """Keepalive command."""
        self._ensure_same_session(payload, allow_empty=True)
        self._touch(payload.cmd)

    def _ensure_same_session(
        self, payload: CommandPayload, allow_empty: bool = False
    ) -> None:
        """Validate session consistency before applying state transitions."""
        if allow_empty and not self.state.session_id:
            return
        if self.state.session_id and self.state.session_id != payload.session_id:
            self.logger.warning(
                "session mismatch current=%s incoming=%s cmd=%s",
                self.state.session_id,
                payload.session_id,
                payload.cmd,
            )
            raise HTTPException(status_code=409, detail="node session mismatch")

    def _ensure_phase(self, allowed: list[NodePhase]) -> None:
        """Validate command can be executed in current phase."""
        if self.state.phase not in allowed:
            allowed_values = [p.value for p in allowed]
            self.logger.warning("phase mismatch current=%s allowed=%s", self.state.phase.value, allowed_values)
            raise HTTPException(
                status_code=409,
                detail=f"invalid phase {self.state.phase}; allowed: {allowed_values}",
            )

    def _touch(self, cmd: str) -> None:
        """Update basic command metadata."""
        self.state.last_command = cmd
        self.state.last_updated_ms = int(time.time() * 1000)


    @staticmethod
    def _summarize_command(payload: CommandPayload) -> str:
        config = payload.config or {}
        if payload.cmd == "CMD_INIT":
            return (
                f"project_type={config.get('project_type')} "
                f"lane_count={config.get('lane_count')}"
            )
        if payload.cmd == "CMD_BINDING_SYNC":
            bindings = config.get("bindings") or []
            return f"bindings={len(bindings)}"
        if payload.cmd == "CMD_START_MONITOR":
            return (
                f"start_at={config.get('expected_start_time')} "
                f"countdown={config.get('countdown_seconds', 3)} "
                f"tracking={config.get('tracking_active', True)}"
            )
        if payload.cmd in {"CMD_STOP", "CMD_RESET_ROUND"}:
            return (
                f"reason={config.get('reason')} "
                f"stop_capture={config.get('stop_capture')}"
            )
        return "-"

    def build_status_report(self) -> NodeStatusReport:
        lane_count = int(self.state.config.get("lane_count", 0) or 0)
        configured_target_lanes = binding_target_lanes(self.state.bindings, lane_count)

        observed_lanes: list[int] = []
        lane_debug = self.state.lane_layout_debug
        if isinstance(lane_debug, dict):
            observations = lane_debug.get("observations") or []
            if isinstance(observations, list):
                for item in observations:
                    if isinstance(item, dict) and isinstance(item.get("lane"), int):
                        observed_lanes.append(int(item["lane"]))
        observed_lanes = list(dict.fromkeys(observed_lanes))

        # Prefer lanes actually observed in current frame; fallback to configured lanes.
        if configured_target_lanes and observed_lanes:
            observed_set = set(observed_lanes)
            target_lanes = [lane for lane in configured_target_lanes if lane in observed_set]
            if not target_lanes:
                target_lanes = configured_target_lanes
        else:
            target_lanes = configured_target_lanes

        binding_students_by_lane: dict[int, str] = {}
        for item in self.state.bindings:
            if not isinstance(item, dict):
                continue
            lane = item.get("lane")
            student_id = item.get("student_id")
            if isinstance(lane, int) and student_id:
                binding_students_by_lane[int(lane)] = str(student_id)

        binding_target_students = [
            binding_students_by_lane[lane]
            for lane in target_lanes
            if lane in binding_students_by_lane
        ]
        confirmed_students = list(dict.fromkeys(self.state.binding_confirmed_students))
        confirmed_lanes = list(dict.fromkeys(self.state.binding_confirmed_lanes))
        pending_students = [
            student_id for student_id in binding_target_students if student_id not in confirmed_students
        ]
        pending_lanes = [
            lane for lane in target_lanes if lane not in confirmed_lanes
        ]
        binding_required = bool(binding_target_students) and self.settings.node_role.upper() in {"START", "ALL_IN_ONE"}
        if self.settings.node_role.upper() in {"START", "ALL_IN_ONE"}:
            binding_required = bool(target_lanes)
        binding_ready = (not binding_required) or not pending_lanes
        now_ms = int(time.time() * 1000)
        session_stage = self.state.phase.value
        if self.state.phase == NodePhase.BINDING:
            session_stage = "WAIT_BINDING" if binding_required and not binding_ready else "BOUND"
        elif self.state.phase == NodePhase.MONITORING:
            if self.state.expected_start_time and now_ms < int(self.state.expected_start_time):
                session_stage = "COUNTDOWN"
            else:
                session_stage = "RUNNING"
        elif self.state.phase == NodePhase.STOPPED:
            session_stage = "STOPPED"

        lane_layout_status = inspect_lane_layout(
            frame_width=int(self.settings.capture_width),
            frame_height=int(self.settings.capture_height),
            target_lanes=target_lanes,
            lane_ranges_text=self.settings.lane_x_ranges,
            lane_polygons_text=self.settings.lane_polygons,
            lane_layout_file=self.settings.lane_layout_file,
        )
        start_line_status = inspect_line_definition(
            frame_width=int(self.settings.capture_width),
            frame_height=int(self.settings.capture_height),
            line_file=self.settings.start_line_file,
            fallback_y=int(self.settings.start_line_y),
            line_name="start_line",
        )
        finish_line_status = inspect_line_definition(
            frame_width=int(self.settings.capture_width),
            frame_height=int(self.settings.capture_height),
            line_file=self.settings.finish_line_file,
            fallback_y=int(self.settings.finish_line_y),
            line_name="finish_line",
        )

        return NodeStatusReport(
            node_id=self.state.node_id,
            session_id=self.state.session_id or "",
            timestamp=int(time.time() * 1000),
            data={
                "session_stage": session_stage,
                "phase": self.state.phase.value,
                "last_command": self.state.last_command,
                "capture_running": self.state.capture_running,
                "capture_fps_est": self.state.capture_fps_est,
                "last_frame_ts": self.state.last_frame_ts,
                "capture_error": self.state.capture_error,
                "binding_required": binding_required,
                "binding_ready": binding_ready,
                "binding_target_count": len(target_lanes),
                "binding_target_lanes": target_lanes,
                "binding_configured_lanes": configured_target_lanes,
                "binding_observed_lanes": observed_lanes,
                "binding_confirmed_count": len(confirmed_lanes),
                "binding_confirmed_lanes": confirmed_lanes,
                "binding_pending_count": len(pending_lanes),
                "binding_pending_lanes": pending_lanes,
                "binding_confirmed_students": confirmed_students,
                "binding_pending_students": pending_students,
                "binding_assignments": self.state.binding_assignments,
                "binding_confirmed_at_ms": self.state.binding_confirmed_at_ms,
                "last_face_ts": self.state.last_face_ts,
                "lane_layout_status": lane_layout_status,
                "start_line_status": start_line_status,
                "finish_line_status": finish_line_status,
                "lane_layout_debug": self.state.lane_layout_debug,
                "last_lane_observation_ts": self.state.last_lane_observation_ts,
                "camera_ready": self.state.capture_running and not self.state.capture_error,
                "tracking_active": bool(self.state.config.get("tracking_active", False)),
                "expected_start_time": self.state.expected_start_time,
                "ready_ts": self.state.config.get("ready_ts"),
                "countdown_seconds": self.state.config.get("countdown_seconds"),
                "binding_timeout_sec": self.state.config.get("binding_timeout_sec"),
                "race_timeout_sec": self.state.config.get("race_timeout_sec"),
                "last_false_start_ts": self.state.last_false_start_ts,
                "node_role": self.settings.node_role,
            },
        )

    def set_publisher(self, publisher) -> None:
        self.publisher = publisher
        self.event_sim.publisher = publisher
        self.algo.publisher = publisher

    def snapshot(self) -> dict:
        """Return full in-memory node state snapshot."""
        return self.state.model_dump()

    def _persist_state(self) -> None:
        """Persist state snapshot to disk for troubleshooting/recovery."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("w", encoding="utf-8") as f:
            json.dump(self.state.model_dump(), f, ensure_ascii=False, indent=2)

    def _load_state(self) -> NodeState:
        """Load previous state snapshot; fallback to empty state if unavailable."""
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return NodeState(**data)
        except FileNotFoundError:
            return NodeState(node_id=self.settings.node_id)
        except Exception as exc:  # corrupted or incompatible state file
            logging.getLogger("edge.command").warning("failed to load state.json, fallback to empty state: %s", exc)
            return NodeState(node_id=self.settings.node_id)
