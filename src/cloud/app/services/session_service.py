from datetime import datetime
from typing import Dict, List

from ..models.schemas import CommandPayload, Session, SessionCreate


PROJECT_NODE_RULES: dict[str, dict[int, int]] = {
    "50m": {1: 2, 4: 5},
    "100m": {1: 3, 4: 6},
    "200m": {7: 6, 8: 3},
    "400m": {7: 3, 8: 6},
    "800m": {7: 3, 8: 6},
    "1000m": {1: 6, 4: 3},
}


class SessionService:
    """In-memory session store and cloud-side command builder."""

    def __init__(self):
        self._sessions: Dict[str, Session] = {}

    def create(self, payload: SessionCreate) -> Session:
        project_type = self.normalize_project_type(payload.project_type)
        finish_node_id = self.derive_finish_node_id(
            project_type=project_type, start_node_id=payload.start_node_id
        )
        tracking_node_ids = self.derive_tracking_node_ids(
            project_type=project_type,
            start_node_id=payload.start_node_id,
            finish_node_id=finish_node_id,
        )
        candidate_lanes = self.derive_candidate_lanes(
            project_type=project_type,
            start_node_id=payload.start_node_id,
        )

        session_id = self._generate_session_id()
        session = Session(
            session_id=session_id,
            status="CREATED",
            created_at=datetime.utcnow(),
            project_type=project_type,
            lane_count=max(candidate_lanes) if candidate_lanes else 8,
            start_node_id=payload.start_node_id,
            finish_node_id=finish_node_id,
            tracking_node_ids=tracking_node_ids,
            bindings=[],
            candidate_lanes=candidate_lanes,
            active_lanes=[],
            binding_mode="DISCOVER",
            sync_time_ms=None,
            require_bindings=True,
            auto_start=payload.auto_start,
            binding_timeout_sec=payload.binding_timeout_sec,
            start_delay_ms=payload.start_delay_ms,
            countdown_seconds=payload.countdown_seconds,
            race_timeout_sec=payload.race_timeout_sec,
            audio_plan=payload.audio_plan,
            tracking_active=payload.tracking_active,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list(self) -> list[Session]:
        return list(self._sessions.values())

    def build_init_command(self, session: Session, node_id: int) -> CommandPayload:
        return CommandPayload(
            cmd="CMD_INIT",
            session_id=session.session_id,
            node_id=node_id,
            task_mode="TRACK_RACE",
            config={
                "project_type": session.project_type,
                "lane_count": session.lane_count,
                "candidate_lanes": session.candidate_lanes,
                "binding_mode": session.binding_mode,
                "sync_time": session.sync_time_ms,
                "binding_timeout_sec": session.binding_timeout_sec,
                "race_timeout_sec": session.race_timeout_sec,
            },
        )

    def build_binding_command(self, session: Session, node_id: int) -> CommandPayload:
        return CommandPayload(
            cmd="CMD_BINDING_SYNC",
            session_id=session.session_id,
            node_id=node_id,
            task_mode="TRACK_RACE",
            config={
                "bindings": session.bindings,
                "candidate_lanes": session.candidate_lanes,
                "binding_mode": session.binding_mode,
            },
        )

    def update_status(self, session_id: str, status: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.status = status
        return session

    def set_expected_start_time(
        self, session_id: str, expected_start_time: int
    ) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.expected_start_time = expected_start_time
        return session

    def set_active_lanes(
        self, session_id: str, active_lanes: List[int]
    ) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.active_lanes = sorted(
            dict.fromkeys(int(lane) for lane in active_lanes if isinstance(lane, int))
        )
        return session

    def finish(
        self,
        session_id: str,
        status: str,
        reason: str | None = None,
        finished_at_ms: int | None = None,
    ) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.status = status
        session.terminal_reason = reason
        session.finished_at_ms = finished_at_ms
        return session

    @staticmethod
    def node_ids(session: Session) -> List[int]:
        ids = [session.start_node_id, session.finish_node_id, *session.tracking_node_ids]
        seen: set[int] = set()
        ordered: List[int] = []
        for node_id in ids:
            if node_id not in seen:
                seen.add(node_id)
                ordered.append(node_id)
        return ordered

    @staticmethod
    def build_start_command(
        session: Session,
        node_id: int,
        expected_start_time: int,
        countdown_seconds: int,
        audio_plan: str,
        tracking_active: bool = True,
    ) -> CommandPayload:
        return CommandPayload(
            cmd="CMD_START_MONITOR",
            session_id=session.session_id,
            node_id=node_id,
            task_mode="TRACK_RACE",
            config={
                "expected_start_time": expected_start_time,
                "countdown_seconds": countdown_seconds,
                "audio_plan": audio_plan,
                "tracking_active": tracking_active,
                "race_timeout_sec": session.race_timeout_sec,
            },
        )

    @staticmethod
    def build_stop_command(session: Session, node_id: int, reason: str) -> CommandPayload:
        return CommandPayload(
            cmd="CMD_STOP",
            session_id=session.session_id,
            node_id=node_id,
            task_mode="TRACK_RACE",
            config={"reason": reason},
        )

    @staticmethod
    def target_lanes(session: Session) -> List[int]:
        if session.active_lanes:
            return sorted(dict.fromkeys(int(lane) for lane in session.active_lanes))
        lanes = [
            int(item.get("lane"))
            for item in session.bindings
            if isinstance(item, dict) and isinstance(item.get("lane"), int)
        ]
        if lanes:
            return sorted(dict.fromkeys(lanes))
        return []

    @staticmethod
    def is_node_ready(
        session: Session, status_data: dict | None, node_role: str | None = None
    ) -> bool:
        if not status_data:
            return False
        camera_ready = bool(status_data.get("camera_ready"))
        role = (node_role or status_data.get("node_role") or "").upper()
        binding_required = session.require_bindings and role in {"START", "ALL_IN_ONE"}
        if binding_required:
            return camera_ready and bool(status_data.get("binding_ready"))
        return camera_ready

    @staticmethod
    def normalize_project_type(project_type: str) -> str:
        normalized = str(project_type or "").strip().lower()
        if normalized not in PROJECT_NODE_RULES:
            supported = ", ".join(PROJECT_NODE_RULES.keys())
            raise ValueError(
                f"unsupported project_type={project_type}; supported: {supported}"
            )
        return normalized

    @staticmethod
    def derive_finish_node_id(project_type: str, start_node_id: int) -> int:
        rule = PROJECT_NODE_RULES.get(project_type, {})
        if start_node_id not in rule:
            allowed = sorted(rule.keys())
            raise ValueError(
                f"invalid start_node_id={start_node_id} for project_type={project_type}; allowed: {allowed}"
            )
        return int(rule[start_node_id])

    @staticmethod
    def derive_tracking_node_ids(
        project_type: str, start_node_id: int, finish_node_id: int
    ) -> List[int]:
        _ = (project_type, start_node_id, finish_node_id)
        return []

    @staticmethod
    def derive_candidate_lanes(project_type: str, start_node_id: int) -> List[int]:
        _ = (project_type, start_node_id)
        return list(range(1, 9))

    @staticmethod
    def _generate_session_id() -> str:
        now = datetime.utcnow()
        return now.strftime("RUN_%Y%m%d_%H%M%S_%f")[:-3]
