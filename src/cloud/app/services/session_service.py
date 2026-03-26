from datetime import datetime
from typing import Dict, List

from ..models.schemas import SessionCreate, Session, CommandPayload


class SessionService:
    def __init__(self):
        # TODO: replace in-memory store with real DB
        self._sessions: Dict[str, Session] = {}

    def create(self, payload: SessionCreate) -> Session:
        session_id = self._generate_session_id()
        session = Session(
            session_id=session_id,
            status="INIT",
            created_at=datetime.utcnow(),
            project_type=payload.project_type,
            lane_count=payload.lane_count,
            start_node_id=payload.start_node_id,
            finish_node_id=payload.finish_node_id,
            tracking_node_ids=payload.tracking_node_ids,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list(self) -> list[Session]:
        return list(self._sessions.values())

    def build_init_command(self, session: Session, payload: SessionCreate) -> CommandPayload:
        return CommandPayload(
            cmd="CMD_INIT",
            session_id=session.session_id,
            node_id=payload.start_node_id,
            task_mode="TRACK_RACE",
            config={
                "project_type": payload.project_type,
                "lane_count": payload.lane_count,
                "sync_time": payload.sync_time_ms,
            },
        )

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
    def build_start_command(session: Session, node_id: int, expected_start_time: int, countdown_seconds: int, audio_plan: str, tracking_active: bool = True) -> CommandPayload:
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
            },
        )

    @staticmethod
    def _generate_session_id() -> str:
        # Format: RUN_YYYYMMDD_HHMMSS_mmm
        now = datetime.utcnow()
        return now.strftime("RUN_%Y%m%d_%H%M%S_%f")[:-3]
