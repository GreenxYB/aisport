from datetime import datetime
from typing import Dict

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
            task_mode="IDENTITY_BINDING",
            config={
                "project_type": payload.project_type,
                "lane_count": payload.lane_count,
                "sync_time": payload.sync_time_ms,
            },
        )

    @staticmethod
    def _generate_session_id() -> str:
        # Format: RUN_YYYYMMDD_HHMMSS_mmm
        now = datetime.utcnow()
        return now.strftime("RUN_%Y%m%d_%H%M%S_%f")[:-3]
