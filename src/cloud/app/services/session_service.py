from datetime import datetime
from typing import Dict, List

from ..models.schemas import SessionCreate, Session, CommandPayload


class SessionService:
    """会话领域服务。

    负责会话生命周期的内存态管理，以及构造下发到 Edge 的标准命令。
    """

    def __init__(self):
        # TODO: replace in-memory store with real DB
        self._sessions: Dict[str, Session] = {}

    def create(self, payload: SessionCreate) -> Session:
        """创建会话并初始化调度参数。"""
        session_id = self._generate_session_id()
        session = Session(
            session_id=session_id,
            status="CREATED",
            created_at=datetime.utcnow(),
            project_type=payload.project_type,
            lane_count=payload.lane_count,
            start_node_id=payload.start_node_id,
            finish_node_id=payload.finish_node_id,
            tracking_node_ids=payload.tracking_node_ids,
            bindings=payload.bindings,
            sync_time_ms=payload.sync_time_ms,
            require_bindings=bool(payload.bindings),
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
            config={"bindings": session.bindings},
        )

    def update_status(self, session_id: str, status: str) -> Session | None:
        """更新会话状态（如 CREATED -> RUNNING）。"""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.status = status
        return session

    def set_expected_start_time(self, session_id: str, expected_start_time: int) -> Session | None:
        """写入统一起跑时间戳（毫秒）。"""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.expected_start_time = expected_start_time
        return session

    def finish(self, session_id: str, status: str, reason: str | None = None, finished_at_ms: int | None = None) -> Session | None:
        """结束会话并记录终止原因。"""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        session.status = status
        session.terminal_reason = reason
        session.finished_at_ms = finished_at_ms
        return session

    @staticmethod
    def node_ids(session: Session) -> List[int]:
        """返回去重后节点列表（保持原有顺序）。"""
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
        """构造开始监控命令（包含统一起跑时间）。"""
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
        """计算本场目标赛道：优先使用绑定赛道，兜底用 1..lane_count。"""
        lanes = [
            int(item.get("lane"))
            for item in session.bindings
            if isinstance(item, dict) and isinstance(item.get("lane"), int)
        ]
        if lanes:
            return sorted(dict.fromkeys(lanes))
        return list(range(1, int(session.lane_count) + 1))

    @staticmethod
    def is_node_ready(session: Session, status_data: dict | None, node_role: str | None = None) -> bool:
        """判断节点是否可开赛。

        START/ALL_IN_ONE 节点在要求绑定时，除 camera_ready 外还必须 binding_ready。
        """
        if not status_data:
            return False
        camera_ready = bool(status_data.get("camera_ready"))
        role = (node_role or status_data.get("node_role") or "").upper()
        binding_required = session.require_bindings and role in {"START", "ALL_IN_ONE"}
        if binding_required:
            return camera_ready and bool(status_data.get("binding_ready"))
        return camera_ready

    @staticmethod
    def _generate_session_id() -> str:
        # Format: RUN_YYYYMMDD_HHMMSS_mmm
        now = datetime.utcnow()
        return now.strftime("RUN_%Y%m%d_%H%M%S_%f")[:-3]
