from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NodePhase(str, Enum):
    IDLE = "IDLE"
    BINDING = "BINDING"
    MONITORING = "MONITORING"
    STOPPED = "STOPPED"


class NodeState(BaseModel):
    node_id: int
    session_id: Optional[str] = None
    phase: NodePhase = NodePhase.IDLE
    last_command: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    bindings: List[Dict[str, Any]] = Field(default_factory=list)
    expected_start_time: Optional[int] = None
    stop_reason: Optional[str] = None
    last_updated_ms: Optional[int] = None
    capture_running: bool = False
    capture_fps_est: Optional[float] = None
    last_frame_ts: Optional[int] = None
    capture_error: Optional[str] = None
    events_generated: int = 0
    last_event_ts: Optional[int] = None
    finish_reports_generated: int = 0
    last_finish_ts: Optional[int] = None
    reports_sent: int = 0
    reports_failed: int = 0
    algo_events_generated: int = 0
    last_algo_ts: Optional[int] = None
    last_face_result: Optional[Dict[str, Any]] = None
    last_face_ts: Optional[int] = None
    last_false_start_event: Optional[Dict[str, Any]] = None
    last_false_start_ts: Optional[int] = None
    last_toe_proxy_debug: Optional[Dict[str, Any]] = None
    last_toe_proxy_ts: Optional[int] = None
