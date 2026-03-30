from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from common.protocol import (  # type: ignore
    CommandPayload,
    FinishReport,
    NodeStatusReport,
    ViolationReport,
)


class SessionCreate(BaseModel):
    project_type: str = Field(..., example="200m")
    start_node_id: int = Field(..., example=1)
    auto_start: bool = True
    binding_timeout_sec: int = Field(10, ge=1, le=600)
    start_delay_ms: int = Field(5000, ge=1000)
    countdown_seconds: int = Field(3, ge=0, le=10)
    race_timeout_sec: int = Field(60, ge=1, le=3600)
    audio_plan: str = "START_321_GO"
    tracking_active: bool = True


class Session(BaseModel):
    session_id: str
    status: str
    created_at: datetime
    project_type: str
    lane_count: int
    start_node_id: int
    finish_node_id: int
    tracking_node_ids: List[int] = Field(default_factory=list)
    bindings: List[Dict[str, Any]] = Field(default_factory=list)
    candidate_lanes: List[int] = Field(default_factory=list)
    active_lanes: List[int] = Field(default_factory=list)
    binding_mode: str = "DISCOVER"
    sync_time_ms: Optional[int] = None
    require_bindings: bool = False
    auto_start: bool = True
    binding_timeout_sec: int = 10
    start_delay_ms: int = 5000
    countdown_seconds: int = 3
    race_timeout_sec: int = 60
    audio_plan: str = "START_321_GO"
    tracking_active: bool = True
    expected_start_time: Optional[int] = None
    finished_at_ms: Optional[int] = None
    terminal_reason: Optional[str] = None


class StartMonitorRequest(BaseModel):
    expected_start_time: int
    countdown_seconds: int = 3
    tracking_active: bool = True
    audio_plan: str = "START_321_GO"
