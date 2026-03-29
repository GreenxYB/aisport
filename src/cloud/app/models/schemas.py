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
    lane_count: int = Field(..., ge=1, le=10)
    start_node_id: int = Field(..., example=1)
    finish_node_id: int = Field(..., example=7)
    tracking_node_ids: List[int] = Field(default_factory=list, example=[3, 6])
    bindings: List[Dict[str, Any]] = Field(default_factory=list)
    auto_start: bool = True
    start_delay_ms: int = Field(5000, ge=1000)
    countdown_seconds: int = Field(3, ge=0, le=10)
    audio_plan: str = "START_321_GO"
    tracking_active: bool = True
    sync_time_ms: Optional[int] = Field(
        None, example=1738416000000, description="Absolute timestamp for time sync"
    )


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
    sync_time_ms: Optional[int] = None
    require_bindings: bool = False
    auto_start: bool = True
    start_delay_ms: int = 5000
    countdown_seconds: int = 3
    audio_plan: str = "START_321_GO"
    tracking_active: bool = True


class StartMonitorRequest(BaseModel):
    expected_start_time: int
    countdown_seconds: int = 3
    tracking_active: bool = True
    audio_plan: str = "START_321_GO"
