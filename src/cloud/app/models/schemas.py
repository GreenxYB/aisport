from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from common.protocol import CommandPayload, ViolationReport, FinishReport  # type: ignore


class SessionCreate(BaseModel):
    project_type: str = Field(..., example="200m")
    lane_count: int = Field(..., ge=1, le=10)
    start_node_id: int = Field(..., example=1)
    finish_node_id: int = Field(..., example=7)
    tracking_node_ids: List[int] = Field(default_factory=list, example=[3, 6])
    sync_time_ms: Optional[int] = Field(
        None, example=1738416000000, description="Absolute timestamp for time sync"
    )


class Session(BaseModel):
    session_id: str
    status: str
    created_at: datetime
    project_type: str
    lane_count: int
