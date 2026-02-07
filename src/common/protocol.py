from typing import List, Optional

from pydantic import BaseModel, Field


class CommandPayload(BaseModel):
    cmd: str
    session_id: str
    node_id: int
    task_mode: Optional[str] = None
    config: dict = Field(default_factory=dict)


class ViolationReport(BaseModel):
    msg_type: str = "VIOLATION_EVENT"
    node_id: int
    session_id: str
    event: str
    lane: Optional[int]
    timestamp: int
    evidence_frame: Optional[str]


class FinishReportItem(BaseModel):
    lane: int
    finish_ts: int


class FinishReport(BaseModel):
    msg_type: str = "FINISH_REPORT"
    node_id: int
    session_id: str
    results: List[FinishReportItem]
