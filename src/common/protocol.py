from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CommandPayload(BaseModel):
    cmd: str
    session_id: str
    node_id: int
    task_mode: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class NodeConnectPayload(BaseModel):
    node_id: int
    node_role: Literal["START", "FINISH", "MID", "ALL_IN_ONE"]
    site_id: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    token: Optional[str] = None


class CommandAckMessage(BaseModel):
    msg_type: Literal["COMMAND_ACK"] = "COMMAND_ACK"
    node_id: int
    session_id: str
    timestamp: int
    cmd: str
    status: Literal["accepted", "error"]
    phase: Optional[str] = None
    error: Optional[str] = None


class IdReportItem(BaseModel):
    lane: Optional[int] = None
    student_id: Optional[str] = None
    confidence: Optional[float] = None
    face_token: Optional[str] = None
    name: Optional[str] = None
    status: Optional[str] = None


class IdReport(BaseModel):
    msg_type: Literal["ID_REPORT"] = "ID_REPORT"
    node_id: int
    session_id: str
    timestamp: int
    data: List[IdReportItem] = Field(default_factory=list)


class ViolationEventItem(BaseModel):
    event: str
    lane: Optional[int] = None
    track_id: Optional[int] = None
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    score: Optional[float] = None
    bbox: Optional[List[float]] = None
    keypoints: Optional[Any] = None
    toe_proxy_points: Optional[List[Dict[str, Any]]] = None
    evidence_frame: Optional[str] = None
    start_line_y: Optional[int] = None
    finish_line_y: Optional[int] = None


class ViolationReport(BaseModel):
    msg_type: Literal["VIOLATION_EVENT"] = "VIOLATION_EVENT"
    node_id: int
    session_id: str
    timestamp: int
    data: List[ViolationEventItem] = Field(default_factory=list)


class FinishReportItem(BaseModel):
    lane: int
    finish_ts: int
    track_id: Optional[int] = None
    rank: Optional[int] = None


class FinishReport(BaseModel):
    msg_type: Literal["FINISH_REPORT"] = "FINISH_REPORT"
    node_id: int
    session_id: str
    timestamp: int
    data: List[FinishReportItem] = Field(default_factory=list)


class NodeStatusReport(BaseModel):
    msg_type: Literal["NODE_STATUS"] = "NODE_STATUS"
    node_id: int
    session_id: str
    timestamp: int
    data: Dict[str, Any] = Field(default_factory=dict)
