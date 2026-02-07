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
