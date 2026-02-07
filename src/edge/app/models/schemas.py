from pydantic import BaseModel

from common.protocol import CommandPayload


class CommandAck(BaseModel):
    status: str = "accepted"
    session_id: str
    node_id: int
    cmd: str
    phase: str | None = None
    last_updated_ms: int | None = None
