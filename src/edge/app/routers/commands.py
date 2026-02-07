from fastapi import APIRouter, Depends

from common.protocol import CommandPayload
from ..models.schemas import CommandAck
from ..services.command_handler import CommandHandler

router = APIRouter()


def get_handler() -> CommandHandler:
    if not hasattr(get_handler, "_handler"):
        get_handler._handler = CommandHandler()
    return get_handler._handler  # type: ignore[attr-defined]


@router.post("/", response_model=CommandAck)
def receive_command(payload: CommandPayload, handler: CommandHandler = Depends(get_handler)):
    handler.handle(payload)
    return CommandAck(
        session_id=payload.session_id,
        node_id=payload.node_id,
        cmd=payload.cmd,
        phase=handler.state.phase.value,
        last_updated_ms=handler.state.last_updated_ms,
    )
