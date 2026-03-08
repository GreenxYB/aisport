from fastapi import APIRouter, Depends

from ..services.command_handler import CommandHandler
from .commands import get_handler

router = APIRouter()


@router.get("/last")
def last_face(handler: CommandHandler = Depends(get_handler)):
    return {
        "last_face_ts": handler.state.last_face_ts,
        "last_face_result": handler.state.last_face_result,
    }
