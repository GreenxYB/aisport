from fastapi import APIRouter, Depends, HTTPException, Response

from ..services.command_handler import CommandHandler
from .commands import get_handler

router = APIRouter()


@router.get("/snapshot", responses={200: {"content": {"image/jpeg": {}}}})
def snapshot(handler: CommandHandler = Depends(get_handler)):
    jpeg = handler.capture.snapshot_jpeg()
    if not jpeg:
        raise HTTPException(status_code=503, detail="No frame available yet")
    return Response(content=jpeg, media_type="image/jpeg")
