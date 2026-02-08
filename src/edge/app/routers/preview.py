from fastapi import APIRouter, Depends, HTTPException, Response

from ..services.command_handler import CommandHandler
from .commands import get_handler

router = APIRouter()


@router.get("/snapshot", responses={200: {"content": {"image/jpeg": {}}}})
def snapshot(handler: CommandHandler = Depends(get_handler)):
    jpeg = handler.capture.snapshot_jpeg()
    if not jpeg:
        err = handler.capture.last_encode_error()
        detail = "No frame available yet"
        if err:
            detail = f"No frame available yet: {err}"
        raise HTTPException(status_code=503, detail=detail)
    return Response(content=jpeg, media_type="image/jpeg")
