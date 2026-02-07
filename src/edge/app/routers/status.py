from fastapi import APIRouter, Depends

from ..services.command_handler import CommandHandler
from ..routers.commands import get_handler

router = APIRouter()


@router.get("/")
def current_status(handler: CommandHandler = Depends(get_handler)):
    return handler.snapshot()
