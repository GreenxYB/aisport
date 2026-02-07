from fastapi import APIRouter, Depends, HTTPException

from ..models.schemas import SessionCreate, Session, CommandPayload
from ..services.session_service import SessionService

router = APIRouter()


def get_service() -> SessionService:
    # Simple singleton for now; replace with dependency injection as needed
    if not hasattr(get_service, "_svc"):
        get_service._svc = SessionService()
    return get_service._svc  # type: ignore[attr-defined]


@router.post("/", response_model=Session)
def create_session(payload: SessionCreate, svc: SessionService = Depends(get_service)):
    session = svc.create(payload)
    return session


@router.get("/", response_model=list[Session])
def list_sessions(svc: SessionService = Depends(get_service)):
    return svc.list()


@router.get("/{session_id}", response_model=Session)
def get_session(session_id: str, svc: SessionService = Depends(get_service)):
    session = svc.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/{session_id}/commands/init", response_model=CommandPayload)
def build_init_command(
    session_id: str, payload: SessionCreate, svc: SessionService = Depends(get_service)
):
    session = svc.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return svc.build_init_command(session, payload)
