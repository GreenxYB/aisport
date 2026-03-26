from fastapi import APIRouter, Depends, HTTPException

from ..models.schemas import SessionCreate, Session, CommandPayload, StartMonitorRequest
from ..services.session_service import SessionService
from ..services.node_connection_manager import NodeConnectionManager, get_node_manager

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


@router.get("/{session_id}/readiness")
async def get_readiness(
    session_id: str,
    svc: SessionService = Depends(get_service),
    manager: NodeConnectionManager = Depends(get_node_manager),
):
    session = svc.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    online_nodes = {row["node_id"]: row for row in await manager.list_online()}
    required = svc.node_ids(session)
    nodes = []
    all_ready = True
    for node_id in required:
        row = online_nodes.get(node_id)
        last_status = row.get("last_status") if row else None
        status_data = (last_status or {}).get("data", {})
        ready = bool(status_data.get("binding_ready")) and bool(status_data.get("camera_ready"))
        online = bool(row and row.get("online"))
        all_ready = all_ready and online and ready
        nodes.append(
            {
                "node_id": node_id,
                "online": online,
                "ready": ready,
                "last_status": last_status,
            }
        )
    return {"session_id": session_id, "all_ready": all_ready, "nodes": nodes}


@router.post("/{session_id}/commands/start-monitor")
async def dispatch_start_monitor(
    session_id: str,
    payload: StartMonitorRequest,
    svc: SessionService = Depends(get_service),
    manager: NodeConnectionManager = Depends(get_node_manager),
):
    session = svc.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    online_nodes = {row["node_id"]: row for row in await manager.list_online()}
    required = svc.node_ids(session)
    missing_or_not_ready: list[int] = []
    for node_id in required:
        row = online_nodes.get(node_id)
        last_status = row.get("last_status") if row else None
        status_data = (last_status or {}).get("data", {})
        ready = bool(status_data.get("binding_ready")) and bool(status_data.get("camera_ready"))
        online = bool(row and row.get("online"))
        if not (online and ready):
            missing_or_not_ready.append(node_id)

    if missing_or_not_ready:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Some nodes are offline or not ready",
                "node_ids": missing_or_not_ready,
            },
        )

    queued = []
    for node_id in required:
        command = svc.build_start_command(
            session=session,
            node_id=node_id,
            expected_start_time=payload.expected_start_time,
            countdown_seconds=payload.countdown_seconds,
            audio_plan=payload.audio_plan,
            tracking_active=payload.tracking_active,
        )
        delivered = await manager.send_command(node_id, command)
        if not delivered:
            raise HTTPException(status_code=409, detail=f"Node {node_id} went offline during dispatch")
        queued.append({"node_id": node_id, "cmd": command.cmd})

    return {"session_id": session_id, "queued": queued}
