from fastapi import APIRouter, Depends, HTTPException

from ..models.schemas import SessionCreate, Session, CommandPayload, StartMonitorRequest
from ..services.session_service import SessionService
from ..services.node_connection_manager import NodeConnectionManager, get_node_manager
from ..services.orchestrator import SessionOrchestrator, get_orchestrator

router = APIRouter()


def get_service() -> SessionService:
    # Simple singleton for now; replace with dependency injection as needed
    if not hasattr(get_service, "_svc"):
        get_service._svc = SessionService()
    return get_service._svc  # type: ignore[attr-defined]


def get_orchestrator_service(
    svc: SessionService = Depends(get_service),
    manager: NodeConnectionManager = Depends(get_node_manager),
) -> SessionOrchestrator:
    return get_orchestrator(svc, manager)


@router.post("/", response_model=Session)
async def create_session(
    payload: SessionCreate,
    svc: SessionService = Depends(get_service),
    orchestrator: SessionOrchestrator = Depends(get_orchestrator_service),
):
    session = svc.create(payload)
    await orchestrator.register_session(session.session_id)
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
    return svc.build_init_command(session, payload.start_node_id)


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
        session_match = (last_status or {}).get("session_id") == session.session_id
        node_role = row.get("node_role") if row else None
        ready = session_match and svc.is_node_ready(session, status_data, node_role)
        online = bool(row and row.get("online"))
        all_ready = all_ready and online and ready
        nodes.append(
            {
                "node_id": node_id,
                "node_role": node_role,
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
        session_match = (last_status or {}).get("session_id") == session.session_id
        node_role = row.get("node_role") if row else None
        ready = session_match and svc.is_node_ready(session, status_data, node_role)
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

    svc.set_expected_start_time(session_id, payload.expected_start_time)
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


@router.get("/{session_id}/results")
async def get_session_results(
    session_id: str,
    svc: SessionService = Depends(get_service),
    manager: NodeConnectionManager = Depends(get_node_manager),
):
    session = svc.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    reports = await manager.get_session_reports(session_id)
    bindings_by_lane = {
        int(item["lane"]): item
        for item in session.bindings
        if isinstance(item, dict) and isinstance(item.get("lane"), int)
    }

    false_start_by_lane: dict[int, dict] = {}
    for report in reports["violations"]:
        for item in report.get("data") or []:
            if item.get("event") != "FALSE_START":
                continue
            lane = item.get("lane")
            if isinstance(lane, int) and lane not in false_start_by_lane:
                false_start_by_lane[lane] = item

    latest_finish_by_lane: dict[int, dict] = {}
    for report in reports["finishes"]:
        for item in report.get("data") or []:
            lane = item.get("lane")
            if not isinstance(lane, int):
                continue
            current = latest_finish_by_lane.get(lane)
            if current is None or int(item.get("finish_ts", 0)) >= int(current.get("finish_ts", 0)):
                latest_finish_by_lane[lane] = item

    lanes = sorted(set(bindings_by_lane) | set(false_start_by_lane) | set(latest_finish_by_lane))
    results = []
    for lane in lanes:
        binding = bindings_by_lane.get(lane, {})
        finish = latest_finish_by_lane.get(lane)
        false_start = false_start_by_lane.get(lane)
        finish_ts = int(finish["finish_ts"]) if finish and finish.get("finish_ts") is not None else None
        elapsed_ms = None
        if finish_ts is not None and session.expected_start_time is not None:
            elapsed_ms = finish_ts - int(session.expected_start_time)
        results.append(
            {
                "lane": lane,
                "student_id": binding.get("student_id"),
                "feature_id": binding.get("feature_id"),
                "finish_ts": finish_ts,
                "expected_start_time": session.expected_start_time,
                "elapsed_ms": elapsed_ms,
                "rank": finish.get("rank") if finish else None,
                "false_start": bool(false_start),
                "false_start_detail": false_start,
            }
        )

    return {
        "session_id": session_id,
        "status": session.status,
        "expected_start_time": session.expected_start_time,
        "results": results,
        "report_counts": {
            "id_reports": len(reports["id_reports"]),
            "violations": len(reports["violations"]),
            "finishes": len(reports["finishes"]),
        },
    }


@router.get("/{session_id}/diagnostics")
async def get_session_diagnostics(
    session_id: str,
    svc: SessionService = Depends(get_service),
    manager: NodeConnectionManager = Depends(get_node_manager),
    orchestrator: SessionOrchestrator = Depends(get_orchestrator_service),
):
    session = svc.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    readiness = await get_readiness(session_id=session_id, svc=svc, manager=manager)
    results = await get_session_results(session_id=session_id, svc=svc, manager=manager)
    workflow = await orchestrator.get_workflow_snapshot(session_id)

    online_nodes = {row["node_id"]: row for row in await manager.list_online()}
    required_nodes = []
    for node_id in svc.node_ids(session):
        row = online_nodes.get(node_id, {})
        required_nodes.append(
            {
                "node_id": node_id,
                "node_role": row.get("node_role"),
                "online": bool(row.get("online")),
                "last_status": row.get("last_status"),
                "last_ack": row.get("last_ack"),
                "last_id_report": row.get("last_id_report"),
                "last_violation": row.get("last_violation"),
                "last_finish": row.get("last_finish"),
            }
        )

    return {
        "session": session.model_dump(),
        "workflow": workflow,
        "readiness": readiness,
        "results": results,
        "nodes": required_nodes,
    }
