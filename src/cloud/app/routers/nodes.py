import logging

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from ..models.schemas import FinishReport, NodeStatusReport, ViolationReport
from common.protocol import CommandPayload, NodeConnectPayload
from ..services.node_connection_manager import NodeConnectionManager, get_node_manager

router = APIRouter()
logger = logging.getLogger("cloud.nodes")


@router.websocket("/ws")
async def node_ws(websocket: WebSocket, manager: NodeConnectionManager = Depends(get_node_manager)):
    await websocket.accept()
    node_id: int | None = None
    try:
        hello = await websocket.receive_json()
        payload = NodeConnectPayload(**hello)
        node_id = payload.node_id
        await manager.register(websocket, payload)
        await websocket.send_json({"type": "CONNECTED", "node_id": payload.node_id})

        while True:
            message = await websocket.receive_json()
            msg_type = message.get("msg_type")
            try:
                if msg_type == "NODE_STATUS":
                    await manager.update_status(NodeStatusReport(**message))
                elif msg_type == "COMMAND_ACK":
                    await manager.record_ack(message)
                elif msg_type == "ID_REPORT":
                    await manager.record_id_report(message)
                elif msg_type == "VIOLATION_EVENT":
                    await manager.record_violation(ViolationReport(**message))
                elif msg_type == "FINISH_REPORT":
                    await manager.record_finish(FinishReport(**message))
                else:
                    logger.debug("ignore websocket message without supported msg_type: %s", msg_type)
            except Exception as exc:
                logger.warning(
                    "failed to process node websocket message node_id=%s msg_type=%s error=%s",
                    node_id,
                    msg_type,
                    exc,
                )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("node websocket crashed node_id=%s error=%s", node_id, exc)
    finally:
        if node_id is not None:
            await manager.unregister(node_id)


@router.get("/online")
async def list_online_nodes(manager: NodeConnectionManager = Depends(get_node_manager)):
    return {"nodes": await manager.list_online()}


@router.post("/{node_id}/dispatch")
async def dispatch_command(
    node_id: int,
    payload: CommandPayload,
    manager: NodeConnectionManager = Depends(get_node_manager),
):
    delivered = await manager.send_command(node_id, payload)
    if not delivered:
        raise HTTPException(status_code=409, detail=f"Node {node_id} is offline")
    return {"status": "queued", "node_id": node_id, "cmd": payload.cmd, "session_id": payload.session_id}


@router.post("/reports/violation")
async def receive_violation(
    report: ViolationReport, manager: NodeConnectionManager = Depends(get_node_manager)
):
    await manager.record_violation(report)
    # TODO: persist and fan-out alerts
    return {"status": "accepted", "session_id": report.session_id, "count": len(report.data)}


@router.post("/reports/finish")
async def receive_finish(report: FinishReport, manager: NodeConnectionManager = Depends(get_node_manager)):
    await manager.record_finish(report)
    # TODO: persist results and trigger scoring pipeline
    return {"status": "accepted", "session_id": report.session_id, "count": len(report.data)}


@router.post("/reports/status")
async def receive_status(
    report: NodeStatusReport, manager: NodeConnectionManager = Depends(get_node_manager)
):
    await manager.update_status(report)
    # TODO: persist node status for readiness and health checks
    return {"status": "accepted", "session_id": report.session_id, "data": report.data}
