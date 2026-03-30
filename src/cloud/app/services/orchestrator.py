import asyncio
import logging
import time
from dataclasses import dataclass, field

from .node_connection_manager import NodeConnectionManager
from .session_service import SessionService


@dataclass
class SessionWorkflow:
    """单场会话的下发进度快照。"""

    session_id: str
    init_sent_to: set[int] = field(default_factory=set)
    binding_sent_to: set[int] = field(default_factory=set)
    start_sent: bool = False
    stop_sent: bool = False
    init_sent_at_ms: int | None = None
    binding_sent_at_ms: int | None = None
    all_ready_at_ms: int | None = None
    start_sent_at_ms: int | None = None
    stop_sent_at_ms: int | None = None


class SessionOrchestrator:
    """会话自动编排器。

    按固定 tick 执行以下流程：
    1) 检查节点在线
    2) 依次下发 INIT / BINDING
    3) 检查 readiness 后下发 START
    4) 根据报告判断 FINISHED / TIMEOUT 并下发 STOP
    """

    def __init__(
        self, session_service: SessionService, node_manager: NodeConnectionManager
    ):
        self._session_service = session_service
        self._node_manager = node_manager
        self._lock = asyncio.Lock()
        self._workflows: dict[str, SessionWorkflow] = {}
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._logger = logging.getLogger("cloud.orchestrator")
        self._throttled_log_ts: dict[str, float] = {}

    def _log_throttled(
        self, key: str, interval_sec: float, level: str, msg: str, *args
    ) -> None:
        """按 key 节流日志，避免高频轮询阶段刷屏。"""
        now = time.time()
        last = self._throttled_log_ts.get(key, 0.0)
        if now - last < interval_sec:
            return
        self._throttled_log_ts[key] = now
        getattr(self._logger, level)(msg, *args)

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_loop(), name="session-orchestrator")
        self._logger.info("session orchestrator started")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._logger.info("session orchestrator stopped")

    async def register_session(self, session_id: str) -> None:
        async with self._lock:
            self._workflows.setdefault(
                session_id, SessionWorkflow(session_id=session_id)
            )
        self._logger.info("session registered session_id=%s", session_id)

    async def _run_loop(self) -> None:
        """后台调度循环。"""
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover
                self._logger.exception("Orchestrator tick failed: %s", exc)
            await asyncio.sleep(0.5)

    async def _tick(self) -> None:
        """遍历当前所有会话并推进状态机。"""
        async with self._lock:
            session_ids = list(self._workflows.keys())

        for session_id in session_ids:
            await self._tick_session(session_id)

    async def _tick_session(self, session_id: str) -> None:
        """推进单个会话：在线检查、命令下发、启动与收敛。"""
        session = self._session_service.get(session_id)
        if session is None:
            async with self._lock:
                self._workflows.pop(session_id, None)
            self._logger.info(
                "session removed from orchestrator session_id=%s", session_id
            )
            return

        async with self._lock:
            workflow = self._workflows.setdefault(
                session_id, SessionWorkflow(session_id=session_id)
            )

        online_nodes = {
            row["node_id"]: row for row in await self._node_manager.list_online()
        }
        required = self._session_service.node_ids(session)
        now_ms = int(time.time() * 1000)

        if session.status in {"BINDING_TIMEOUT", "RACE_TIMEOUT", "FINISHED", "ABORTED"}:
            await self._send_stop_if_needed(
                session, workflow, required, reason=session.status
            )
            return

        all_online = all(
            bool(online_nodes.get(node_id, {}).get("online")) for node_id in required
        )
        if not all_online:
            offline_nodes = [
                node_id
                for node_id in required
                if not bool(online_nodes.get(node_id, {}).get("online"))
            ]
            self._log_throttled(
                f"offline:{session_id}",
                5,
                "warning",
                "session waiting offline nodes session=%s status=%s offline_nodes=%s",
                session_id,
                session.status,
                offline_nodes,
            )
            return

        # 阶段 1：确保所有 required 节点都收到 INIT。
        init_dispatched = False
        for node_id in required:
            if node_id in workflow.init_sent_to:
                continue
            command = self._session_service.build_init_command(session, node_id)
            delivered = await self._node_manager.send_command(node_id, command)
            if delivered:
                workflow.init_sent_to.add(node_id)
                init_dispatched = True
        if init_dispatched:
            workflow.init_sent_at_ms = int(time.time() * 1000)
            self._session_service.update_status(session.session_id, "INIT_SENT")
            self._logger.info(
                "init commands dispatched session=%s delivered=%s/%s",
                session.session_id,
                len(workflow.init_sent_to),
                len(required),
            )

        if len(workflow.init_sent_to) < len(required):
            return

        # 阶段 2：确保所有 required 节点都收到 BINDING。
        binding_dispatched = False
        for node_id in required:
            if node_id in workflow.binding_sent_to:
                continue
            command = self._session_service.build_binding_command(session, node_id)
            delivered = await self._node_manager.send_command(node_id, command)
            if delivered:
                workflow.binding_sent_to.add(node_id)
                binding_dispatched = True
        if binding_dispatched:
            workflow.binding_sent_at_ms = int(time.time() * 1000)
            self._session_service.update_status(session.session_id, "BINDING_SENT")
            self._logger.info(
                "binding commands dispatched session=%s delivered=%s/%s",
                session.session_id,
                len(workflow.binding_sent_to),
                len(required),
            )

        if len(workflow.binding_sent_to) < len(required):
            return

        if (
            session.require_bindings
            and not workflow.start_sent
            and workflow.binding_sent_at_ms is not None
            and now_ms - workflow.binding_sent_at_ms
            >= int(session.binding_timeout_sec * 1000)
        ):
            self._logger.warning(
                "binding timeout reached session=%s timeout_sec=%s",
                session.session_id,
                session.binding_timeout_sec,
            )
            self._session_service.finish(
                session.session_id,
                status="BINDING_TIMEOUT",
                reason="No valid binding completed before timeout",
                finished_at_ms=now_ms,
            )
            await self._send_stop_if_needed(
                session, workflow, required, reason="BINDING_TIMEOUT"
            )
            return

        if not session.auto_start or workflow.start_sent:
            # 若不自动开赛或已开赛，转入终态收敛检查。
            await self._finalize_running_session_if_needed(
                session, workflow, required, now_ms
            )
            return

        all_ready = True
        blocking_node: int | None = None
        blocking_reason = ""
        for node_id in required:
            row = online_nodes.get(node_id)
            last_status = row.get("last_status") if row else None
            status_data = (last_status or {}).get("data", {})
            if (last_status or {}).get("session_id") != session.session_id:
                all_ready = False
                blocking_node = node_id
                blocking_reason = "session_not_synced"
                break
            node_role = row.get("node_role") if row else None
            if not self._session_service.is_node_ready(session, status_data, node_role):
                all_ready = False
                blocking_node = node_id
                blocking_reason = "node_not_ready"
                break

        if not all_ready:
            self._log_throttled(
                f"not_ready:{session_id}",
                5,
                "info",
                "session waiting readiness session=%s blocking_node=%s reason=%s",
                session_id,
                blocking_node,
                blocking_reason,
            )
            return

        # 阶段 3：全部就绪后下发统一起跑时间。
        if workflow.all_ready_at_ms is None:
            workflow.all_ready_at_ms = int(time.time() * 1000)
            self._logger.info("session all required nodes ready session=%s", session_id)
        active_lanes = self._derive_active_lanes(session, online_nodes)
        if active_lanes:
            self._session_service.set_active_lanes(session.session_id, active_lanes)
        expected_start_time = int(time.time() * 1000) + session.start_delay_ms
        self._session_service.set_expected_start_time(
            session.session_id, expected_start_time
        )
        queued = 0
        for node_id in required:
            command = self._session_service.build_start_command(
                session=session,
                node_id=node_id,
                expected_start_time=expected_start_time,
                countdown_seconds=session.countdown_seconds,
                audio_plan=session.audio_plan,
                tracking_active=session.tracking_active,
            )
            delivered = await self._node_manager.send_command(node_id, command)
            if delivered:
                queued += 1

        if queued == len(required):
            workflow.start_sent = True
            workflow.start_sent_at_ms = int(time.time() * 1000)
            self._session_service.update_status(session.session_id, "RUNNING")
            self._logger.info(
                "start monitor dispatched session=%s expected_start_time=%s nodes=%s",
                session.session_id,
                expected_start_time,
                required,
            )
        else:
            self._logger.warning(
                "start monitor partial dispatch session=%s delivered=%s/%s",
                session.session_id,
                queued,
                len(required),
            )

        await self._finalize_running_session_if_needed(
            session, workflow, required, now_ms
        )

    async def _finalize_running_session_if_needed(
        self,
        session,
        workflow: SessionWorkflow,
        required: list[int],
        now_ms: int,
    ) -> None:
        """根据会话报告判断是否应结束比赛并触发 STOP。"""
        if not workflow.start_sent or session.expected_start_time is None:
            return
        if session.status in {"BINDING_TIMEOUT", "RACE_TIMEOUT", "FINISHED", "ABORTED"}:
            await self._send_stop_if_needed(
                session, workflow, required, reason=session.status
            )
            return

        reports = await self._node_manager.get_session_reports(session.session_id)
        target_lanes = set(self._session_service.target_lanes(session))
        finish_lanes = {
            int(item.get("lane"))
            for report in reports["finishes"]
            for item in report.get("data") or []
            if isinstance(item, dict) and isinstance(item.get("lane"), int)
        }
        false_start_lanes = {
            int(item.get("lane"))
            for report in reports["violations"]
            for item in report.get("data") or []
            if isinstance(item, dict)
            and item.get("event") == "FALSE_START"
            and isinstance(item.get("lane"), int)
        }
        terminal_lanes = finish_lanes | false_start_lanes

        if target_lanes and target_lanes.issubset(terminal_lanes):
            self._logger.info(
                "session finished all lanes terminal session=%s target_lanes=%s terminal_lanes=%s",
                session.session_id,
                sorted(target_lanes),
                sorted(terminal_lanes),
            )
            self._session_service.finish(
                session.session_id,
                status="FINISHED",
                reason="All target lanes reached a terminal result",
                finished_at_ms=now_ms,
            )
            await self._send_stop_if_needed(
                session, workflow, required, reason="FINISHED"
            )
            return

        race_deadline_ms = int(session.expected_start_time) + int(
            session.race_timeout_sec * 1000
        )
        if now_ms >= race_deadline_ms:
            self._logger.warning(
                "race timeout reached session=%s deadline_ms=%s now_ms=%s",
                session.session_id,
                race_deadline_ms,
                now_ms,
            )
            self._session_service.finish(
                session.session_id,
                status="RACE_TIMEOUT",
                reason="Race timeout reached before all lanes finished",
                finished_at_ms=now_ms,
            )
            await self._send_stop_if_needed(
                session, workflow, required, reason="RACE_TIMEOUT"
            )

    @staticmethod
    def _derive_active_lanes(session, online_nodes: dict[int, dict]) -> list[int]:
        start_row = online_nodes.get(session.start_node_id) or {}
        last_status = start_row.get("last_status") or {}
        status_data = last_status.get("data") or {}
        for key in ("binding_target_lanes", "binding_confirmed_lanes", "binding_observed_lanes"):
            values = status_data.get(key)
            if isinstance(values, list):
                lanes = [int(lane) for lane in values if isinstance(lane, int)]
                if lanes:
                    return sorted(dict.fromkeys(lanes))
        return []

    async def _send_stop_if_needed(
        self,
        session,
        workflow: SessionWorkflow,
        required: list[int],
        reason: str,
    ) -> None:
        """幂等发送 STOP，避免重复停机命令。"""
        if workflow.stop_sent:
            return
        delivered = 0
        for node_id in required:
            command = self._session_service.build_stop_command(
                session, node_id, reason=reason
            )
            if await self._node_manager.send_command(node_id, command):
                delivered += 1
        if delivered:
            workflow.stop_sent = True
            workflow.stop_sent_at_ms = int(time.time() * 1000)
            self._logger.info(
                "stop dispatched session=%s reason=%s delivered=%s/%s",
                session.session_id,
                reason,
                delivered,
                len(required),
            )

    async def get_workflow_snapshot(self, session_id: str) -> dict | None:
        async with self._lock:
            workflow = self._workflows.get(session_id)
            if workflow is None:
                return None
            return {
                "session_id": workflow.session_id,
                "init_sent_to": sorted(workflow.init_sent_to),
                "binding_sent_to": sorted(workflow.binding_sent_to),
                "start_sent": workflow.start_sent,
                "stop_sent": workflow.stop_sent,
                "init_sent_at_ms": workflow.init_sent_at_ms,
                "binding_sent_at_ms": workflow.binding_sent_at_ms,
                "all_ready_at_ms": workflow.all_ready_at_ms,
                "start_sent_at_ms": workflow.start_sent_at_ms,
                "stop_sent_at_ms": workflow.stop_sent_at_ms,
            }


def get_orchestrator(
    session_service: SessionService, node_manager: NodeConnectionManager
) -> SessionOrchestrator:
    if not hasattr(get_orchestrator, "_orchestrator"):
        get_orchestrator._orchestrator = SessionOrchestrator(
            session_service, node_manager
        )
    return get_orchestrator._orchestrator  # type: ignore[attr-defined]
