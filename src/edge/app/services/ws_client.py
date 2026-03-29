import asyncio
import contextlib
import json
import logging
import queue
import threading
import time
from typing import Optional

from common.protocol import CommandAckMessage, CommandPayload, NodeConnectPayload


def _to_jsonable(value):
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return value


class EdgeWsClient:
    def __init__(self, handler):
        self.handler = handler
        self.settings = handler.settings
        self.logger = logging.getLogger("edge.ws")
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._outgoing: "queue.Queue[dict]" = queue.Queue()
        self._last_disconnect_log_ts = 0.0
        self._disconnect_count = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.handler.set_publisher(self)
        self._thread = threading.Thread(target=self._run, daemon=True, name="EdgeWsClient")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.handler.set_publisher(None)

    def publish(self, message: dict) -> bool:
        self._outgoing.put(_to_jsonable(message))
        return True

    def _run(self) -> None:
        try:
            asyncio.run(self._run_forever())
        except asyncio.CancelledError:  # pragma: no cover
            return
        except Exception as exc:  # pragma: no cover
            self.logger.error("Websocket client stopped unexpectedly: %s", exc)

    async def _run_forever(self) -> None:
        try:
            import websockets
        except Exception as exc:  # pragma: no cover
            self.logger.error("websockets package unavailable: %s", exc)
            return

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(
                    self.settings.cloud_ws_url,
                    ping_interval=30,
                    ping_timeout=60,
                    close_timeout=5,
                ) as ws:
                    self._disconnect_count = 0
                    self.logger.info("cloud websocket connected url=%s", self.settings.cloud_ws_url)
                    await ws.send(json.dumps(self._build_connect_payload().model_dump()))
                    await ws.send(json.dumps(self.handler.build_status_report().model_dump()))

                    status_task = asyncio.create_task(self._status_loop(ws))
                    sender_task = asyncio.create_task(self._sender_loop(ws))
                    try:
                        async for raw_message in ws:
                            await self._handle_message(ws, raw_message)
                    finally:
                        status_task.cancel()
                        sender_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await status_task
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await sender_task
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._disconnect_count += 1
                now = time.time()
                if now - self._last_disconnect_log_ts >= 10:
                    self._last_disconnect_log_ts = now
                    self.logger.warning(
                        "cloud websocket disconnected retrying count=%s reason=%s",
                        self._disconnect_count,
                        exc,
                    )
                else:
                    self.logger.debug("cloud websocket disconnected: %s", exc)
                await asyncio.sleep(self.settings.ws_reconnect_interval_sec)

    async def _handle_message(self, ws, raw_message: str) -> None:
        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            self.logger.warning("Ignoring non-JSON websocket message: %s", raw_message)
            return

        if payload.get("type") == "CONNECTED":
            self.logger.info("cloud websocket registered node=%s", payload.get("node_id"))
            return

        if "cmd" not in payload:
            self.logger.debug("Ignoring websocket payload without cmd: %s", payload)
            return

        try:
            command = CommandPayload(**payload)
            await asyncio.to_thread(self.handler.handle, command)
            ack = CommandAckMessage(
                node_id=self.settings.node_id,
                session_id=command.session_id,
                timestamp=int(time.time() * 1000),
                cmd=command.cmd,
                status="accepted",
                phase=self.handler.state.phase.value,
            )
            await ws.send(json.dumps(ack.model_dump()))
        except Exception as exc:
            self.logger.error("Failed to handle websocket command: %s", exc)
            session_id = payload.get("session_id", self.handler.state.session_id or "")
            cmd = payload.get("cmd", "UNKNOWN")
            ack = CommandAckMessage(
                node_id=self.settings.node_id,
                session_id=session_id,
                timestamp=int(time.time() * 1000),
                cmd=cmd,
                status="error",
                phase=self.handler.state.phase.value,
                error=str(exc),
            )
            await ws.send(json.dumps(ack.model_dump()))
        finally:
            await ws.send(json.dumps(self.handler.build_status_report().model_dump()))

    async def _status_loop(self, ws) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(self.settings.ws_status_interval_sec)
            await ws.send(json.dumps(self.handler.build_status_report().model_dump()))

    async def _sender_loop(self, ws) -> None:
        while not self._stop_event.is_set():
            try:
                message = self._outgoing.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue

            await ws.send(json.dumps(message))
            if message.get("msg_type") in {"ID_REPORT", "VIOLATION_EVENT", "FINISH_REPORT"}:
                self.handler.state.reports_sent += 1

    def _build_connect_payload(self) -> NodeConnectPayload:
        capabilities = [
            item.strip()
            for item in self.settings.node_capabilities.split(",")
            if item.strip()
        ]
        return NodeConnectPayload(
            node_id=self.settings.node_id,
            node_role=self.settings.node_role,  # type: ignore[arg-type]
            site_id=self.settings.site_id,
            capabilities=capabilities,
        )


def get_ws_client(handler):
    if not hasattr(get_ws_client, "_client"):
        get_ws_client._client = EdgeWsClient(handler)
    return get_ws_client._client  # type: ignore[attr-defined]
