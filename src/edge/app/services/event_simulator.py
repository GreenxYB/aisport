import json
import queue
import random
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

from ..core.config import get_settings
from ..core.state import NodeState


class EventSimulator:
    def __init__(self, state: NodeState):
        self.settings = get_settings()
        self.state = state
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._log_path = Path(__file__).resolve().parents[4] / "logs" / "events.jsonl"
        self._fail_path = Path(__file__).resolve().parents[4] / "logs" / "events_failed.jsonl"
        self._retry_queue: "queue.Queue[dict]" = queue.Queue()
        self._retry_thread: Optional[threading.Thread] = None
        self._retry_running = threading.Event()

    def start(self, session_id: str, lane_count: int) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._loop, args=(session_id, lane_count), daemon=True, name="event-sim"
        )
        self._thread.start()
        if self.settings.report_enabled and self.settings.report_retry_enabled:
            self._start_retry_worker()

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2)
        self._stop_retry_worker()

    def _loop(self, session_id: str, lane_count: int) -> None:
        interval = max(self.settings.event_interval_sec, 0.5)
        finish_interval = max(self.settings.finish_interval_sec, 1.0)
        next_finish = time.time() + finish_interval
        lanes = max(lane_count, 1)
        while self._running.is_set():
            ts_ms = int(time.time() * 1000)
            event = {
                "msg_type": "VIOLATION_EVENT",
                "node_id": self.state.node_id,
                "session_id": session_id,
                "event": random.choice(["FALSE_START", "LANE_DEVIATION"]),
                "lane": random.randint(1, lanes),
                "timestamp": ts_ms,
                "evidence_frame": None,
            }
            self._append(event)
            self._report(event)
            self.state.events_generated += 1
            self.state.last_event_ts = ts_ms

            now = time.time()
            if self.settings.simulate_finish_reports and now >= next_finish:
                finish = {
                    "msg_type": "FINISH_REPORT",
                    "node_id": self.state.node_id,
                    "session_id": session_id,
                    "results": [
                        {"lane": lane, "finish_ts": ts_ms + random.randint(1000, 5000)}
                        for lane in range(1, lanes + 1)
                    ],
                }
                self._append(finish)
                self._report(finish)
                self.state.finish_reports_generated += 1
                self.state.last_finish_ts = ts_ms
                next_finish = now + finish_interval
            time.sleep(interval)

    def _append(self, event: dict) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _report(self, event: dict) -> None:
        if not self.settings.report_enabled:
            return
        ok = self._send(event)
        if ok:
            self.state.reports_sent += 1
            return
        self.state.reports_failed += 1
        if self.settings.report_retry_enabled:
            self._retry_queue.put({"event": event, "attempt": 1})
        self._append_failed(event, Exception("report_failed"))

    def _send(self, event: dict) -> bool:
        msg_type = event.get("msg_type")
        if msg_type == "VIOLATION_EVENT":
            endpoint = "violation"
        elif msg_type == "FINISH_REPORT":
            endpoint = "finish"
        else:
            return True
        url = f"{self.settings.report_base_url.rstrip('/')}/{endpoint}"
        data = json.dumps(event).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.settings.report_timeout_sec) as resp:
                _ = resp.read()
            return True
        except Exception:
            return False

    def _append_failed(self, event: dict, exc: Exception) -> None:
        self._fail_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"event": event, "error": str(exc), "ts_ms": int(time.time() * 1000)}
        with self._fail_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _start_retry_worker(self) -> None:
        if self._retry_running.is_set():
            return
        self._retry_running.set()
        self._retry_thread = threading.Thread(
            target=self._retry_loop, daemon=True, name="event-retry"
        )
        self._retry_thread.start()

    def _stop_retry_worker(self) -> None:
        if not self._retry_running.is_set():
            return
        self._retry_running.clear()
        if self._retry_thread:
            self._retry_thread.join(timeout=2)

    def _retry_loop(self) -> None:
        interval = max(self.settings.report_retry_interval_sec, 1.0)
        while self._retry_running.is_set():
            try:
                item = self._retry_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            event = item.get("event")
            attempt = int(item.get("attempt", 1))
            ok = self._send(event)
            if ok:
                self.state.reports_sent += 1
                continue
            if attempt < self.settings.report_retry_max:
                time.sleep(interval)
                item["attempt"] = attempt + 1
                self._retry_queue.put(item)
            else:
                self.state.reports_failed += 1
                self._append_failed(event, Exception("retry_exhausted"))
