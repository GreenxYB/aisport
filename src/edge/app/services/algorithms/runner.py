import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np

from ...core.config import get_settings
from ...core.state import NodePhase, NodeState
from .face_binding import FaceBindingAlgo
from .violation import ViolationAlgo
from .finish_line import FinishLineAlgo


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class AlgorithmRunner:
    def __init__(self, state: NodeState, publisher: Any = None):
        self.settings = get_settings()
        self.state = state
        self.publisher = publisher
        self.face = FaceBindingAlgo()
        self.violation = ViolationAlgo(self.state)
        self.finish = FinishLineAlgo(self.state)
        self._last_run_ms: Optional[int] = None
        self._log_path = Path(self.settings.algo_log_path)

    def process_frame(self, frame: np.ndarray, ts_ms: float) -> List[Dict]:
        if not self.settings.algo_enabled:
            return []
        if not self._should_run(ts_ms):
            return []
        events: List[Dict] = []

        # Placeholder calls; replace with real models later
        if self.state.phase == NodePhase.BINDING:
            events.extend(self.face.process(frame, ts_ms))
        if self.state.phase == NodePhase.MONITORING:
            events.extend(self.violation.process(frame, ts_ms))
            finish_report = self.finish.process_detections(
                dets=self.violation.last_dets,
                track_ids=self.violation.last_track_ids,
                ts_ms=ts_ms,
                frame_shape=frame.shape[:2],
            )
            if finish_report:
                events.append(finish_report)

        # Add common fields
        for ev in events:
            ev.setdefault("node_id", self.state.node_id)
            ev.setdefault("session_id", self.state.session_id)
            ev.setdefault("timestamp", int(ts_ms))

        events = [_to_jsonable(ev) for ev in events]

        if events:
            for ev in events:
                if ev.get("msg_type") == "ID_REPORT":
                    self.state.last_face_result = ev
                    self.state.last_face_ts = int(ts_ms)
                if ev.get("msg_type") == "VIOLATION_EVENT":
                    data = ev.get("data") or []
                    if data and isinstance(data, list):
                        first_item = data[0]
                        if isinstance(first_item, dict) and first_item.get("event") == "FALSE_START":
                            self.state.last_false_start_event = first_item
                            self.state.last_false_start_ts = int(ts_ms)
            self._append(events)
            self._publish(events)
            self.state.algo_events_generated += len(events)
            self.state.last_algo_ts = int(ts_ms)

        return events

    def process_pipeline_result(
        self, frame: np.ndarray, tracker_result, ts_ms: float
    ) -> List[Dict]:
        """
        Process the result from the inference pipeline.
        tracker_result: TrackerResults object containing tracks and keypoints.
        """
        # Update capture stats here or in pipeline.
        # For now, we delegate to the existing frame processing logic
        # which handles phase checks and event generation.
        # In the future, we can use tracker_result.result (tracks) and tracker_result.keypoints directly.

        return self.process_frame(frame, ts_ms)

    def _should_run(self, ts_ms: float) -> bool:
        interval_ms = int(1000 / max(self.settings.algo_target_fps, 1))
        now = int(ts_ms)
        if self._last_run_ms is None or now - self._last_run_ms >= interval_ms:
            self._last_run_ms = now
            return True
        return False

    def _append(self, events: List[Dict]) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def _publish(self, events: List[Dict]) -> None:
        if not self.publisher:
            return
        for ev in events:
            try:
                self.publisher.publish(ev)
            except Exception:
                pass
