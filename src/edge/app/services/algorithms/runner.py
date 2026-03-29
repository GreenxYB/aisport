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
        self._last_face_report_ms: Optional[int] = None
        self._last_face_signature: Optional[str] = None
        self._log_path = Path(self.settings.algo_log_path)

    def process_frame(self, frame: np.ndarray, ts_ms: float) -> List[Dict]:
        if not self.settings.algo_enabled:
            return []
        if not self._should_run(ts_ms):
            return []
        events = self._process_role_logic(
            frame=frame,
            ts_ms=ts_ms,
            dets=[],
            track_ids=[],
            keypoints=[],
        )
        return self._finalize_events(events, ts_ms)

    def process_pipeline_result(
        self, frame: np.ndarray, tracker_result, ts_ms: float
    ) -> List[Dict]:
        """
        Use tracked detections from the pipeline so start/finish business logic
        actually follows the same boxes, IDs, and keypoints shown in preview.
        """
        if not self.settings.algo_enabled:
            return []
        if not self._should_run(ts_ms):
            return []

        dets, track_ids, keypoints = self._extract_tracker_inputs(tracker_result)
        events = self._process_role_logic(
            frame=frame,
            ts_ms=ts_ms,
            dets=dets,
            track_ids=track_ids,
            keypoints=keypoints,
        )
        return self._finalize_events(events, ts_ms)

    def _process_role_logic(
        self,
        frame: np.ndarray,
        ts_ms: float,
        dets: List[Dict],
        track_ids: List[int],
        keypoints: List,
    ) -> List[Dict]:
        role = (self.settings.node_role or "START").upper()
        events: List[Dict] = []

        if role in {"START", "ALL_IN_ONE"}:
            events.extend(self._run_face_binding(frame, ts_ms))

        if self.state.phase == NodePhase.MONITORING and role in {"START", "ALL_IN_ONE"}:
            events.extend(
                self.violation.process_frame_logic(
                    frame=frame,
                    track_ids=track_ids,
                    boxes=dets,
                    kps=keypoints,
                    current_time=int(ts_ms),
                )
            )

        if self.state.phase == NodePhase.MONITORING and role in {"FINISH", "ALL_IN_ONE"}:
            finish_report = self.finish.process_detections(
                dets=dets,
                track_ids=track_ids,
                ts_ms=ts_ms,
                frame_shape=frame.shape[:2],
            )
            if finish_report:
                events.append(finish_report)

        return events

    def _run_face_binding(self, frame: np.ndarray, ts_ms: float) -> List[Dict]:
        expected_start = self.state.expected_start_time
        in_binding_window = self.state.phase == NodePhase.BINDING or (
            self.state.phase == NodePhase.MONITORING
            and expected_start is not None
            and int(ts_ms) < int(expected_start)
        )
        if not in_binding_window:
            return []

        interval_ms = int(max(self.settings.face_report_interval_sec, 0.2) * 1000)
        now_ms = int(ts_ms)
        if self._last_face_report_ms is not None and now_ms - self._last_face_report_ms < interval_ms:
            return []

        events = self.face.process(frame, ts_ms)
        if not events:
            return []

        signature = json.dumps(events[0].get("data", []), ensure_ascii=False, sort_keys=True)
        if signature == self._last_face_signature and self._last_face_report_ms is not None:
            return []

        self._last_face_signature = signature
        self._last_face_report_ms = now_ms
        return events

    def _extract_tracker_inputs(self, tracker_result) -> tuple[List[Dict], List[int], List]:
        if tracker_result is None:
            return [], [], []

        raw_tracks = getattr(tracker_result, "result", None)
        raw_keypoints = getattr(tracker_result, "keypoints", None) or []
        if raw_tracks is None or len(raw_tracks) == 0:
            return [], [], []

        dets: List[Dict] = []
        track_ids: List[int] = []
        for idx, row in enumerate(raw_tracks):
            if len(row) < 7:
                continue
            x1, y1, x2, y2, track_id, conf, cls = row[:7]
            keypoints = raw_keypoints[idx] if idx < len(raw_keypoints) else []
            dets.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(conf),
                    "class_id": int(cls),
                    "keypoints": keypoints,
                }
            )
            track_ids.append(int(track_id))
        return dets, track_ids, [d.get("keypoints") for d in dets]

    def _finalize_events(self, events: List[Dict], ts_ms: float) -> List[Dict]:
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
                    data = ev.get("data") or []
                    confirmed_students = list(self.state.binding_confirmed_students)
                    for item in data:
                        student_id = item.get("student_id") if isinstance(item, dict) else None
                        if student_id and student_id not in confirmed_students:
                            confirmed_students.append(str(student_id))
                    self.state.binding_confirmed_students = confirmed_students
                    if confirmed_students:
                        self.state.binding_confirmed_at_ms = int(ts_ms)
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
