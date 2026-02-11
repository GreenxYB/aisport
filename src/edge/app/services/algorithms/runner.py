import json
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from ...core.config import get_settings
from ...core.state import NodePhase, NodeState
from .face_binding import FaceBindingAlgo
from .violation import ViolationAlgo
from .finish_line import FinishLineAlgo


class AlgorithmRunner:
    def __init__(self, state: NodeState):
        self.settings = get_settings()
        self.state = state
        self.face = FaceBindingAlgo()
        self.violation = ViolationAlgo()
        self.finish = FinishLineAlgo()
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
            finish_report = self.finish.process(frame, ts_ms)
            if finish_report:
                events.append(finish_report)

        # Add common fields
        for ev in events:
            ev.setdefault("node_id", self.state.node_id)
            ev.setdefault("session_id", self.state.session_id)

        if events:
            self._append(events)
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
