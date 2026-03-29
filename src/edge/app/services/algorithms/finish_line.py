from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...core.config import get_settings
from ...core.state import NodeState, NodePhase
from .lane_layout import binding_target_lanes, resolve_lane_by_point
from .rules import FinishLineJudge, max_measure_y_for_finish


class FinishLineAlgo:
    def __init__(self, state: NodeState) -> None:
        self.settings = get_settings()
        self.state = state
        self._session_id: Optional[str] = None
        self._judge = FinishLineJudge()

    def _sync_session(self) -> None:
        if self.state.session_id != self._session_id:
            self._session_id = self.state.session_id
            self._judge.reset()

    def _scaled_finish_line_y(self, frame_shape: Optional[Tuple[int, int]] = None) -> int:
        line_y = int(self.settings.finish_line_y)
        if frame_shape is None:
            return line_y
        h = frame_shape[0]
        if h <= 0:
            return line_y
        return int(line_y * h / 640)

    def _resolve_lane(
        self,
        track_id: Optional[int],
        idx: int,
        bbox: Optional[List[float]],
        frame_width: int,
        frame_height: int,
    ) -> int:
        if isinstance(bbox, list) and len(bbox) >= 4 and frame_width > 0 and frame_height > 0:
            center_x = float(bbox[0] + bbox[2]) / 2.0
            center_y = float(bbox[1] + bbox[3]) / 2.0
            lane = resolve_lane_by_point(
                x=center_x,
                y=center_y,
                frame_width=frame_width,
                frame_height=frame_height,
                target_lanes=binding_target_lanes(self.state.bindings, int(self.state.config.get("lane_count", 1) or 1)),
                lane_ranges_text=self.settings.lane_x_ranges,
                lane_polygons_text=self.settings.lane_polygons,
            )
            if lane is not None:
                return lane
        if idx < len(self.state.bindings):
            lane = self.state.bindings[idx].get("lane")
            if isinstance(lane, int):
                return lane
        lane_count = int(self.state.config.get("lane_count", 1) or 1)
        if track_id is not None:
            return int(track_id % lane_count) + 1
        return int(idx % lane_count) + 1

    def process_detections(
        self,
        dets: List[Dict[str, Any]],
        track_ids: List[int],
        ts_ms: float,
        frame_shape: Optional[Tuple[int, int]] = None,
        line_y_override: Optional[int] = None,
    ) -> Optional[Dict]:
        self._sync_session()
        if self.state.phase != NodePhase.MONITORING:
            return None

        current_time = int(ts_ms)
        expected_start = self.state.expected_start_time
        if expected_start is not None and current_time < int(expected_start):
            return None

        line_y = (
            int(line_y_override)
            if line_y_override is not None
            else self._scaled_finish_line_y(frame_shape)
        )
        results = []
        for idx, item in enumerate(dets):
            track_id = track_ids[idx] if idx < len(track_ids) else idx
            bbox = item.get("bbox")
            lane = self._resolve_lane(
                track_id,
                idx,
                bbox,
                frame_shape[1] if frame_shape else 0,
                frame_shape[0] if frame_shape else 0,
            )
            keypoints = item.get("keypoints")
            measure_y = max_measure_y_for_finish(
                bbox=bbox,
                keypoints=keypoints,
                conf_thres=self.settings.kps_conf_thres,
                toe_scale=self.settings.toe_proxy_scale,
            )
            finish_ev = self._judge.update(
                track_id=track_id,
                measure_y=measure_y,
                line_y=line_y,
                current_time=current_time,
                enabled=True,
            )
            if not finish_ev:
                continue
            results.append(
                {
                    "lane": lane,
                    "track_id": track_id,
                    "rank": finish_ev["rank"],
                    "finish_ts": finish_ev["finish_ts"],
                }
            )

        if not results:
            return None
        return {
            "msg_type": "FINISH_REPORT",
            "timestamp": current_time,
            "data": results,
            "line_y": line_y,
        }

    def process(self, frame: np.ndarray, ts_ms: float) -> Optional[Dict]:
        # Frame-only path is kept for compatibility.
        _ = frame
        _ = ts_ms
        return None
