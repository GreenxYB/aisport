import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np

from ...core.config import get_settings
from ...core.state import NodePhase, NodeState
from .face_binding import FaceBindingAlgo
from .lane_layout import binding_target_lanes, build_lane_shapes, resolve_lane_by_point
from .violation import ViolationAlgo
from .finish_line import FinishLineAlgo


def _to_jsonable(value: Any) -> Any:
    """递归转换 numpy 类型，确保事件可 JSON 序列化。"""
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
    """算法统一调度器。

    将人脸绑定、起跑违规、冲线判定收敛到同一入口，
    并在 `_finalize_events()` 统一补字段、落盘与上报。
    """

    def __init__(self, state: NodeState, publisher: Any = None):
        self.settings = get_settings()
        self.state = state
        self.publisher = publisher
        self.face = FaceBindingAlgo()
        self.violation = ViolationAlgo(self.state)
        self.finish = FinishLineAlgo(self.state)
        self.logger = logging.getLogger("edge.binding")
        self._last_run_ms: Optional[int] = None
        self._last_face_report_ms: Optional[int] = None
        self._last_face_signature: Optional[str] = None
        self._last_binding_diag_ms: Optional[int] = None
        self._log_path = Path(self.settings.algo_log_path)

    def reset_binding_runtime(self) -> None:
        """Reset per-session face-binding runtime cache."""
        self._last_face_report_ms = None
        self._last_face_signature = None
        self._last_binding_diag_ms = None
        self.face.reset(self.state.session_id)

    def _binding_diag(self, ts_ms: float, message: str, *args) -> None:
        """Throttled binding diagnostics to avoid log flooding."""
        now_ms = int(ts_ms)
        if (
            self._last_binding_diag_ms is not None
            and now_ms - self._last_binding_diag_ms < 1500
        ):
            return
        self._last_binding_diag_ms = now_ms
        self.logger.info(message, *args)

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
        # 按节点角色执行对应业务，避免无关算法占用算力。
        role = (self.settings.node_role or "START").upper()
        events: List[Dict] = []
        self.face.bind_session(self.state.session_id)
        self._record_lane_debug(frame, ts_ms, dets, track_ids)

        if role in {"START", "ALL_IN_ONE"}:
            events.extend(self._run_face_binding(frame, ts_ms, dets, track_ids))

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

        if self.state.phase == NodePhase.MONITORING and role in {
            "FINISH",
            "ALL_IN_ONE",
        }:
            finish_report = self.finish.process_detections(
                dets=dets,
                track_ids=track_ids,
                ts_ms=ts_ms,
                frame_shape=frame.shape[:2],
            )
            if finish_report:
                events.append(finish_report)

        return events

    def _record_lane_debug(
        self,
        frame: np.ndarray,
        ts_ms: float,
        dets: List[Dict],
        track_ids: List[int],
    ) -> None:
        # 记录赛道几何与观测快照，供状态/诊断接口读取。
        if frame is None or frame.size == 0:
            return
        lane_targets = self._binding_target_lanes()
        shapes = build_lane_shapes(
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            target_lanes=lane_targets,
            lane_ranges_text=self.settings.lane_x_ranges,
            lane_polygons_text=self.settings.lane_polygons,
            lane_layout_file=self.settings.lane_layout_file,
        )
        observations = []
        for idx, det in enumerate(dets):
            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox[:4]
            center_x = float((x1 + x2) / 2.0)
            center_y = float((y1 + y2) / 2.0)
            lane = resolve_lane_by_point(
                x=center_x,
                y=center_y,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                target_lanes=lane_targets,
                lane_ranges_text=self.settings.lane_x_ranges,
                lane_polygons_text=self.settings.lane_polygons,
                lane_layout_file=self.settings.lane_layout_file,
            )
            observations.append(
                {
                    "track_id": track_ids[idx] if idx < len(track_ids) else None,
                    "lane": lane,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [center_x, center_y],
                    "score": float(det.get("score") or 0.0),
                }
            )
        self.state.lane_layout_debug = {
            "target_lanes": lane_targets,
            "shapes": shapes,
            "observations": observations,
        }
        self.state.last_lane_observation_ts = int(ts_ms)

    def _run_face_binding(
        self, frame: np.ndarray, ts_ms: float, dets: List[Dict], track_ids: List[int]
    ) -> List[Dict]:
        # 仅在绑定阶段（或开跑前窗口）执行人脸绑定。
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
        if (
            self._last_face_report_ms is not None
            and now_ms - self._last_face_report_ms < interval_ms
        ):
            self._binding_diag(
                ts_ms,
                "binding skip: interval throttle remain_ms=%s",
                interval_ms - (now_ms - self._last_face_report_ms),
            )
            return []

        lane_targets = self._binding_target_lanes()
        confirmed_lanes = {
            int(lane)
            for lane in self.state.binding_confirmed_lanes
            if isinstance(lane, int)
        }
        pending_lanes = [lane for lane in lane_targets if lane not in confirmed_lanes]
        if lane_targets and not pending_lanes:
            return []

        candidates = self._build_face_candidates(frame, dets, track_ids, lane_targets)
        if not candidates:
            self._binding_diag(
                ts_ms,
                "binding skip: no candidates target_lanes=%s dets=%s",
                lane_targets,
                len(dets),
            )
            return []

        events = self.face.process_candidates(candidates, ts_ms)
        if not events:
            candidate_lanes = [item.get("lane") for item in candidates]
            self._binding_diag(
                ts_ms,
                "binding skip: face no match candidate_lanes=%s candidates=%s",
                candidate_lanes,
                len(candidates),
            )
            return []

        signature = json.dumps(
            events[0].get("data", []), ensure_ascii=False, sort_keys=True
        )
        # 对同一识别结果做签名去重，避免重复上报。
        if (
            signature == self._last_face_signature
            and self._last_face_report_ms is not None
        ):
            lanes = [
                item.get("lane")
                for item in (events[0].get("data") or [])
                if isinstance(item, dict)
            ]
            self._binding_diag(
                ts_ms,
                "binding skip: duplicate report lanes=%s",
                lanes,
            )
            return []

        self._last_face_signature = signature
        self._last_face_report_ms = now_ms
        lanes = [
            item.get("lane")
            for item in (events[0].get("data") or [])
            if isinstance(item, dict)
        ]
        students = [
            item.get("student_id")
            for item in (events[0].get("data") or [])
            if isinstance(item, dict)
        ]
        self.logger.info(
            "binding report generated lanes=%s students=%s",
            lanes,
            students,
        )
        return events

    def _binding_target_lanes(self) -> List[int]:
        lane_count = int(self.state.config.get("lane_count", 0) or 0)
        return binding_target_lanes(self.state.bindings, lane_count)

    def _build_face_candidates(
        self,
        frame: np.ndarray,
        dets: List[Dict],
        track_ids: List[int],
        lane_targets: List[int],
    ) -> List[Dict]:
        # 每条赛道仅保留一个最优候选框用于人脸识别。
        if frame is None or frame.size == 0 or not dets or not lane_targets:
            return []

        confirmed_lanes = {
            int(lane)
            for lane in self.state.binding_confirmed_lanes
            if isinstance(lane, int)
        }
        shapes = build_lane_shapes(
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            target_lanes=lane_targets,
            lane_ranges_text=self.settings.lane_x_ranges,
            lane_polygons_text=self.settings.lane_polygons,
            lane_layout_file=self.settings.lane_layout_file,
        )
        best_by_lane: Dict[int, Dict[str, Any]] = {}
        for idx, det in enumerate(dets):
            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox[:4]
            center_x = float((x1 + x2) / 2.0)
            center_y = float((y1 + y2) / 2.0)
            lane = resolve_lane_by_point(
                x=center_x,
                y=center_y,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                target_lanes=lane_targets,
                lane_ranges_text=self.settings.lane_x_ranges,
                lane_polygons_text=self.settings.lane_polygons,
                lane_layout_file=self.settings.lane_layout_file,
            )
            if lane is None:
                continue
            if lane in confirmed_lanes:
                continue
            score = float(det.get("score") or 0.0)
            area = max(1.0, float(x2 - x1) * float(y2 - y1))
            track_id = track_ids[idx] if idx < len(track_ids) else None
            current = best_by_lane.get(lane)
            if current is None or (score, area) > (current["score"], current["area"]):
                best_by_lane[lane] = {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "score": score,
                    "area": area,
                    "track_id": int(track_id) if track_id is not None else None,
                    "center": [center_x, center_y],
                }

        candidates: List[Dict] = []
        for shape in shapes:
            lane = int(shape["lane"])
            current = best_by_lane.get(lane)
            if current is None:
                continue
            x1, y1, x2, y2 = current["bbox"]
            h, w = frame.shape[:2]
            xi1 = max(0, min(w - 1, int(x1)))
            yi1 = max(0, min(h - 1, int(y1)))
            xi2 = max(xi1 + 1, min(w, int(x2)))
            yi2 = max(yi1 + 1, min(h, int(y2)))
            crop = frame[yi1:yi2, xi1:xi2]
            if crop.size == 0:
                continue
            binding_key = self._face_candidate_key(
                lane=lane,
                track_id=current.get("track_id"),
                center=current.get("center"),
            )
            candidates.append(
                {
                    "lane": lane,
                    "image": crop,
                    "bbox": current["bbox"],
                    "track_id": current.get("track_id"),
                    "binding_key": binding_key,
                }
            )
        return candidates

    @staticmethod
    def _face_candidate_key(
        lane: int, track_id: int | None, center: List[float] | None
    ) -> str:
        if track_id is not None:
            return f"track:{int(track_id)}"
        if isinstance(center, list) and len(center) >= 2:
            cx = int(round(float(center[0]) / 80.0))
            cy = int(round(float(center[1]) / 80.0))
            return f"lane:{lane}:cell:{cx}:{cy}"
        return f"lane:{lane}"

    def _extract_tracker_inputs(
        self, tracker_result
    ) -> tuple[List[Dict], List[int], List]:
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
        # 统一补充公共字段，保证事件结构一致。
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
                    confirmed_lanes = list(self.state.binding_confirmed_lanes)
                    assignments = [
                        item
                        for item in self.state.binding_assignments
                        if isinstance(item, dict)
                    ]
                    assignment_map = {
                        int(item["lane"]): item
                        for item in assignments
                        if isinstance(item.get("lane"), int)
                    }
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        student_id = item.get("student_id")
                        lane = item.get("lane")
                        if student_id and student_id not in confirmed_students:
                            confirmed_students.append(str(student_id))
                        if isinstance(lane, int) and lane not in confirmed_lanes:
                            confirmed_lanes.append(lane)
                        if isinstance(lane, int):
                            assignment_map[lane] = item
                    self.state.binding_confirmed_students = confirmed_students
                    self.state.binding_confirmed_lanes = confirmed_lanes
                    self.state.binding_assignments = [
                        assignment_map[lane] for lane in sorted(assignment_map)
                    ]
                    if confirmed_students or confirmed_lanes:
                        self.state.binding_confirmed_at_ms = int(ts_ms)
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        lane = item.get("lane")
                        if not isinstance(lane, int):
                            continue
                        student_id = item.get("student_id")
                        name = item.get("name")
                        confidence = item.get("confidence")
                        self.logger.info(
                            # 关键里程碑：人脸识别成功（lane -> 人员映射）。
                            "face recognized lane=%s student_id=%s name=%s confidence=%s",
                            lane,
                            student_id,
                            name,
                            confidence,
                        )
                if ev.get("msg_type") == "VIOLATION_EVENT":
                    data = ev.get("data") or []
                    if data and isinstance(data, list):
                        first_item = data[0]
                        if (
                            isinstance(first_item, dict)
                            and first_item.get("event") == "FALSE_START"
                        ):
                            self.state.last_false_start_event = first_item
                            self.state.last_false_start_ts = int(ts_ms)
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        if item.get("event") != "FALSE_START":
                            continue
                        lane = item.get("lane")
                        track_id = item.get("track_id")
                        student = self._binding_assignment_for_lane(lane)
                        self.logger.info(
                            # 关键里程碑：起跑线越线（抢跑）。
                            "start line crossed lane=%s track_id=%s student_id=%s name=%s",
                            lane,
                            track_id,
                            student.get("student_id"),
                            student.get("name"),
                        )
                if ev.get("msg_type") == "FINISH_REPORT":
                    for item in ev.get("data") or []:
                        if not isinstance(item, dict):
                            continue
                        lane = item.get("lane")
                        track_id = item.get("track_id")
                        finish_ts = item.get("finish_ts")
                        rank = item.get("rank")
                        student = self._binding_assignment_for_lane(lane)
                        self.logger.info(
                            # 关键里程碑：到达终点。
                            "finish reached lane=%s track_id=%s student_id=%s name=%s finish_ts=%s rank=%s",
                            lane,
                            track_id,
                            student.get("student_id"),
                            student.get("name"),
                            finish_ts,
                            rank,
                        )
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
        """发布事件到上游（通常为 websocket 客户端）。"""
        if not self.publisher:
            return
        for ev in events:
            try:
                self.publisher.publish(ev)
            except Exception:
                pass

    def _binding_assignment_for_lane(self, lane: Any) -> Dict[str, Any]:
        """按赛道号回查绑定信息，优先实时确认结果。"""
        if not isinstance(lane, int):
            return {}
        for item in self.state.binding_assignments:
            if not isinstance(item, dict):
                continue
            if item.get("lane") == lane:
                return item
        for item in self.state.bindings:
            if not isinstance(item, dict):
                continue
            if item.get("lane") == lane:
                return item
        return {}
