from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

import numpy as np

from ...core.config import get_settings
from ...core.state import NodeState
from .lane_layout import binding_target_lanes, resolve_lane_by_point
from .race_line import inspect_line_definition, line_y_at_x, load_line_definition, point_crossed_line
from .rules import ankle_points_from_keypoints, toe_proxy_points_from_keypoints

logger = logging.getLogger("edge.yolo")


def extract_ultralytics_dets(result) -> List[Dict]:
    """将 Ultralytics 结果标准化为统一检测结构。"""
    if result is None:
        return []
    if result.boxes is None or len(result.boxes) == 0:
        return []
    dets: List[Dict] = []
    boxes = result.boxes
    kps = result.keypoints
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].tolist()
        score = float(boxes.conf[i]) if hasattr(boxes, "conf") else None
        class_id = int(boxes.cls[i]) if hasattr(boxes, "cls") else 0
        keypoints = None
        if kps is not None and kps.data is not None and len(kps.data) > i:
            keypoints = kps.data[i].tolist()
        dets.append(
            {
                "bbox": xyxy,
                "score": score,
                "class_id": class_id,
                "keypoints": keypoints,
            }
        )
    return dets


class ViolationAlgo:
    """违规检测算法。

    - 起跑前：检测 FALSE_START（通过踝点/脚尖代理与起跑线关系）
    - 监控中：按配置节流输出 LANE_DEVIATION 示例事件
    """

    def __init__(self, state: NodeState, load_model: bool = True):
        self.settings = get_settings()
        self.state = state
        self._counter = 0
        self.model = None
        self._names = None
        self._model_type: Optional[str] = None
        self._session_id: Optional[str] = None
        self._false_start_reported: set[int] = set()
        self._last_violation_ms: Optional[int] = None
        self.last_dets: List[Dict[str, Any]] = []
        self.last_track_ids: List[int] = []
        if load_model:
            self._load_model()

    def _load_model(self) -> None:
        """按配置加载 TRT 或 Ultralytics 模型。"""
        engine_path = Path(self.settings.yolo_engine_path)
        names_path = Path(self.settings.yolo_names_path)
        pt_path = Path(self.settings.yolo_pt_path)
        if not engine_path.is_absolute():
            engine_path = Path(self.settings.model_dir) / engine_path
        if not names_path.is_absolute():
            names_path = Path(self.settings.model_dir) / names_path
        if not pt_path.is_absolute():
            pt_path = Path(self.settings.model_dir) / pt_path
        if not engine_path.exists():
            logger.warning("YOLO engine not found: %s", engine_path)
        if not names_path.exists():
            logger.warning("YOLO names not found: %s", names_path)
        if self.settings.yolo_backend.lower() == "trt" and engine_path.exists() and names_path.exists():
            try:
                from .models.yolo_trt import TRTYOLO  # type: ignore
            except Exception as exc:
                logger.error("Failed to import yolo_trt: %s", exc)
            else:
                try:
                    self.model = TRTYOLO(str(engine_path))
                    self._names = [
                        n.strip()
                        for n in names_path.read_text(encoding="utf-8").splitlines()
                        if n.strip()
                    ]
                    # warmup to stabilize latency
                    self.model.warmup(1)
                    self._model_type = "trt"
                    logger.info("YOLO TRT loaded: %s", engine_path)
                    return
                except Exception as exc:
                    logger.error("YOLO TRT init failed: %s", exc)
                    self.model = None

        # Ultralytics .pt
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            logger.error("Failed to import ultralytics: %s", exc)
            return
        if not pt_path.exists():
            candidates = sorted(Path(self.settings.model_dir).glob("*.pt"))
            if candidates:
                pt_path = candidates[0]
            else:
                logger.warning("YOLO .pt not found in %s", self.settings.model_dir)
                return
        try:
            self.model = YOLO(str(pt_path))
            if hasattr(self.model, "names") and self.model.names:
                self._names = [str(self.model.names[i]) for i in sorted(self.model.names.keys())]
            elif names_path.exists():
                self._names = [
                    n.strip()
                    for n in names_path.read_text(encoding="utf-8").splitlines()
                    if n.strip()
                ]
            self._model_type = "ultralytics"
            logger.info("Ultralytics YOLO loaded: %s", pt_path)
        except Exception as exc:
            logger.error("Ultralytics YOLO init failed: %s", exc)
            self.model = None

    def process(self, frame: np.ndarray, ts_ms: float) -> List[Dict]:
        """兼容入口：直接做模型推理并进入违规逻辑。"""
        self._sync_session()
        if not self.model:
            self.last_dets = []
            self.last_track_ids = []
            return []
        self._counter += 1
        try:
            if self._model_type == "trt":
                dets = self.model.infer(
                    frame,
                    conf_thres=self.settings.yolo_conf_thres,
                    iou_thres=self.settings.yolo_iou_thres,
                )
            else:
                dets = self._infer_ultralytics(frame)
        except Exception as exc:
            logger.error("YOLO infer failed: %s", exc)
            self.last_dets = []
            self.last_track_ids = []
            return []

        track_ids = list(range(len(dets)))
        self.last_dets = dets
        self.last_track_ids = track_ids
        boxes = dets
        kps = [d.get("keypoints") for d in dets]
        return self.process_frame_logic(frame, track_ids, boxes, kps, int(ts_ms))

    def _infer_ultralytics(self, frame: np.ndarray) -> List[Dict]:
        results = self.model.predict(
            source=frame,
            conf=self.settings.yolo_conf_thres,
            iou=self.settings.yolo_iou_thres,
            imgsz=self.settings.yolo_imgsz,
            verbose=False,
        )
        if not results:
            return []
        return extract_ultralytics_dets(results[0])

    def _sync_session(self) -> None:
        """会话切换时重置会话级判定缓存。"""
        if self.state.session_id != self._session_id:
            self._session_id = self.state.session_id
            self._false_start_reported.clear()
            self._last_violation_ms = None

    def _resolve_lane(self, track_id: Optional[int], idx: int, item: Any, frame_width: int) -> int:
        """推断目标赛道（几何优先，绑定兜底，索引再兜底）。"""
        bbox = self._extract_box(item)
        if isinstance(bbox, list) and len(bbox) >= 4 and frame_width > 0 and getattr(self, "_frame_height", 0) > 0:
            center_x = float(bbox[0] + bbox[2]) / 2.0
            center_y = float(bbox[1] + bbox[3]) / 2.0
            lane = resolve_lane_by_point(
                x=center_x,
                y=center_y,
                frame_width=frame_width,
                frame_height=self._frame_height,
                target_lanes=binding_target_lanes(self.state.bindings, int(self.state.config.get("lane_count", 1) or 1)),
                lane_ranges_text=self.settings.lane_x_ranges,
                lane_polygons_text=self.settings.lane_polygons,
                lane_layout_file=self.settings.lane_layout_file,
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

    def _extract_box(self, item: Any) -> Optional[List[float]]:
        if isinstance(item, dict):
            bbox = item.get("bbox") or item.get("box")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                return list(bbox[:4])
            return None
        if isinstance(item, (list, tuple)) and len(item) >= 4:
            return list(item[:4])
        return None

    def _extract_score(self, item: Any) -> Optional[float]:
        if isinstance(item, dict):
            score = item.get("score")
            if score is None:
                return None
            try:
                return float(score)
            except (TypeError, ValueError):
                return None
        return None

    def _extract_class_id(self, item: Any) -> int:
        if isinstance(item, dict):
            class_id = item.get("class_id")
            if isinstance(class_id, (int, float)):
                return int(class_id)
        return 0

    def _start_line(self, frame: np.ndarray) -> Dict[str, Any]:
        """加载当前分辨率下的起跑线定义。"""
        frame_h = int(frame.shape[0]) if frame is not None and frame.size > 0 else 640
        frame_w = int(frame.shape[1]) if frame is not None and frame.size > 0 else 1280
        return load_line_definition(
            frame_width=frame_w,
            frame_height=frame_h,
            line_file=self.settings.start_line_file,
            fallback_y=int(self.settings.start_line_y),
            line_name="start_line",
        )

    def _toe_proxy_points(self, keypoints: Any) -> List[Dict[str, Any]]:
        return toe_proxy_points_from_keypoints(
            keypoints=keypoints,
            conf_thres=self.settings.kps_conf_thres,
            toe_scale=self.settings.toe_proxy_scale,
        )

    def _ankle_points(self, keypoints: Any) -> List[Dict[str, Any]]:
        return ankle_points_from_keypoints(
            keypoints=keypoints,
            conf_thres=self.settings.kps_conf_thres,
        )

    def _foot_crossed_line(
        self, keypoints: Any, line: Dict[str, Any]
    ) -> tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """判断脚部关键点是否越过起跑线。"""
        ankle_points = self._ankle_points(keypoints)
        toe_points = self._toe_proxy_points(keypoints)
        for point in ankle_points:
            ankle = point.get("ankle")
            if point_crossed_line(ankle, line):
                return True, ankle_points, toe_points
        return False, ankle_points, toe_points

    def process_frame_logic(
        self,
        frame: np.ndarray,
        track_ids: List[int],
        boxes: List,
        kps: List,
        current_time: int,
    ) -> List[Dict]:
        """违规主逻辑。

        1) expected_start_time 前，仅做 FALSE_START 检测
        2) expected_start_time 后，按节流输出运行中违规事件
        """
        events: List[Dict] = []
        self._frame_height = int(frame.shape[0]) if frame is not None else 0

        expected_start = self.state.expected_start_time
        if expected_start is not None and current_time < int(expected_start):
            if self.state.config.get("false_start_check", True):
                # 起跑前：只关心是否越线，且同赛道仅上报一次 FALSE_START。
                line = self._start_line(frame)
                line_y = int(max(line["p1"][1], line["p2"][1]))
                debug_items: List[Dict[str, Any]] = []
                for idx, item in enumerate(boxes):
                    track_id = track_ids[idx] if idx < len(track_ids) else None
                    lane = self._resolve_lane(track_id, idx, item, frame.shape[1] if frame is not None else 0)
                    kps_item = kps[idx] if idx < len(kps) else None
                    crossed, ankle_points, toe_points = self._foot_crossed_line(kps_item, line)
                    debug_items.append(
                        {
                            "lane": lane,
                            "track_id": track_id,
                            "bbox": self._extract_box(item),
                            "ankle_points": ankle_points,
                            "toe_proxy_points": toe_points,
                            "crossed": crossed,
                        }
                    )
                    if lane in self._false_start_reported:
                        continue
                    if not crossed:
                        continue
                    class_id = self._extract_class_id(item)
                    class_name = (
                        self._names[class_id]
                        if self._names and 0 <= class_id < len(self._names)
                        else str(class_id)
                    )
                    events.append(
                        {
                            "msg_type": "VIOLATION_EVENT",
                            "timestamp": current_time,
                            "data": [
                                {
                                    "event": "FALSE_START",
                                    "lane": lane,
                                    "track_id": track_id,
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "score": self._extract_score(item),
                                    "bbox": self._extract_box(item),
                                    "keypoints": kps_item,
                                    "ankle_points": ankle_points,
                                    "toe_proxy_points": toe_points,
                                    "evidence_frame": None,
                                    "start_line_y": line_y,
                                    "start_line": line,
                                }
                            ],
                        }
                    )
                    self._false_start_reported.add(lane)
                self.state.last_toe_proxy_debug = {
                    "start_line_y": line_y,
                    "start_line": line,
                    "start_line_status": inspect_line_definition(
                        frame_width=int(frame.shape[1]) if frame is not None else 1280,
                        frame_height=int(frame.shape[0]) if frame is not None else 640,
                        line_file=self.settings.start_line_file,
                        fallback_y=int(self.settings.start_line_y),
                        line_name="start_line",
                    ),
                    "items": debug_items,
                }
                self.state.last_toe_proxy_ts = current_time
            return events

        self.state.last_toe_proxy_debug = None
        self.state.last_toe_proxy_ts = None

        if not self.state.config.get("tracking_active", True):
            return events

        # 运行中违规上报节流，避免高频重复事件刷屏。
        interval_ms = int(max(self.settings.event_interval_sec, 0.5) * 1000)
        if self._last_violation_ms is not None and current_time - self._last_violation_ms < interval_ms:
            return events

        for idx, item in enumerate(boxes):
            track_id = track_ids[idx] if idx < len(track_ids) else None
            lane = self._resolve_lane(track_id, idx, item, frame.shape[1] if frame is not None else 0)
            class_id = self._extract_class_id(item)
            class_name = (
                self._names[class_id]
                if self._names and 0 <= class_id < len(self._names)
                else str(class_id)
            )
            events.append(
                {
                    "msg_type": "VIOLATION_EVENT",
                    "timestamp": current_time,
                    "data": [
                        {
                            "event": "LANE_DEVIATION",
                            "lane": lane,
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": class_name,
                            "score": self._extract_score(item),
                            "bbox": self._extract_box(item),
                            "keypoints": kps[idx] if idx < len(kps) else None,
                            "evidence_frame": None,
                        }
                    ],
                }
            )

        if events:
            self._last_violation_ms = current_time
        return events
