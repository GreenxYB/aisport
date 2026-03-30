from typing import Any, Dict, List, Optional


def ankle_points_from_keypoints(
    keypoints: Any,
    conf_thres: float,
) -> List[Dict[str, Any]]:
    """提取左右踝关键点（COCO 15/16）。"""
    if not keypoints or len(keypoints) < 17:
        return []

    points: List[Dict[str, Any]] = []
    joints = ((15, "left"), (16, "right"))
    for ankle_idx, side in joints:
        try:
            ankle_x, ankle_y, ankle_s = keypoints[ankle_idx]
        except Exception:
            continue
        if ankle_s is None or ankle_s < conf_thres:
            continue
        points.append(
            {
                "side": side,
                "ankle": [float(ankle_x), float(ankle_y)],
            }
        )
    return points


def toe_proxy_points_from_keypoints(
    keypoints: Any,
    conf_thres: float,
    toe_scale: float,
) -> List[Dict[str, Any]]:
    """生成脚尖代理点。

    优先使用膝-踝向量外推；不可用时回退到踝点本身。
    """
    if not keypoints or len(keypoints) < 17:
        return []

    points: List[Dict[str, Any]] = []
    # COCO: left(13 knee, 15 ankle), right(14 knee, 16 ankle)
    joints = ((13, 15, "left"), (14, 16, "right"))
    for knee_idx, ankle_idx, side in joints:
        try:
            ankle_x, ankle_y, ankle_s = keypoints[ankle_idx]
        except Exception:
            continue

        if ankle_s is None or ankle_s < conf_thres:
            continue

        toe_x = float(ankle_x)
        toe_y = float(ankle_y)
        method = "ankle_fallback"
        try:
            knee_x, knee_y, knee_s = keypoints[knee_idx]
            if knee_s is not None and knee_s >= conf_thres:
                dx = float(ankle_x) - float(knee_x)
                dy = float(ankle_y) - float(knee_y)
                toe_x = float(ankle_x) + float(toe_scale) * dx
                toe_y = float(ankle_y) + float(toe_scale) * dy
                method = "knee_ankle_proxy"
        except Exception:
            pass

        points.append(
            {
                "side": side,
                "ankle": [float(ankle_x), float(ankle_y)],
                "toe": [toe_x, toe_y],
                "method": method,
            }
        )
    return points


def max_measure_y_for_finish(
    bbox: Optional[List[float]],
    keypoints: Any,
    conf_thres: float,
    toe_scale: float,
) -> Optional[float]:
    """计算用于终点判定的 y 测量值（优先脚尖代理，兜底 bbox 底边）。"""
    toe_points = toe_proxy_points_from_keypoints(keypoints, conf_thres, toe_scale)
    if toe_points:
        y_list = []
        for point in toe_points:
            toe = point.get("toe")
            if isinstance(toe, list) and len(toe) >= 2:
                y_list.append(float(toe[1]))
        if y_list:
            return max(y_list)
    if isinstance(bbox, list) and len(bbox) >= 4:
        return float(bbox[3])
    return None


def max_ankle_measure_delta_for_finish(
    keypoints: Any,
    conf_thres: float,
    line_y_at_x,
) -> Optional[float]:
    """计算踝点相对终点线的最大垂直差值。"""
    ankle_points = ankle_points_from_keypoints(keypoints, conf_thres)
    if not ankle_points:
        return None
    deltas = []
    for point in ankle_points:
        ankle = point.get("ankle")
        if isinstance(ankle, list) and len(ankle) >= 2:
            deltas.append(float(ankle[1]) - float(line_y_at_x(float(ankle[0]))))
    if not deltas:
        return None
    return max(deltas)


class FinishLineJudge:
    """终点过线判定器。

    通过连续两帧测量值（prev_y -> measure_y）判断是否发生跨线，
    并为首次跨线目标分配名次。
    """

    def __init__(self) -> None:
        self._last_y_by_track: dict[int, float] = {}
        self._finished: dict[int, Dict[str, Any]] = {}
        self._rank = 0

    def reset(self) -> None:
        self._last_y_by_track.clear()
        self._finished.clear()
        self._rank = 0

    @property
    def finished(self) -> dict[int, Dict[str, Any]]:
        return self._finished

    def update(
        self,
        track_id: int,
        measure_y: Optional[float],
        line_y: int,
        current_time: int,
        enabled: bool,
    ) -> Optional[Dict[str, Any]]:
        """更新单个 track 的状态并在跨线时返回事件。"""
        if measure_y is None:
            return None

        prev_y = self._last_y_by_track.get(track_id)
        self._last_y_by_track[track_id] = float(measure_y)

        if not enabled:
            return None
        if track_id in self._finished:
            return None
        if prev_y is None:
            return None
        if not (prev_y < line_y <= float(measure_y)):
            return None

        self._rank += 1
        event = {
            "track_id": track_id,
            "rank": self._rank,
            "finish_ts": int(current_time),
        }
        self._finished[track_id] = event
        return event

