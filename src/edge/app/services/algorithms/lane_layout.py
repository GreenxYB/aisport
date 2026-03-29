from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def binding_target_lanes(bindings: List[Dict[str, Any]], lane_count: int) -> List[int]:
    lanes = [
        int(item.get("lane"))
        for item in bindings
        if isinstance(item, dict) and isinstance(item.get("lane"), int)
    ]
    if lanes:
        return sorted(dict.fromkeys(lanes))
    return list(range(1, max(int(lane_count), 0) + 1))


def _clamp_point(x: float, y: float, frame_width: int, frame_height: int) -> Tuple[int, int]:
    xi = max(0, min(frame_width - 1, int(round(x))))
    yi = max(0, min(frame_height - 1, int(round(y))))
    return xi, yi


def _resolve_layout_path(lane_layout_file: str) -> Optional[Path]:
    if not lane_layout_file:
        return None
    path = Path(str(lane_layout_file))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _load_lane_layout_payload(lane_layout_file: str) -> Optional[Dict[str, Any]]:
    path = _resolve_layout_path(lane_layout_file)
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"lanes": payload}
    return None


def parse_lane_polygons(
    lane_polygons_text: str,
    frame_width: int,
    frame_height: int,
) -> Dict[int, List[Tuple[int, int]]]:
    polygons: Dict[int, List[Tuple[int, int]]] = {}
    if not lane_polygons_text:
        return polygons

    raw_text = str(lane_polygons_text).strip()
    if not raw_text:
        return polygons

    # First try JSON:
    # [{"lane":1,"points":[[x,y],...]}]
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                lane = item.get("lane")
                points = item.get("points")
                if not isinstance(lane, int) or not isinstance(points, list):
                    continue
                normalized = []
                for point in points:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        continue
                    normalized.append(_clamp_point(point[0], point[1], frame_width, frame_height))
                if len(normalized) >= 3:
                    polygons[lane] = normalized
            if polygons:
                return polygons
    except Exception:
        pass

    # Plain text:
    # 1:20-40|280-40|240-620|40-620;2:...
    for raw_item in raw_text.split(";"):
        item = raw_item.strip()
        if not item or ":" not in item:
            continue
        lane_text, points_text = item.split(":", 1)
        try:
            lane = int(lane_text.strip())
        except ValueError:
            continue
        normalized = []
        for raw_point in points_text.split("|"):
            point = raw_point.strip()
            if not point or "-" not in point:
                continue
            x_text, y_text = point.split("-", 1)
            try:
                x = float(x_text.strip())
                y = float(y_text.strip())
            except ValueError:
                continue
            normalized.append(_clamp_point(x, y, frame_width, frame_height))
        if len(normalized) >= 3:
            polygons[lane] = normalized
    return polygons


def load_lane_polygons_from_file(
    lane_layout_file: str,
    frame_width: int,
    frame_height: int,
) -> Dict[int, List[Tuple[int, int]]]:
    payload = _load_lane_layout_payload(lane_layout_file)
    if payload is None:
        return {}
    src_width = int(payload.get("frame_width") or frame_width or 1)
    src_height = int(payload.get("frame_height") or frame_height or 1)
    scale_x = frame_width / max(src_width, 1)
    scale_y = frame_height / max(src_height, 1)
    lanes = payload.get("lanes") if isinstance(payload, dict) else payload
    if not isinstance(lanes, list):
        return {}

    polygons: Dict[int, List[Tuple[int, int]]] = {}
    for item in lanes:
        if not isinstance(item, dict):
            continue
        lane = item.get("lane")
        points = item.get("points")
        if not isinstance(lane, int) or not isinstance(points, list):
            continue
        normalized = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            normalized.append(
                _clamp_point(float(point[0]) * scale_x, float(point[1]) * scale_y, frame_width, frame_height)
            )
        if len(normalized) >= 3:
            polygons[lane] = normalized
    return polygons


def available_lane_targets(
    bindings: List[Dict[str, Any]],
    lane_count: int,
    lane_ranges_text: str = "",
    lane_polygons_text: str = "",
    lane_layout_file: str = "",
) -> List[int]:
    payload = _load_lane_layout_payload(lane_layout_file)
    if payload and isinstance(payload.get("lanes"), list):
        lanes = [
            int(item.get("lane"))
            for item in payload["lanes"]
            if isinstance(item, dict) and isinstance(item.get("lane"), int)
        ]
        if lanes:
            return sorted(dict.fromkeys(lanes))

    polygons = parse_lane_polygons(lane_polygons_text, 1280, 640)
    if polygons:
        return sorted(polygons)

    ranges = parse_lane_ranges(lane_ranges_text, 1280)
    if ranges:
        return sorted(ranges)

    return binding_target_lanes(bindings, lane_count)


def inspect_lane_layout(
    *,
    frame_width: int,
    frame_height: int,
    target_lanes: List[int],
    lane_ranges_text: str = "",
    lane_polygons_text: str = "",
    lane_layout_file: str = "",
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "target_lanes": list(target_lanes),
        "source": "auto",
        "calibrated": False,
        "available": False,
        "warning": None,
        "file": lane_layout_file or None,
    }

    if lane_layout_file:
        path = _resolve_layout_path(lane_layout_file)
        info["file"] = str(path)
        if not path.exists():
            info["warning"] = f"lane layout file not found: {path}"
            return info
        payload = _load_lane_layout_payload(str(path)) or {}
        file_width = payload.get("frame_width")
        file_height = payload.get("frame_height")
        polygons = load_lane_polygons_from_file(str(path), frame_width, frame_height)
        covered = sorted(lane for lane in target_lanes if lane in polygons)
        info["source"] = "file"
        info["covered_lanes"] = covered
        info["available"] = bool(polygons)
        info["calibrated"] = bool(polygons)
        info["runtime_frame"] = [frame_width, frame_height]
        info["file_frame"] = [file_width, file_height] if file_width and file_height else None
        info["scaled"] = bool(file_width and file_height and (int(file_width) != frame_width or int(file_height) != frame_height))
        if not polygons:
            info["warning"] = f"lane layout file has no valid polygons: {path}"
        elif covered != sorted(target_lanes):
            missing = [lane for lane in target_lanes if lane not in polygons]
            info["warning"] = f"lane layout file missing lanes: {missing}"
        return info

    polygons = parse_lane_polygons(lane_polygons_text, frame_width, frame_height)
    if polygons:
        info["source"] = "inline_polygon"
        info["available"] = True
        info["calibrated"] = True
        info["covered_lanes"] = sorted(lane for lane in target_lanes if lane in polygons)
        if info["covered_lanes"] != sorted(target_lanes):
            missing = [lane for lane in target_lanes if lane not in polygons]
            info["warning"] = f"inline lane polygons missing lanes: {missing}"
        return info

    ranges = parse_lane_ranges(lane_ranges_text, frame_width)
    if ranges:
        info["source"] = "x_ranges"
        info["available"] = True
        info["calibrated"] = True
        info["covered_lanes"] = sorted(lane for lane in target_lanes if lane in ranges)
        if info["covered_lanes"] != sorted(target_lanes):
            missing = [lane for lane in target_lanes if lane not in ranges]
            info["warning"] = f"lane x ranges missing lanes: {missing}"
        return info

    info["warning"] = "lane layout missing; using equal-width fallback"
    return info


def parse_lane_ranges(lane_ranges_text: str, frame_width: int) -> Dict[int, tuple[int, int]]:
    ranges: Dict[int, tuple[int, int]] = {}
    if not lane_ranges_text:
        return ranges
    for raw_item in str(lane_ranges_text).split(","):
        item = raw_item.strip()
        if not item or ":" not in item or "-" not in item:
            continue
        lane_text, span_text = item.split(":", 1)
        start_text, end_text = span_text.split("-", 1)
        try:
            lane = int(lane_text.strip())
            x1 = int(float(start_text.strip()))
            x2 = int(float(end_text.strip()))
        except ValueError:
            continue
        x1 = max(0, min(frame_width - 1, x1))
        x2 = max(x1 + 1, min(frame_width, x2))
        ranges[lane] = (x1, x2)
    return ranges


def build_lane_shapes(
    frame_width: int,
    frame_height: int,
    target_lanes: List[int],
    lane_ranges_text: str = "",
    lane_polygons_text: str = "",
    lane_layout_file: str = "",
) -> List[Dict[str, Any]]:
    if frame_width <= 0 or frame_height <= 0 or not target_lanes:
        return []

    polygons = load_lane_polygons_from_file(lane_layout_file, frame_width, frame_height)
    if not polygons:
        polygons = parse_lane_polygons(lane_polygons_text, frame_width, frame_height)
    if polygons:
        shapes = []
        for lane in target_lanes:
            points = polygons.get(lane)
            if points:
                xs = [p[0] for p in points]
                shapes.append(
                    {
                        "lane": lane,
                        "kind": "polygon",
                        "points": points,
                        "x1": min(xs),
                        "x2": max(xs) + 1,
                    }
                )
        if shapes:
            return shapes

    parsed = parse_lane_ranges(lane_ranges_text, frame_width)
    if parsed:
        shapes = []
        for lane in target_lanes:
            if lane in parsed:
                x1, x2 = parsed[lane]
                shapes.append(
                    {
                        "lane": lane,
                        "kind": "segment",
                        "x1": x1,
                        "x2": x2,
                        "points": [(x1, 0), (x2 - 1, 0), (x2 - 1, frame_height - 1), (x1, frame_height - 1)],
                    }
                )
        if shapes:
            return shapes

    width_per_lane = frame_width / max(len(target_lanes), 1)
    shapes: List[Dict[str, Any]] = []
    for idx, lane in enumerate(target_lanes):
        x1 = int(round(idx * width_per_lane))
        x2 = int(round((idx + 1) * width_per_lane))
        x1 = max(0, min(frame_width - 1, x1))
        x2 = max(x1 + 1, min(frame_width, x2))
        shapes.append(
            {
                "lane": lane,
                "kind": "segment",
                "x1": x1,
                "x2": x2,
                "points": [(x1, 0), (x2 - 1, 0), (x2 - 1, frame_height - 1), (x1, frame_height - 1)],
            }
        )
    return shapes


def _point_in_polygon(x: float, y: float, polygon: List[Tuple[int, int]]) -> bool:
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        denom = float(yj - yi)
        if abs(denom) < 1e-6:
            denom = 1e-6 if denom >= 0 else -1e-6
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / denom + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def resolve_lane_by_point(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
    target_lanes: List[int],
    lane_ranges_text: str = "",
    lane_polygons_text: str = "",
    lane_layout_file: str = "",
) -> Optional[int]:
    shapes = build_lane_shapes(
        frame_width=frame_width,
        frame_height=frame_height,
        target_lanes=target_lanes,
        lane_ranges_text=lane_ranges_text,
        lane_polygons_text=lane_polygons_text,
        lane_layout_file=lane_layout_file,
    )
    if not shapes:
        return None
    for shape in shapes:
        if shape["kind"] == "polygon" and _point_in_polygon(x, y, shape["points"]):
            return int(shape["lane"])
        if shape["kind"] == "segment" and shape["x1"] <= x < shape["x2"]:
            return int(shape["lane"])
    if x < shapes[0]["x1"]:
        return int(shapes[0]["lane"])
    return int(shapes[-1]["lane"])


def resolve_lane_by_center_x(
    center_x: float,
    frame_width: int,
    target_lanes: List[int],
    lane_ranges_text: str = "",
) -> Optional[int]:
    return resolve_lane_by_point(
        x=center_x,
        y=0,
        frame_width=frame_width,
        frame_height=1,
        target_lanes=target_lanes,
        lane_ranges_text=lane_ranges_text,
    )
