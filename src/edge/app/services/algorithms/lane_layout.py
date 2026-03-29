from __future__ import annotations

from typing import Any, Dict, List, Optional


def binding_target_lanes(bindings: List[Dict[str, Any]], lane_count: int) -> List[int]:
    lanes = [
        int(item.get("lane"))
        for item in bindings
        if isinstance(item, dict) and isinstance(item.get("lane"), int)
    ]
    if lanes:
        return sorted(dict.fromkeys(lanes))
    return list(range(1, max(int(lane_count), 0) + 1))


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


def build_lane_segments(
    frame_width: int,
    target_lanes: List[int],
    lane_ranges_text: str = "",
) -> List[Dict[str, int]]:
    if frame_width <= 0 or not target_lanes:
        return []

    parsed = parse_lane_ranges(lane_ranges_text, frame_width)
    if parsed:
        segments = []
        for lane in target_lanes:
            if lane in parsed:
                x1, x2 = parsed[lane]
                segments.append({"lane": lane, "x1": x1, "x2": x2})
        if segments:
            return segments

    width_per_lane = frame_width / max(len(target_lanes), 1)
    segments: List[Dict[str, int]] = []
    for idx, lane in enumerate(target_lanes):
        x1 = int(round(idx * width_per_lane))
        x2 = int(round((idx + 1) * width_per_lane))
        x1 = max(0, min(frame_width - 1, x1))
        x2 = max(x1 + 1, min(frame_width, x2))
        segments.append({"lane": lane, "x1": x1, "x2": x2})
    return segments


def resolve_lane_by_center_x(
    center_x: float,
    frame_width: int,
    target_lanes: List[int],
    lane_ranges_text: str = "",
) -> Optional[int]:
    segments = build_lane_segments(frame_width, target_lanes, lane_ranges_text=lane_ranges_text)
    if not segments:
        return None
    cx = float(center_x)
    for segment in segments:
        if segment["x1"] <= cx < segment["x2"]:
            return int(segment["lane"])
    if cx < segments[0]["x1"]:
        return int(segments[0]["lane"])
    return int(segments[-1]["lane"])
