from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _resolve_path(line_file: str) -> Optional[Path]:
    if not line_file:
        return None
    path = Path(str(line_file))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _clamp_point(x: float, y: float, frame_width: int, frame_height: int) -> Tuple[int, int]:
    xi = max(0, min(frame_width - 1, int(round(x))))
    yi = max(0, min(frame_height - 1, int(round(y))))
    return xi, yi


def _load_payload(line_file: str) -> Optional[Dict[str, Any]]:
    path = _resolve_path(line_file)
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_line_definition(
    *,
    frame_width: int,
    frame_height: int,
    line_file: str = "",
    fallback_y: int,
    line_name: str,
) -> Dict[str, Any]:
    payload = _load_payload(line_file)
    if payload is not None:
        p1 = payload.get("p1")
        p2 = payload.get("p2")
        src_width = int(payload.get("frame_width") or frame_width or 1)
        src_height = int(payload.get("frame_height") or frame_height or 1)
        if isinstance(p1, (list, tuple)) and len(p1) >= 2 and isinstance(p2, (list, tuple)) and len(p2) >= 2:
            scale_x = frame_width / max(src_width, 1)
            scale_y = frame_height / max(src_height, 1)
            sp1 = _clamp_point(float(p1[0]) * scale_x, float(p1[1]) * scale_y, frame_width, frame_height)
            sp2 = _clamp_point(float(p2[0]) * scale_x, float(p2[1]) * scale_y, frame_width, frame_height)
            return {
                "name": line_name,
                "source": "file",
                "file": str(_resolve_path(line_file)),
                "runtime_frame": [frame_width, frame_height],
                "file_frame": [src_width, src_height],
                "p1": [sp1[0], sp1[1]],
                "p2": [sp2[0], sp2[1]],
            }

    y = int(fallback_y * frame_height / 640) if frame_height > 0 else int(fallback_y)
    return {
        "name": line_name,
        "source": "fallback_y",
        "file": str(_resolve_path(line_file)) if line_file else None,
        "runtime_frame": [frame_width, frame_height],
        "file_frame": None,
        "p1": [0, y],
        "p2": [max(frame_width - 1, 0), y],
    }


def inspect_line_definition(
    *,
    frame_width: int,
    frame_height: int,
    line_file: str = "",
    fallback_y: int,
    line_name: str,
) -> Dict[str, Any]:
    info = {
        "name": line_name,
        "source": "fallback_y",
        "available": True,
        "calibrated": False,
        "warning": None,
        "file": None,
    }
    path = _resolve_path(line_file)
    if path is not None:
        info["file"] = str(path)
        if not path.exists():
            info["warning"] = f"{line_name} calibration file not found: {path}"
            return info
        payload = _load_payload(str(path))
        if payload is None or not isinstance(payload.get("p1"), (list, tuple)) or not isinstance(payload.get("p2"), (list, tuple)):
            info["warning"] = f"{line_name} calibration file invalid: {path}"
            return info
        info["source"] = "file"
        info["calibrated"] = True
        info["runtime_frame"] = [frame_width, frame_height]
        info["file_frame"] = [payload.get("frame_width"), payload.get("frame_height")]
    return info


def line_y_at_x(line: Dict[str, Any], x: float) -> float:
    p1 = line.get("p1") or [0, 0]
    p2 = line.get("p2") or [0, 0]
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    if abs(x2 - x1) < 1e-6:
        return max(y1, y2)
    t = (float(x) - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def point_crossed_line(point: Tuple[float, float] | list[float], line: Dict[str, Any]) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return False
    x, y = float(point[0]), float(point[1])
    return y >= line_y_at_x(line, x)
