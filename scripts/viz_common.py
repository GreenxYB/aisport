import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2


def scale_line_y(y_on_640: int, frame_h: int) -> int:
    if frame_h <= 0:
        return int(y_on_640)
    return int(y_on_640 * frame_h / 640)


def parse_bgr(color_text: str, default: Tuple[int, int, int]) -> Tuple[int, int, int]:
    try:
        parts = [int(x.strip()) for x in str(color_text).split(",")]
        if len(parts) != 3:
            return default
        return (
            max(0, min(255, parts[0])),
            max(0, min(255, parts[1])),
            max(0, min(255, parts[2])),
        )
    except Exception:
        return default


def load_env_config(paths: Iterable[Path] | None = None) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    candidates = list(paths) if paths is not None else [Path(".env.edge"), Path("configs/edge.example.env")]
    for p in candidates:
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            cfg[key.strip()] = value.strip()
    # OS env has highest priority among env sources
    for key, value in os.environ.items():
        cfg[key] = value
    return cfg


def pick_text(cli_value: str | None, env_cfg: Dict[str, str], env_key: str, fallback: str) -> str:
    if cli_value is not None and str(cli_value).strip() != "":
        return str(cli_value).strip()
    val = env_cfg.get(env_key)
    if val is None or str(val).strip() == "":
        return fallback
    return str(val).strip()


def pick_text_multi(
    cli_value: str | None,
    env_cfg: Dict[str, str],
    env_keys: List[str],
    fallback: str,
) -> str:
    if cli_value is not None and str(cli_value).strip() != "":
        return str(cli_value).strip()
    for key in env_keys:
        val = env_cfg.get(key)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return fallback


def pick_float(cli_value: float | None, env_cfg: Dict[str, str], env_key: str, fallback: float) -> float:
    if cli_value is not None:
        return float(cli_value)
    val = env_cfg.get(env_key)
    if val is None or str(val).strip() == "":
        return float(fallback)
    try:
        return float(val)
    except Exception:
        return float(fallback)


def pick_float_multi(
    cli_value: float | None,
    env_cfg: Dict[str, str],
    env_keys: List[str],
    fallback: float,
) -> float:
    if cli_value is not None:
        return float(cli_value)
    for key in env_keys:
        val = env_cfg.get(key)
        if val is None or str(val).strip() == "":
            continue
        try:
            return float(val)
        except Exception:
            continue
    return float(fallback)


def pick_int(cli_value: int | None, env_cfg: Dict[str, str], env_key: str, fallback: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    val = env_cfg.get(env_key)
    if val is None or str(val).strip() == "":
        return int(fallback)
    try:
        return int(float(val))
    except Exception:
        return int(fallback)


def pick_int_multi(
    cli_value: int | None,
    env_cfg: Dict[str, str],
    env_keys: List[str],
    fallback: int,
) -> int:
    if cli_value is not None:
        return int(cli_value)
    for key in env_keys:
        val = env_cfg.get(key)
        if val is None or str(val).strip() == "":
            continue
        try:
            return int(float(val))
        except Exception:
            continue
    return int(fallback)


def draw_horizontal_line(
    frame,
    line_y: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    cv2.line(frame, (0, int(line_y)), (frame.shape[1] - 1, int(line_y)), color, thickness)


def draw_left_top_text(
    frame,
    text: str,
    color: Tuple[int, int, int],
    line: int = 0,
    font_scale: float = 1.0,
    thickness: int = 2,
    base_y: int = 30,
    line_gap: int = 30,
) -> None:
    y = base_y + line * line_gap
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_right_top_text(
    frame,
    text: str,
    color: Tuple[int, int, int],
    margin_x: int = 10,
    y: int = 30,
    font_scale: float = 1.0,
    thickness: int = 2,
) -> None:
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(10, frame.shape[1] - tw - margin_x)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_toe_proxy_points(
    frame,
    toe_points: Any,
    ankle_color: Tuple[int, int, int] = (255, 0, 0),
    toe_color: Tuple[int, int, int] = (0, 255, 255),
    ankle_radius: int = 3,
    toe_radius: int = 4,
    link_thickness: int = 1,
) -> None:
    if not isinstance(toe_points, list):
        return
    for point in toe_points:
        ankle = point.get("ankle")
        toe = point.get("toe")
        if isinstance(ankle, list) and len(ankle) >= 2:
            cv2.circle(frame, (int(ankle[0]), int(ankle[1])), ankle_radius, ankle_color, -1)
        if isinstance(toe, list) and len(toe) >= 2:
            cv2.circle(frame, (int(toe[0]), int(toe[1])), toe_radius, toe_color, -1)
        if (
            isinstance(ankle, list)
            and len(ankle) >= 2
            and isinstance(toe, list)
            and len(toe) >= 2
        ):
            cv2.line(
                frame,
                (int(ankle[0]), int(ankle[1])),
                (int(toe[0]), int(toe[1])),
                toe_color,
                link_thickness,
            )
