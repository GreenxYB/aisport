import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate lane polygons by clicking points on a frame.")
    parser.add_argument("--source", default="0", help="Camera index, video path, or image path")
    parser.add_argument("--output", default="configs/lane_layout_4lane.json", help="Output JSON path")
    parser.add_argument("--lanes", type=int, default=4, help="Number of lanes to calibrate")
    parser.add_argument("--width", type=int, default=1280, help="Calibration frame width")
    parser.add_argument("--height", type=int, default=640, help="Calibration frame height")
    return parser.parse_args()


def open_source(source: str):
    path = Path(source)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        return cap, None
    if path.exists() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        image = cv2.imread(str(path))
        return None, image
    cap = cv2.VideoCapture(str(source))
    return cap, None


def acquire_frame(source: str, width: int, height: int) -> np.ndarray:
    cap, image = open_source(source)
    if image is not None:
        return cv2.resize(image, (width, height))
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame from source: {source}")
    return cv2.resize(frame, (width, height))


def draw_overlay(base: np.ndarray, lanes: list[dict], current_lane: int, current_points: list[tuple[int, int]]):
    frame = base.copy()
    cv2.putText(
        frame,
        f"Lane {current_lane}: click 4 points clockwise | n=next r=reset u=undo s=save q=quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for lane_item in lanes:
        pts = np.array(lane_item["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        x, y = lane_item["points"][0]
        cv2.putText(frame, f"L{lane_item['lane']}", (x + 6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for idx, point in enumerate(current_points):
        cv2.circle(frame, point, 4, (0, 0, 255), -1)
        cv2.putText(frame, str(idx + 1), (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if len(current_points) >= 2:
        pts = np.array(current_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=1)
    return frame


def main():
    args = parse_args()
    frame = acquire_frame(args.source, args.width, args.height)

    lanes: list[dict] = []
    current_points: list[tuple[int, int]] = []
    state = {"lane": 1}

    def on_click(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["lane"] > args.lanes:
            return
        if len(current_points) >= 4:
            return
        current_points.append((int(x), int(y)))

    window = "Lane Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.width, args.height)
    cv2.setMouseCallback(window, on_click)

    while True:
        preview = draw_overlay(frame, lanes, state["lane"], current_points)
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            break
        if key == ord("u") and current_points:
            current_points.pop()
        if key == ord("r"):
            current_points.clear()
        if key == ord("n"):
            if len(current_points) >= 3:
                lanes.append({"lane": state["lane"], "points": current_points.copy()})
                current_points.clear()
                state["lane"] += 1
        if len(current_points) == 4:
            lanes.append({"lane": state["lane"], "points": current_points.copy()})
            current_points.clear()
            state["lane"] += 1
        if key == ord("s"):
            break

    cv2.destroyAllWindows()

    payload = {
        "frame_width": args.width,
        "frame_height": args.height,
        "lanes": lanes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] saved lane layout -> {output}")


if __name__ == "__main__":
    main()
