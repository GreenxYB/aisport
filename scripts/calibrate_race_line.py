import argparse
import json
from pathlib import Path

import cv2


POINTS = []


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a single start/finish line by clicking two points.")
    parser.add_argument("--source", default="0", help="Camera index, video file, or RTSP url")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open source: {args.source}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise SystemExit("Failed to read frame from source")
    frame = cv2.resize(frame, (args.width, args.height))

    window = "Calibrate Race Line"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(POINTS) < 2:
            POINTS.append([int(x), int(y)])

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        vis = frame.copy()
        for idx, point in enumerate(POINTS):
            cv2.circle(vis, tuple(point), 4, (0, 255, 255), -1)
            cv2.putText(vis, f"P{idx+1}", (point[0] + 6, point[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(POINTS) == 2:
            cv2.line(vis, tuple(POINTS[0]), tuple(POINTS[1]), (0, 0, 255), 2)

        cv2.putText(vis, "Left click 2 points, 'u' undo, 's' save, 'q' quit", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and POINTS:
            POINTS.pop()
        elif key == ord("s") and len(POINTS) == 2:
            payload = {
                "frame_width": args.width,
                "frame_height": args.height,
                "p1": POINTS[0],
                "p2": POINTS[1],
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            break
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
