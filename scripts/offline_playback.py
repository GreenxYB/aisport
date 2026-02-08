import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edge.app.core.state import NodeState  # noqa: E402
from edge.app.services.algorithms.runner import AlgorithmRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline video playback for algorithm runner")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--session", default="OFFLINE_SESSION", help="Session id")
    parser.add_argument("--node", type=int, default=1, help="Node id")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    state = NodeState(node_id=args.node, session_id=args.session)
    runner = AlgorithmRunner(state)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Failed to open video")
        return

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        runner.process_frame(frame, ts_ms)
        frame_count += 1

    cap.release()
    print(f"offline playback done, frames={frame_count}, algo_events={state.algo_events_generated}")


if __name__ == "__main__":
    main()
