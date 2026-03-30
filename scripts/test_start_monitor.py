import argparse
import time
from pathlib import Path
import sys

import cv2
import numpy as np
from viz_common import (
    draw_horizontal_line,
    draw_left_top_text,
    draw_right_top_text,
    draw_toe_proxy_points,
    load_env_config,
    parse_bgr,
    pick_float_multi,
    pick_int_multi,
    pick_text_multi,
    scale_line_y,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def process_video(
    model,
    video_path: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    mirror: bool,
    show: bool,
    output_path: Path | None = None,
    false_start_test: bool = False,
    countdown_at: float = 21.0,
    countdown_sec: float = 3.0,
    style: dict | None = None,
) -> None:
    style = {
        "line_color": (0, 0, 255),
        "ready_color": (0, 255, 0),
        "alert_color": (0, 0, 255),
        "countdown_color": (0, 255, 255),
        "box_color": (0, 255, 0),
        "toe_ankle_color": (255, 0, 0),
        "toe_color": (0, 255, 255),
        "font_scale": 1.0,
        "font_thickness": 2,
        "line_thickness": 2,
        "box_thickness": 1,
        "toe_ankle_radius": 3,
        "toe_radius": 4,
        "toe_link_thickness": 1,
        **(style or {}),
    }
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 640
    cap.release()

    settings = None
    target_w, target_h = width, height
    if false_start_test:
        from edge.app.core.config import get_settings

        settings = get_settings()
        target_w = settings.capture_width
        target_h = settings.capture_height

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path if output_path else output_dir / f"{video_path.stem}_res.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (target_w, target_h),
    )

    results = model(
        source=str(video_path),
        stream=True,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )

    state = None
    algo = None
    last_false_start = None
    last_false_start_ts = None
    countdown_sent = False
    frame_idx = 0
    scale_x = target_w / width if width else 1.0
    scale_y = target_h / height if height else 1.0
    if false_start_test:
        from edge.app.core.state import NodeState, NodePhase
        from edge.app.services.algorithms.rules import toe_proxy_points_from_keypoints
        from edge.app.services.algorithms.violation import ViolationAlgo, extract_ultralytics_dets

        state = NodeState(node_id=1, session_id="LOCAL_TEST", phase=NodePhase.MONITORING)
        state.config = {
            "ready_ts": 0,
            "false_start_check": True,
            "tracking_active": True,
            "lane_count": 1,
        }
        algo = ViolationAlgo(state, load_model=False)
        line_y = scale_line_y(settings.start_line_y, target_h)
        kps_thres = settings.kps_conf_thres

    for r in results:
        plotted = r.plot()
        if false_start_test and (plotted.shape[1] != target_w or plotted.shape[0] != target_h):
            plotted = cv2.resize(plotted, (target_w, target_h))
        if false_start_test and state is not None and algo is not None:
            current_ms = int(frame_idx / max(fps, 1) * 1000)
            if not countdown_sent and current_ms >= int(countdown_at * 1000):
                state.expected_start_time = current_ms + int(countdown_sec * 1000)
                countdown_sent = True

            dets = extract_ultralytics_dets(r)
            if scale_x != 1.0 or scale_y != 1.0:
                for det in dets:
                    bbox = det.get("bbox")
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        det["bbox"] = [
                            bbox[0] * scale_x,
                            bbox[1] * scale_y,
                            bbox[2] * scale_x,
                            bbox[3] * scale_y,
                        ]
                    kps_det = det.get("keypoints")
                    if kps_det and len(kps_det) >= 17:
                        scaled = []
                        for x, y, s in kps_det:
                            scaled.append([x * scale_x, y * scale_y, s])
                        det["keypoints"] = scaled
            # Always draw toe proxy debug during false-start test
            for det in dets:
                bbox = det.get("bbox")
                if isinstance(bbox, list) and len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(plotted, (x1, y1), (x2, y2), style["box_color"], style["box_thickness"])
                toe_points = toe_proxy_points_from_keypoints(
                    keypoints=det.get("keypoints"),
                    conf_thres=algo.settings.kps_conf_thres,
                    toe_scale=algo.settings.toe_proxy_scale,
                )
                draw_toe_proxy_points(
                    plotted,
                    toe_points,
                    ankle_color=style["toe_ankle_color"],
                    toe_color=style["toe_color"],
                    ankle_radius=style["toe_ankle_radius"],
                    toe_radius=style["toe_radius"],
                    link_thickness=style["toe_link_thickness"],
                )
            kps = [d.get("keypoints") for d in dets]
            events = algo.process_frame_logic(
                frame=np.zeros((target_h, target_w, 3), dtype=np.uint8),
                track_ids=list(range(len(dets))),
                boxes=dets,
                kps=kps,
                current_time=current_ms,
            )
            for ev in events:
                if ev.get("event") == "FALSE_START":
                    last_false_start = ev
                    last_false_start_ts = current_ms

            # Draw start line
            draw_horizontal_line(
                plotted,
                line_y,
                style["line_color"],
                thickness=style["line_thickness"],
            )

            # HUD: left-top status, right-top countdown
            status_text = "READY"
            status_color = style["ready_color"]
            if countdown_sent and state.expected_start_time is not None and current_ms < state.expected_start_time:
                remain_ms = max(0, int(state.expected_start_time - current_ms))
                countdown_num = max(1, int(np.ceil(remain_ms / 1000.0)))
                countdown_text = f"COUNTDOWN: {countdown_num}"
                draw_right_top_text(
                    plotted,
                    countdown_text,
                    style["countdown_color"],
                    font_scale=style["font_scale"],
                    thickness=style["font_thickness"],
                )

            # Draw false start overlay for 2 seconds
            if last_false_start and last_false_start_ts is not None:
                if current_ms - last_false_start_ts <= 2000:
                    status_text = "FALSE START"
                    status_color = style["alert_color"]
                    bbox = last_false_start.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        cv2.rectangle(
                            plotted,
                            (x1, y1),
                            (x2, y2),
                            style["alert_color"],
                            style["line_thickness"],
                        )
                    kps_ev = last_false_start.get("keypoints")
                    if kps_ev and len(kps_ev) >= 17:
                        for idx in (15, 16):
                            try:
                                x, y, s = kps_ev[idx]
                            except Exception:
                                continue
                            if s is not None and s >= kps_thres:
                                cv2.circle(
                                    plotted,
                                    (int(x), int(y)),
                                    style["toe_radius"],
                                    style["alert_color"],
                                    -1,
                                )
                    toe_points = last_false_start.get("toe_proxy_points")
                    draw_toe_proxy_points(
                        plotted,
                        toe_points,
                        ankle_color=style["toe_ankle_color"],
                        toe_color=style["toe_color"],
                        ankle_radius=style["toe_ankle_radius"],
                        toe_radius=style["toe_radius"],
                        link_thickness=style["toe_link_thickness"],
                    )
                    lane = last_false_start.get("lane")
                    label = f"FALSE START" if lane is None else f"FALSE START L{lane}"
                    draw_left_top_text(
                        plotted,
                        label,
                        style["alert_color"],
                        line=1,
                        font_scale=style["font_scale"],
                        thickness=style["font_thickness"],
                    )
            draw_left_top_text(
                plotted,
                status_text,
                status_color,
                line=0,
                font_scale=style["font_scale"],
                thickness=style["font_thickness"],
            )

        if mirror:
            plotted = cv2.flip(plotted, 1)
        writer.write(plotted)
        if show:
            cv2.imshow("Ultralytics YOLO Batch", plotted)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start monitoring test (false start + 3-2-1 countdown)")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--model", default="data/models/yolo26n-pose.pt", help="YOLO .pt model path under data/models")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--input-dir", default="data/samples", help="Batch input video folder (recursive)")
    parser.add_argument("--output-dir", default="data/sample_res", help="Batch output folder")
    parser.add_argument("--input", default="", help="Single video path (overrides input-dir)")
    parser.add_argument("--output", default="", help="Output file path for single video")
    parser.add_argument("--mirror", action="store_true", help="Enable mirror display")
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    parser.add_argument("--false-start-test", action="store_true", help="Enable false start test logic")
    parser.add_argument("--countdown-at", type=float, default=21.0, help="Countdown start time (sec)")
    parser.add_argument("--countdown-sec", type=float, default=3.0, help="Countdown duration (sec)")
    parser.add_argument("--line-color", default=None, help="Start line color in B,G,R")
    parser.add_argument("--ready-color", default=None, help="READY text color in B,G,R")
    parser.add_argument("--alert-color", default=None, help="Alert text/box color in B,G,R")
    parser.add_argument("--countdown-color", default=None, help="Countdown text color in B,G,R")
    parser.add_argument("--box-color", default=None, help="Tracking debug box color in B,G,R")
    parser.add_argument("--toe-ankle-color", default=None, help="Ankle point color in B,G,R")
    parser.add_argument("--toe-color", default=None, help="Toe proxy point/link color in B,G,R")
    parser.add_argument("--hud-font-scale", type=float, default=None, help="HUD font scale")
    parser.add_argument("--hud-font-thickness", type=int, default=None, help="HUD font thickness")
    parser.add_argument("--line-thickness", type=int, default=None, help="Start line thickness")
    parser.add_argument("--box-thickness", type=int, default=None, help="Debug bbox thickness")
    parser.add_argument("--toe-ankle-radius", type=int, default=None, help="Ankle point radius")
    parser.add_argument("--toe-radius", type=int, default=None, help="Toe proxy point radius")
    parser.add_argument("--toe-link-thickness", type=int, default=None, help="Toe link thickness")
    args = parser.parse_args()
    if not args.false_start_test:
        args.false_start_test = True

    env_cfg = load_env_config()
    style = {
        "line_color": parse_bgr(
            pick_text_multi(args.line_color, env_cfg, ["VIZ_LINE_COLOR", "START_LINE_COLOR"], "0,0,255"),
            (0, 0, 255),
        ),
        "ready_color": parse_bgr(
            pick_text_multi(args.ready_color, env_cfg, ["VIZ_READY_COLOR", "START_READY_COLOR"], "0,255,0"),
            (0, 255, 0),
        ),
        "alert_color": parse_bgr(
            pick_text_multi(args.alert_color, env_cfg, ["VIZ_ALERT_COLOR", "START_ALERT_COLOR"], "0,0,255"),
            (0, 0, 255),
        ),
        "countdown_color": parse_bgr(
            pick_text_multi(
                args.countdown_color,
                env_cfg,
                ["VIZ_COUNTDOWN_COLOR", "START_COUNTDOWN_COLOR"],
                "0,255,255",
            ),
            (0, 255, 255),
        ),
        "box_color": parse_bgr(
            pick_text_multi(args.box_color, env_cfg, ["VIZ_BOX_COLOR", "START_BOX_COLOR"], "0,255,0"),
            (0, 255, 0),
        ),
        "toe_ankle_color": parse_bgr(
            pick_text_multi(
                args.toe_ankle_color,
                env_cfg,
                ["VIZ_TOE_ANKLE_COLOR", "START_TOE_ANKLE_COLOR"],
                "255,0,0",
            ),
            (255, 0, 0),
        ),
        "toe_color": parse_bgr(
            pick_text_multi(args.toe_color, env_cfg, ["VIZ_TOE_COLOR", "START_TOE_COLOR"], "0,255,255"),
            (0, 255, 255),
        ),
        "font_scale": pick_float_multi(
            args.hud_font_scale, env_cfg, ["VIZ_HUD_FONT_SCALE", "START_HUD_FONT_SCALE"], 1.0
        ),
        "font_thickness": pick_int_multi(
            args.hud_font_thickness,
            env_cfg,
            ["VIZ_HUD_FONT_THICKNESS", "START_HUD_FONT_THICKNESS"],
            2,
        ),
        "line_thickness": pick_int_multi(
            args.line_thickness,
            env_cfg,
            ["VIZ_LINE_THICKNESS", "START_LINE_THICKNESS"],
            2,
        ),
        "box_thickness": pick_int_multi(
            args.box_thickness,
            env_cfg,
            ["VIZ_BOX_THICKNESS", "START_BOX_THICKNESS"],
            1,
        ),
        "toe_ankle_radius": pick_int_multi(
            args.toe_ankle_radius,
            env_cfg,
            ["VIZ_TOE_ANKLE_RADIUS", "START_TOE_ANKLE_RADIUS"],
            3,
        ),
        "toe_radius": pick_int_multi(
            args.toe_radius,
            env_cfg,
            ["VIZ_TOE_RADIUS", "START_TOE_RADIUS"],
            4,
        ),
        "toe_link_thickness": pick_int_multi(
            args.toe_link_thickness,
            env_cfg,
            ["VIZ_TOE_LINK_THICKNESS", "START_TOE_LINK_THICKNESS"],
            1,
        ),
    }

    from ultralytics import YOLO

    model_path = Path(args.model)
    if not model_path.exists():
        candidates = sorted(Path("data/models").glob("*.pt"))
        if not candidates:
            raise FileNotFoundError("No .pt model found in data/models; pass --model explicitly.")
        model_path = candidates[0]

    model = YOLO(str(model_path))

    mirror = args.mirror
    show = not args.no_show

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        output_path = Path(args.output) if args.output else None
        process_video(
            model,
            input_path,
            Path(args.output_dir),
            args.conf,
            args.iou,
            args.imgsz,
            mirror,
            show,
            output_path,
            false_start_test=args.false_start_test,
            countdown_at=args.countdown_at,
            countdown_sec=args.countdown_sec,
            style=style,
        )
        cv2.destroyAllWindows()
        return

    input_dir = Path(args.input_dir)
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    batch_files = []
    if input_dir.exists():
        batch_files = [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in video_exts]

    if batch_files:
        for video_path in batch_files:
            rel_parent = video_path.parent.relative_to(input_dir)
            out_dir = Path(args.output_dir) / rel_parent
            process_video(
                model,
                video_path,
                out_dir,
                args.conf,
                args.iou,
                args.imgsz,
                mirror,
                show,
                style=style,
            )
    else:
        source = int(args.source) if str(args.source).isdigit() else args.source
        results = model(
            source=source,
            stream=True,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )
        last = time.time()
        fps = 0.0
        for r in results:
            plotted = r.plot()
            now = time.time()
            dt = now - last
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last = now
            draw_left_top_text(
                plotted,
                f"FPS: {fps:.1f}",
                style["ready_color"],
                line=0,
                font_scale=style["font_scale"],
                thickness=style["font_thickness"],
            )
            if mirror:
                plotted = cv2.flip(plotted, 1)
            if show:
                cv2.imshow("Ultralytics YOLO Realtime", plotted)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

