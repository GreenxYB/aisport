import argparse
from pathlib import Path
import sys

import cv2
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
    output_path: Path,
    conf: float,
    iou: float,
    imgsz: int,
    monitor_start_at: float,
    finish_line_y: int,
    tracker: str,
    show: bool,
    mirror: bool,
    style: dict | None = None,
) -> None:
    style = {
        "line_color": (255, 0, 255),
        "ready_color": (0, 255, 0),
        "running_color": (0, 255, 255),
        "timer_color": (0, 255, 255),
        "id_color": (255, 255, 0),
        "rank_color": (0, 255, 0),
        "toe_ankle_color": (255, 0, 0),
        "toe_color": (0, 255, 255),
        "font_scale": 1.0,
        "font_thickness": 2,
        "id_font_scale": 0.6,
        "id_font_thickness": 2,
        "rank_font_scale": 0.7,
        "rank_font_thickness": 2,
        "line_thickness": 2,
        "toe_ankle_radius": 3,
        "toe_radius": 4,
        "toe_link_thickness": 1,
        **(style or {}),
    }
    from edge.app.core.state import NodePhase, NodeState
    from edge.app.services.algorithms.finish_line import FinishLineAlgo
    from edge.app.services.algorithms.rules import toe_proxy_points_from_keypoints
    from edge.app.services.algorithms.violation import extract_ultralytics_dets

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 640
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    monitor_started = False
    start_ms = int(monitor_start_at * 1000)
    finish_events: dict[int, dict] = {}

    state = NodeState(node_id=1, session_id="FINISH_TEST", phase=NodePhase.MONITORING)
    state.expected_start_time = start_ms
    finish_algo = FinishLineAlgo(state)

    results = model.track(
        source=str(video_path),
        stream=True,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        persist=True,
        tracker=tracker,
        verbose=False,
    )

    for r in results:
        plotted = r.plot()
        if plotted.shape[1] != width or plotted.shape[0] != height:
            plotted = cv2.resize(plotted, (width, height))

        current_ms = int(frame_idx / max(fps, 1) * 1000)
        if not monitor_started and current_ms >= start_ms:
            monitor_started = True
            state.expected_start_time = current_ms

        line_y = scale_line_y(finish_line_y, plotted.shape[0])
        draw_horizontal_line(
            plotted,
            line_y,
            style["line_color"],
            thickness=style["line_thickness"],
        )

        dets = extract_ultralytics_dets(r)
        if r.boxes is not None and getattr(r.boxes, "id", None) is not None:
            track_ids = [int(x) for x in r.boxes.id.int().cpu().tolist()]
        else:
            track_ids = list(range(len(dets)))

        for idx, det in enumerate(dets):
            track_id = track_ids[idx] if idx < len(track_ids) else idx
            bbox = det.get("bbox")
            keypoints = det.get("keypoints")
            toe_points = toe_proxy_points_from_keypoints(
                keypoints=keypoints,
                conf_thres=finish_algo.settings.kps_conf_thres,
                toe_scale=finish_algo.settings.toe_proxy_scale,
            )

            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.putText(
                    plotted,
                    f"ID {track_id}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    style["id_font_scale"],
                    style["id_color"],
                    style["id_font_thickness"],
                    cv2.LINE_AA,
                )
                if track_id in finish_events:
                    cv2.putText(
                        plotted,
                        f"FINISH #{finish_events[track_id]['rank']}",
                        (x1, min(plotted.shape[0] - 10, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        style["rank_font_scale"],
                        style["rank_color"],
                        style["rank_font_thickness"],
                        cv2.LINE_AA,
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

        finish_report = finish_algo.process_detections(
            dets=dets,
            track_ids=track_ids,
            ts_ms=current_ms,
            frame_shape=(plotted.shape[0], plotted.shape[1]),
            line_y_override=line_y,
        )
        if finish_report:
            for item in finish_report.get("results", []):
                track_id = int(item.get("track_id", -1))
                if track_id < 0:
                    continue
                finish_events[track_id] = item

        status_text = "READY" if not monitor_started else f"RUNNING  FINISHED:{len(finish_events)}"
        status_color = style["ready_color"] if not monitor_started else style["running_color"]
        draw_left_top_text(
            plotted,
            status_text,
            status_color,
            line=0,
            font_scale=style["font_scale"],
            thickness=style["font_thickness"],
        )

        if monitor_started:
            elapsed = max(0.0, (current_ms - start_ms) / 1000.0)
            right_text = f"T+{elapsed:.1f}s"
            draw_right_top_text(
                plotted,
                right_text,
                style["timer_color"],
                font_scale=style["font_scale"],
                thickness=style["font_thickness"],
            )

        if mirror:
            plotted = cv2.flip(plotted, 1)
        writer.write(plotted)
        if show:
            cv2.imshow("Finish Line Test", plotted)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    writer.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish line monitoring test script")
    parser.add_argument("--model", default="data/models/yolo26n-pose.pt", help="YOLO .pt model path")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--monitor-start-at", type=float, default=0.0, help="Monitoring start time (sec)")
    parser.add_argument("--finish-line-y", type=int, default=520, help="Finish line y on 640-height frame")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config")
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    parser.add_argument("--mirror", action="store_true", help="Enable mirror display")
    parser.add_argument("--line-color", default=None, help="Finish line color in B,G,R")
    parser.add_argument("--ready-color", default=None, help="READY text color in B,G,R")
    parser.add_argument("--running-color", default=None, help="RUNNING text color in B,G,R")
    parser.add_argument("--timer-color", default=None, help="Timer text color in B,G,R")
    parser.add_argument("--id-color", default=None, help="Track ID text color in B,G,R")
    parser.add_argument("--rank-color", default=None, help="Finish rank text color in B,G,R")
    parser.add_argument("--toe-ankle-color", default=None, help="Ankle point color in B,G,R")
    parser.add_argument("--toe-color", default=None, help="Toe proxy point color in B,G,R")
    parser.add_argument("--hud-font-scale", type=float, default=None, help="HUD font scale")
    parser.add_argument("--hud-font-thickness", type=int, default=None, help="HUD font thickness")
    parser.add_argument("--id-font-scale", type=float, default=None, help="ID label font scale")
    parser.add_argument("--id-font-thickness", type=int, default=None, help="ID label font thickness")
    parser.add_argument("--rank-font-scale", type=float, default=None, help="Rank label font scale")
    parser.add_argument("--rank-font-thickness", type=int, default=None, help="Rank label font thickness")
    parser.add_argument("--line-thickness", type=int, default=None, help="Finish line thickness")
    parser.add_argument("--toe-ankle-radius", type=int, default=None, help="Ankle point radius")
    parser.add_argument("--toe-radius", type=int, default=None, help="Toe proxy point radius")
    parser.add_argument("--toe-link-thickness", type=int, default=None, help="Toe link thickness")
    args = parser.parse_args()

    from ultralytics import YOLO

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            Path("data/sample_res")
            / input_path.parent.name
            / f"{input_path.stem}_finish_res.mp4"
        )

    model = YOLO(str(model_path))
    env_cfg = load_env_config()
    style = {
        "line_color": parse_bgr(
            pick_text_multi(args.line_color, env_cfg, ["VIZ_LINE_COLOR", "FINISH_LINE_COLOR"], "255,0,255"),
            (255, 0, 255),
        ),
        "ready_color": parse_bgr(
            pick_text_multi(args.ready_color, env_cfg, ["VIZ_READY_COLOR", "FINISH_READY_COLOR"], "0,255,0"),
            (0, 255, 0),
        ),
        "running_color": parse_bgr(
            pick_text_multi(
                args.running_color,
                env_cfg,
                ["VIZ_COUNTDOWN_COLOR", "FINISH_RUNNING_COLOR"],
                "0,255,255",
            ),
            (0, 255, 255),
        ),
        "timer_color": parse_bgr(
            pick_text_multi(args.timer_color, env_cfg, ["VIZ_COUNTDOWN_COLOR", "FINISH_TIMER_COLOR"], "0,255,255"),
            (0, 255, 255),
        ),
        "id_color": parse_bgr(
            pick_text_multi(args.id_color, env_cfg, ["FINISH_ID_COLOR"], "255,255,0"), (255, 255, 0)
        ),
        "rank_color": parse_bgr(
            pick_text_multi(args.rank_color, env_cfg, ["FINISH_RANK_COLOR"], "0,255,0"), (0, 255, 0)
        ),
        "toe_ankle_color": parse_bgr(
            pick_text_multi(
                args.toe_ankle_color,
                env_cfg,
                ["VIZ_TOE_ANKLE_COLOR", "FINISH_TOE_ANKLE_COLOR"],
                "255,0,0",
            ),
            (255, 0, 0),
        ),
        "toe_color": parse_bgr(
            pick_text_multi(args.toe_color, env_cfg, ["VIZ_TOE_COLOR", "FINISH_TOE_COLOR"], "0,255,255"),
            (0, 255, 255),
        ),
        "font_scale": pick_float_multi(
            args.hud_font_scale, env_cfg, ["VIZ_HUD_FONT_SCALE", "FINISH_HUD_FONT_SCALE"], 1.0
        ),
        "font_thickness": pick_int_multi(
            args.hud_font_thickness,
            env_cfg,
            ["VIZ_HUD_FONT_THICKNESS", "FINISH_HUD_FONT_THICKNESS"],
            2,
        ),
        "id_font_scale": pick_float_multi(
            args.id_font_scale,
            env_cfg,
            ["FINISH_ID_FONT_SCALE", "VIZ_HUD_FONT_SCALE"],
            0.6,
        ),
        "id_font_thickness": pick_int_multi(
            args.id_font_thickness,
            env_cfg,
            ["FINISH_ID_FONT_THICKNESS", "VIZ_HUD_FONT_THICKNESS"],
            2,
        ),
        "rank_font_scale": pick_float_multi(
            args.rank_font_scale,
            env_cfg,
            ["FINISH_RANK_FONT_SCALE", "VIZ_HUD_FONT_SCALE"],
            0.7,
        ),
        "rank_font_thickness": pick_int_multi(
            args.rank_font_thickness,
            env_cfg,
            ["FINISH_RANK_FONT_THICKNESS", "VIZ_HUD_FONT_THICKNESS"],
            2,
        ),
        "line_thickness": pick_int_multi(
            args.line_thickness,
            env_cfg,
            ["VIZ_LINE_THICKNESS", "FINISH_LINE_THICKNESS"],
            2,
        ),
        "toe_ankle_radius": pick_int_multi(
            args.toe_ankle_radius,
            env_cfg,
            ["VIZ_TOE_ANKLE_RADIUS", "FINISH_TOE_ANKLE_RADIUS"],
            3,
        ),
        "toe_radius": pick_int_multi(
            args.toe_radius,
            env_cfg,
            ["VIZ_TOE_RADIUS", "FINISH_TOE_RADIUS"],
            4,
        ),
        "toe_link_thickness": pick_int_multi(
            args.toe_link_thickness,
            env_cfg,
            ["VIZ_TOE_LINK_THICKNESS", "FINISH_TOE_LINK_THICKNESS"],
            1,
        ),
    }
    process_video(
        model=model,
        video_path=input_path,
        output_path=output_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        monitor_start_at=args.monitor_start_at,
        finish_line_y=args.finish_line_y,
        tracker=args.tracker,
        show=not args.no_show,
        mirror=args.mirror,
        style=style,
    )


if __name__ == "__main__":
    main()
