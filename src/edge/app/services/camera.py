import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python-headless is required for camera capture") from exc

from ..core.config import get_settings
from ..core.state import NodeState


FrameCallback = Callable[[np.ndarray, float], None]  # frame, ts_ms


class CameraSource:
    def __init__(self):
        self.settings = get_settings()
        self.log = logging.getLogger("edge.camera")
        self.cap: Optional[cv2.VideoCapture] = None
        self.simulate = self.settings.simulate_camera
        self._video_files: list[Path] = []
        self._video_index = 0
        self._video_dir_mode = False

    def open(self) -> None:
        if self.simulate:
            self.log.info("Camera in simulate mode; no physical device needed")
            return
        source = self.settings.rtsp_url or self.settings.camera_device
        # Accept numeric index passed as string
        if isinstance(source, str) and source.isdigit():
            source_idx = int(source)
            # Prefer DirectShow on Windows for more stable webcam open
            self.cap = cv2.VideoCapture(source_idx, cv2.CAP_DSHOW)
        else:
            src_path = Path(str(source))
            if src_path.exists() and src_path.is_dir():
                self._video_files = sorted(
                    p for p in src_path.rglob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
                )
                if not self._video_files:
                    raise RuntimeError(f"No video files found in directory: {source}")
                self._video_dir_mode = True
                self._video_index = 0
                self.cap = cv2.VideoCapture(str(self._video_files[self._video_index]))
                self.log.info("Camera directory mode enabled (%s files)", len(self._video_files))
            else:
                self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {source}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.settings.capture_fps)
        self.log.info("Camera opened source=%s", source)

    def read_ts(self) -> tuple[bool, Optional[np.ndarray], Optional[float]]:
        """
        Returns (success, frame, timestamp_ms).
        """
        if self.simulate:
            # generate dummy frame to keep pipeline alive
            frame = np.zeros(
                (self.settings.capture_height, self.settings.capture_width, 3), dtype=np.uint8
            )
            ts_ms = time.time() * 1000
            return True, frame, ts_ms
        if not self.cap:
            return False, None, None
        ok, frame = self.cap.read()
        if not ok and self._video_dir_mode:
            while True:
                self._video_index += 1
                if self._video_index >= len(self._video_files):
                    self._video_index = 0
                self.cap.release()
                self.cap = cv2.VideoCapture(str(self._video_files[self._video_index]))
                ok, frame = self.cap.read()
                if ok or not self._video_files:
                    break
        ts_ms = time.time() * 1000
        if ok and frame is not None:
            target_w, target_h = self.settings.capture_width, self.settings.capture_height
            if frame.shape[1] != target_w or frame.shape[0] != target_h:
                frame = cv2.resize(frame, (target_w, target_h))
        return ok, frame if ok else None, ts_ms if ok else None

    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
            self.log.info("Camera released")


class CaptureManager:
    def __init__(self, on_frame: FrameCallback, state: Optional[NodeState] = None):
        self.settings = get_settings()
        self.on_frame = on_frame
        self.state = state
        self.camera = CameraSource()
        self.log = logging.getLogger("edge.capture")
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._last_ts: Optional[float] = None
        self._last_jpeg: Optional[bytes] = None
        self._last_encode_error: Optional[str] = None
        self._lock = threading.Lock()
        self._display_thread: Optional[threading.Thread] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_preview_frame: Optional[np.ndarray] = None
        self._style = self._build_style()

    @staticmethod
    def _parse_bgr(text: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
        try:
            parts = [int(x.strip()) for x in str(text).split(",")]
            if len(parts) != 3:
                return default
            return (
                max(0, min(255, parts[0])),
                max(0, min(255, parts[1])),
                max(0, min(255, parts[2])),
            )
        except Exception:
            return default

    def _build_style(self) -> dict:
        return {
            "line_color": self._parse_bgr(self.settings.viz_line_color, (0, 0, 255)),
            "ready_color": self._parse_bgr(self.settings.viz_ready_color, (0, 255, 0)),
            "alert_color": self._parse_bgr(self.settings.viz_alert_color, (0, 0, 255)),
            "box_color": self._parse_bgr(self.settings.viz_box_color, (0, 255, 0)),
            "toe_ankle_color": self._parse_bgr(self.settings.viz_toe_ankle_color, (255, 0, 0)),
            "toe_color": self._parse_bgr(self.settings.viz_toe_color, (0, 255, 255)),
            "font_scale": float(self.settings.viz_hud_font_scale),
            "font_thickness": int(self.settings.viz_hud_font_thickness),
            "line_thickness": int(self.settings.viz_line_thickness),
            "box_thickness": int(self.settings.viz_box_thickness),
            "toe_ankle_radius": int(self.settings.viz_toe_ankle_radius),
            "toe_radius": int(self.settings.viz_toe_radius),
            "toe_link_thickness": int(self.settings.viz_toe_link_thickness),
        }

    def start(self, raise_on_fail: bool = False) -> bool:
        if self._running.is_set():
            return True
        try:
            self.camera.open()
        except Exception as exc:
            self.log.error("Camera open failed: %s", exc)
            if raise_on_fail:
                raise
            return False
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="capture-loop", daemon=True)
        self._thread.start()
        self.log.info("Capture loop started")
        return True

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2)
        self.camera.close()
        self.log.info("Capture loop stopped")

    def _loop(self) -> None:
        interval = 1.0 / max(self.settings.capture_fps, 1)
        while self._running.is_set():
            ok, frame, ts_ms = self.camera.read_ts()
            if ok and ts_ms and frame is not None:
                self._last_ts = ts_ms
                try:
                    frame = np.ascontiguousarray(frame)
                    preview = frame
                    if self.settings.display_mirror:
                        preview = cv2.flip(frame, 1)
                    if self.settings.display_start_line:
                        if preview is frame:
                            preview = preview.copy()
                        line_y = int(self.settings.start_line_y * preview.shape[0] / 640)
                        cv2.line(
                            preview,
                            (0, line_y),
                            (preview.shape[1] - 1, line_y),
                            self._style["line_color"],
                            self._style["line_thickness"],
                        )
                    self._overlay_false_start(preview)
                    # Encode frame for preview
                    ret, buf = cv2.imencode(".jpg", preview)
                    if ret:
                        with self._lock:
                            self._last_jpeg = buf.tobytes()
                            self._last_frame = frame
                            self._last_preview_frame = preview
                            self._last_encode_error = None
                    else:
                        with self._lock:
                            self._last_encode_error = "imencode_failed"
                except Exception as exc:  # pragma: no cover
                    self.log.warning("JPEG encode failed: %s", exc)
                    with self._lock:
                        self._last_encode_error = f"imencode_exception:{exc}"
                try:
                    self.on_frame(frame, ts_ms)
                except Exception as exc:  # pragma: no cover
                    self.log.warning("Frame callback error: %s", exc)
            time.sleep(interval)

    @property
    def last_ts(self) -> Optional[float]:
        return self._last_ts

    def snapshot_jpeg(self) -> Optional[bytes]:
        with self._lock:
            if self._last_jpeg:
                return self._last_jpeg
            if self._last_preview_frame is not None:
                frame = self._last_preview_frame
            elif self._last_frame is not None:
                frame = self._last_frame
                if self.settings.display_mirror:
                    frame = cv2.flip(frame, 1)
            else:
                return None
            # Try encode on demand
            frame = np.ascontiguousarray(frame)
        try:
            ret, buf = cv2.imencode(".jpg", frame)
            if ret:
                jpeg = buf.tobytes()
                with self._lock:
                    self._last_jpeg = jpeg
                    self._last_encode_error = None
                return jpeg
        except Exception as exc:
            with self._lock:
                self._last_encode_error = f"imencode_exception:{exc}"
        return None

    def last_encode_error(self) -> Optional[str]:
        with self._lock:
            return self._last_encode_error

    # ---- local display (debug only) ----
    def start_display(self) -> None:
        if self._display_thread and self._display_thread.is_alive():
            return
        if not self.settings.display_preview:
            return
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True, name="capture-display")
        self._display_thread.start()
        self.log.info("Display thread started (cv2.imshow)")

    def _display_loop(self) -> None:
        try:
            while self._running.is_set():
                frame = None
                with self._lock:
                    if self._last_preview_frame is not None:
                        frame = self._last_preview_frame.copy()
                    elif self._last_frame is not None:
                        frame = self._last_frame.copy()
                        if self.settings.display_mirror:
                            frame = cv2.flip(frame, 1)
                if frame is not None:
                    try:
                        cv2.imshow("Edge Preview", frame)
                        cv2.waitKey(1)
                    except Exception as exc:  # pragma: no cover
                        self.log.warning("Display failed: %s", exc)
                        break
                time.sleep(0.03)
        finally:  # pragma: no cover
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _overlay_false_start(self, preview: np.ndarray) -> None:
        if not self.state:
            return
        debug_payload = self.state.last_toe_proxy_debug
        debug_ts = self.state.last_toe_proxy_ts
        if debug_payload and debug_ts and int(time.time() * 1000) - int(debug_ts) <= 500:
            items = debug_payload.get("items", [])
            for item in items:
                bbox = item.get("bbox")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    cv2.rectangle(
                        preview, (x1, y1), (x2, y2), self._style["box_color"], self._style["box_thickness"]
                    )
                toe_points = item.get("toe_proxy_points")
                if not isinstance(toe_points, list):
                    continue
                for point in toe_points:
                    ankle = point.get("ankle")
                    toe = point.get("toe")
                    if isinstance(ankle, list) and len(ankle) >= 2:
                        cv2.circle(
                            preview,
                            (int(ankle[0]), int(ankle[1])),
                            self._style["toe_ankle_radius"],
                            self._style["toe_ankle_color"],
                            -1,
                        )
                    if isinstance(toe, list) and len(toe) >= 2:
                        cv2.circle(
                            preview,
                            (int(toe[0]), int(toe[1])),
                            self._style["toe_radius"],
                            self._style["toe_color"],
                            -1,
                        )
                    if (
                        isinstance(ankle, list)
                        and len(ankle) >= 2
                        and isinstance(toe, list)
                        and len(toe) >= 2
                    ):
                        cv2.line(
                            preview,
                            (int(ankle[0]), int(ankle[1])),
                            (int(toe[0]), int(toe[1])),
                            self._style["toe_color"],
                            self._style["toe_link_thickness"],
                        )
        event = self.state.last_false_start_event
        ts = self.state.last_false_start_ts
        if not event or not ts:
            return
        if int(time.time() * 1000) - int(ts) > 2000:
            return
        bbox = event.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(
                preview,
                (x1, y1),
                (x2, y2),
                self._style["alert_color"],
                self._style["line_thickness"],
            )
        kps = event.get("keypoints")
        if kps and len(kps) >= 17:
            for idx in (15, 16):
                try:
                    x, y, s = kps[idx]
                except Exception:
                    continue
                if s is not None and s >= self.settings.kps_conf_thres:
                    cv2.circle(
                        preview,
                        (int(x), int(y)),
                        self._style["toe_radius"],
                        self._style["alert_color"],
                        -1,
                    )
        toe_points = event.get("toe_proxy_points")
        if isinstance(toe_points, list):
            for point in toe_points:
                ankle = point.get("ankle")
                toe = point.get("toe")
                if isinstance(ankle, list) and len(ankle) >= 2:
                    cv2.circle(
                        preview,
                        (int(ankle[0]), int(ankle[1])),
                        self._style["toe_ankle_radius"],
                        self._style["toe_ankle_color"],
                        -1,
                    )
                if isinstance(toe, list) and len(toe) >= 2:
                    cv2.circle(
                        preview,
                        (int(toe[0]), int(toe[1])),
                        self._style["toe_radius"],
                        self._style["toe_color"],
                        -1,
                    )
                if (
                    isinstance(ankle, list)
                    and len(ankle) >= 2
                    and isinstance(toe, list)
                    and len(toe) >= 2
                ):
                    cv2.line(
                        preview,
                        (int(ankle[0]), int(ankle[1])),
                        (int(toe[0]), int(toe[1])),
                        self._style["toe_color"],
                        self._style["toe_link_thickness"],
                    )
        lane = event.get("lane")
        label = f"FALSE START" if lane is None else f"FALSE START L{lane}"
        cv2.putText(
            preview,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._style["font_scale"],
            self._style["alert_color"],
            self._style["font_thickness"],
            cv2.LINE_AA,
        )
