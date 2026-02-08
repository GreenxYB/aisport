import logging
import threading
import time
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python-headless is required for camera capture") from exc

from ..core.config import get_settings


FrameCallback = Callable[[np.ndarray, float], None]  # frame, ts_ms


class CameraSource:
    def __init__(self):
        self.settings = get_settings()
        self.log = logging.getLogger("edge.camera")
        self.cap: Optional[cv2.VideoCapture] = None
        self.simulate = self.settings.simulate_camera

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
        ts_ms = time.time() * 1000
        return ok, frame if ok else None, ts_ms if ok else None

    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None
            self.log.info("Camera released")


class CaptureManager:
    def __init__(self, on_frame: FrameCallback):
        self.settings = get_settings()
        self.on_frame = on_frame
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
