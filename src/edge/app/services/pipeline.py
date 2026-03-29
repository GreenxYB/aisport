import threading
import time
import queue
import logging
import cv2
import numpy as np
from queue import Queue
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Any

from ultralytics.trackers import BYTETracker
from ultralytics.utils import IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml

from ..core.config import get_settings
from .algorithms.violation import extract_ultralytics_dets
from .algorithms.lane_layout import binding_target_lanes, build_lane_segments

# 获取 logger 实例
logger = logging.getLogger("edge.pipeline")

# 全局配置:队列最大长度,用于控制内存使用和延迟
QUEUE_MAXSIZE = 10


class PipelineTimer:
    """
    流水线计时器 - 单例模式
    
    用于测量流水线各阶段的耗时,帮助性能分析和优化
    采用单例模式确保全局只有一个计时器实例
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """创建单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(PipelineTimer, cls).__new__(cls)
                    cls._instance._init_instance()
        return cls._instance

    def _init_instance(self):
        """初始化计时器数据结构"""
        self._starts = {}  # 记录每个阶段的开始时间
        self._totals = defaultdict(float)  # 累计耗时
        self._counts = defaultdict(int)  # 调用次数
        self._lock = threading.Lock()  # 线程锁

    def start(self, stage: str):
        """
        开始计时某个阶段
        
        Args:
            stage: 阶段名称,如 "1_capture", "2_inference" 等
        """
        thread_id = threading.get_ident()
        with self._lock:
            self._starts[(thread_id, stage)] = time.perf_counter()

    def end(self, stage: str):
        """
        结束计时某个阶段
        
        Args:
            stage: 阶段名称
        """
        thread_id = threading.get_ident()
        with self._lock:
            key = (thread_id, stage)
            if key in self._starts:
                elapsed = time.perf_counter() - self._starts.pop(key)
                self._totals[stage] += elapsed
                self._counts[stage] += 1

    def report(self):
        """输出各阶段的平均耗时报告"""
        logger.info("=" * 50)
        logger.info("流水线各阶段平均耗时 (ms)")
        for stage in sorted(self._totals.keys()):
            if self._counts[stage] > 0:
                avg_ms = (self._totals[stage] / self._counts[stage]) * 1000
                logger.info(f"{stage:20} : {avg_ms:8.2f} ms (n={self._counts[stage]})")
        logger.info("=" * 50)


class Results:
    """
    YOLO 推理结果封装类
    
    用于存储单帧图像的检测结果,包括边界框、置信度、类别和关键点
    支持索引操作,便于数据筛选和子集获取
    """
    
    def __init__(self, orig_img, confs, boxes, cls, keypoints):
        """
        初始化 Results 对象
        
        Args:
            orig_img: 原始图像 (H, W, C)
            confs: 置信度数组 (N,)
            boxes: 边界框数组 xywh 格式 (N, 4)
            cls: 类别ID数组 (N,)
            keypoints: 关键点列表,每个元素是 (17, 3) 的数组
        """
        self.orig_img = orig_img
        self.confs = np.array(confs)
        self.boxes = np.array(boxes)
        self.cls = np.array(cls)
        self.keypoints = keypoints
        
        # 验证数据一致性
        n = len(self.confs)
        assert len(self.boxes) == n and len(self.cls) == n and len(self.keypoints) == n, \
            "所有输入必须具有相同的长度"

    @property
    def conf(self):
        """返回置信度数组"""
        return self.confs

    @property
    def xywh(self):
        """返回 xywh 格式的边界框"""
        return self.boxes

    def __len__(self):
        """返回检测到的目标数量"""
        return len(self.confs)

    def __getitem__(self, idx):
        """
        支持索引操作
        
        Args:
            idx: 整数索引或布尔掩码/整数数组
            
        Returns:
            Results: 单个结果或子集结果
        """
        if isinstance(idx, (int, np.integer)):
            # 返回单个检测结果
            return Results(
                orig_img=self.orig_img,
                confs=[self.confs[idx]],
                boxes=[self.boxes[idx]],
                cls=[self.cls[idx]],
                keypoints=[self.keypoints[idx]]
            )
        elif isinstance(idx, slice):
            indices = list(range(len(self.confs)))[idx]
            return Results(
                orig_img=self.orig_img,
                confs=self.confs[indices],
                boxes=self.boxes[indices],
                cls=self.cls[indices],
                keypoints=[self.keypoints[i] for i in indices],
            )
        elif isinstance(idx, (list, np.ndarray)):
            if len(idx) == 0:
                # ?????
                return Results(
                    orig_img=self.orig_img,
                    confs=np.array([]),
                    boxes=np.empty((0, 4)),
                    cls=np.array([]),
                    keypoints=[]
                )

            # ?????????????????
            try:
                idx_array = np.asarray(idx)
                if idx_array.dtype == bool:
                    selected_indices = np.flatnonzero(idx_array)
                else:
                    selected_indices = idx_array.astype(int)

                boxes = self.boxes[selected_indices]
                confs = self.confs[selected_indices]
                cls = self.cls[selected_indices]
                keypoints = [self.keypoints[i] for i in selected_indices.tolist()]

                return Results(
                    orig_img=self.orig_img,
                    confs=confs,
                    boxes=boxes,
                    cls=cls,
                    keypoints=keypoints
                )
            except IndexError as e:
                raise IndexError(f"Results.__getitem__ ???????? {e}")
        else:
            raise TypeError(f"索引必须是 int, slice 或 ndarray, 得到 {type(idx)}")


class TrackerResults:
    """
    跟踪结果封装类
    
    存储 BYTETracker 的跟踪结果,包含跟踪框、跟踪ID和对应的关键点
    提供可视化绘制功能
    """
    
    def __init__(self, orig_img, result, keypoints):
        """
        初始化 TrackerResults
        
        Args:
            orig_img: 原始图像
            result: 跟踪结果数组 [x1, y1, x2, y2, track_id, conf, cls]
            keypoints: 关键点列表,与 result 一一对应
        """
        self.orig_img = orig_img
        self.result = result  # 跟踪框和ID
        self.keypoints = keypoints  # 关键点列表

    def draw(self):
        """
        在图像上绘制跟踪结果
        
        Returns:
            绘制了边界框、跟踪ID和关键点的图像
        """
        img = self.orig_img.copy()
        for box, kp in zip(self.result, self.keypoints):
            x1, y1, x2, y2, track_id, conf, cls = box
            # 绘制边界框
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 绘制跟踪ID
            cv2.putText(
                img,
                f"ID:{int(track_id)}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            # 绘制关键点
            if hasattr(kp, "__iter__"):
                for k in kp:
                    if len(k) >= 2:
                        kx, ky = int(k[0]), int(k[1])
                        if kx > 0 and ky > 0:
                            cv2.circle(img, (kx, ky), 3, (0, 0, 255), -1)
        return img


class VideoCaptureThread(threading.Thread):
    """
    视频捕获线程
    
    负责从摄像头或视频源捕获帧,并将帧放入队列供后续处理
    支持真实摄像头和模拟模式
    """
    
    def __init__(
        self,
        source,
        frame_queue: Queue,
        width: int,
        height: int,
        fps: int,
        on_running_change=None,
        on_error=None,
    ):
        super().__init__(daemon=True, name="CaptureThread")
        self.source = source
        self.frame_queue = frame_queue
        self.width = width
        self.height = height
        self.fps_target = fps
        self.running = False
        self.cap = None
        self.timer = PipelineTimer()
        self.simulate = False
        self.on_running_change = on_running_change
        self.on_error = on_error

        if not source and source != 0:
            self.simulate = True

    def run(self):
        """Capture frames from a real or simulated source."""
        if self.simulate:
            logger.info("Starting simulated video capture")
            self.running = True
            if self.on_running_change:
                self.on_running_change(True)
            if self.on_error:
                self.on_error(None)
            while self.running:
                self.timer.start("1_capture")
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"SIMULATION {time.time():.2f}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                self.timer.end("1_capture")

                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

                time.sleep(1.0 / self.fps_target)
            self.frame_queue.put(None)
            return

        try:
            logger.info("Opening video source: %s", self.source)
            src = self.source
            if isinstance(src, str) and src.isdigit():
                src = int(src)

            self.cap = cv2.VideoCapture(src)

            if not self.cap.isOpened():
                logger.error("Failed to open video source: %s", self.source)
                if self.on_error:
                    self.on_error(f"Failed to open camera source: {self.source}")
                if self.on_running_change:
                    self.on_running_change(False)
                time.sleep(1.0)
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)

            self.running = True
            if self.on_running_change:
                self.on_running_change(True)
            if self.on_error:
                self.on_error(None)
            logger.info("Video capture started")
            while self.running:
                self.timer.start("1_capture")
                ret, frame = self.cap.read()
                self.timer.end("1_capture")

                if not ret:
                    logger.warning("Frame read failed, retrying...")
                    time.sleep(0.1)
                    continue

                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
        except Exception as e:
            logger.error("Capture thread error: %s", e)
            if self.on_error:
                self.on_error(str(e))
        finally:
            if self.on_running_change:
                self.on_running_change(False)
            if self.cap:
                self.cap.release()
            try:
                self.frame_queue.put(None, block=False)
            except queue.Full:
                pass

    def stop(self):
        self.running = False


class EdgePipeline:
    """
    边缘视频处理流水线
    
    采用多线程架构,包含以下阶段:
    1. 视频捕获 (Capture) - 从摄像头获取帧
    2. 推理 (Inference) - YOLO 姿态检测
    3. 跟踪 (Tracker) - BYTETracker 多目标跟踪
    4. 业务逻辑 (Logic) - 算法处理和应用逻辑
    
    各阶段通过队列解耦,实现并行处理提高吞吐量
    """

    def __init__(self, algo_runner):
        """
        初始化流水线
        
        Args:
            algo_runner: 算法运行器,用于处理跟踪结果
        """
        self.settings = get_settings()
        self.algo_runner = algo_runner
        self.running = False
        self.timer = PipelineTimer()
        self.logger = logging.getLogger("edge.pipeline")
        self._preview_lock = threading.Lock()
        self._last_preview_frame: Optional[np.ndarray] = None
        self._last_jpeg: Optional[bytes] = None
        self._last_encode_error: Optional[str] = None

        # 创建处理队列,用于线程间数据传递
        self.capture_queue = Queue(maxsize=QUEUE_MAXSIZE)
        self.inference_queue = Queue(maxsize=QUEUE_MAXSIZE)
        self.tracking_queue = Queue(maxsize=QUEUE_MAXSIZE)

        # 确定视频源:优先使用 RTSP URL,否则使用摄像头设备
        video_source = self.settings.rtsp_url or self.settings.camera_device
        if self.settings.simulate_camera:
            video_source = None  # 模拟模式
        logger.info(f"视频源: {video_source}")

        # 创建视频捕获线程
        self.capture_thread = VideoCaptureThread(
            video_source,
            self.capture_queue,
            self.settings.capture_width,
            self.settings.capture_height,
            self.settings.capture_fps,
            on_running_change=self._set_capture_running,
            on_error=self._set_capture_error,
        )

        # 创建工作线程
        self.inference_thread = threading.Thread(
            target=self._inference_worker, daemon=True, name="InferenceThread"
        )
        self.tracker_thread = threading.Thread(
            target=self._tracker_worker, daemon=True, name="TrackerThread"
        )
        self.logic_thread = threading.Thread(
            target=self._logic_worker, daemon=True, name="LogicThread"
        )

        # 模型和跟踪器初始化为 None,将在工作线程中加载
        self.model = None
        self.model_kind = None
        self.tracker = None

        # 确定模型路径
        self.model_dir = Path(self.settings.model_dir)
        self.yolo_backend = self.settings.yolo_backend.lower()
        self.engine_path = Path(self.settings.yolo_engine_path)
        self.pt_path = Path(self.settings.yolo_pt_path)
        if not self.engine_path.is_absolute():
            self.engine_path = self.model_dir / self.engine_path
        if not self.pt_path.is_absolute():
            self.pt_path = self.model_dir / self.pt_path
        self._capture_prev_ts_ms: Optional[int] = None
        
        # 加载 BYTETracker 配置文件
        try:
            self.tracker_cfg = check_yaml('bytetrack.yaml')
            self.cfg = IterableSimpleNamespace(**YAML.load(self.tracker_cfg))
            logger.info(f"BYTETracker 配置加载成功: {self.tracker_cfg}")
        except Exception as e:
            logger.warning(f"加载 bytetrack.yaml 失败: {e}, 将使用默认配置")
            # 使用默认配置
            self.cfg = IterableSimpleNamespace(
                tracker_type="bytetrack",
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
                track_buffer=30,
                match_thresh=0.8,
                fuse_score=True,
            )

    def _overlay_lane_guides(self, frame: np.ndarray) -> np.ndarray:
        if not self.settings.display_lane_guides or frame is None or frame.size == 0:
            return frame
        state = getattr(self.algo_runner, "state", None)
        bindings = getattr(state, "bindings", []) if state is not None else []
        lane_count = 0
        if state is not None:
            lane_count = int(state.config.get("lane_count", 0) or 0)
        target_lanes = binding_target_lanes(bindings, lane_count)
        segments = build_lane_segments(
            frame_width=frame.shape[1],
            target_lanes=target_lanes,
            lane_ranges_text=self.settings.lane_x_ranges,
        )
        if not segments:
            return frame
        annotated = frame.copy()
        for segment in segments:
            x1 = int(segment["x1"])
            x2 = int(segment["x2"])
            lane = int(segment["lane"])
            cv2.line(annotated, (x1, 0), (x1, annotated.shape[0] - 1), (255, 255, 0), 1)
            cv2.line(annotated, (x2 - 1, 0), (x2 - 1, annotated.shape[0] - 1), (255, 255, 0), 1)
            cv2.putText(
                annotated,
                f"L{lane}",
                (x1 + 6, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        return annotated

    def start(self):
        """启动流水线所有线程"""
        if self.running:
            return
        self.logger.info("启动 EdgePipeline...")
        self.running = True
        self._capture_prev_ts_ms = None
        with self._preview_lock:
            self._last_preview_frame = None
            self._last_jpeg = None
            self._last_encode_error = None

        # 清空队列中的残留数据
        while not self.capture_queue.empty():
            try:
                self.capture_queue.get(block=False)
            except queue.Empty:
                break
        while not self.inference_queue.empty():
            try:
                self.inference_queue.get(block=False)
            except queue.Empty:
                break
        while not self.tracking_queue.empty():
            try:
                self.tracking_queue.get(block=False)
            except queue.Empty:
                break

        # 启动所有工作线程
        self.capture_thread.start()
        self.inference_thread.start()
        self.tracker_thread.start()
        self.logic_thread.start()

    def stop(self):
        """停止流水线"""
        if not self.running:
            return
        self.logger.info("停止 EdgePipeline...")
        
        # 首先关闭窗口，避免窗口未响应
        try:
            cv2.destroyAllWindows()
            self.logger.info("窗口已关闭")
        except Exception as e:
            self.logger.warning(f"关闭窗口时出错: {e}")
        
        # 标记为停止状态
        self.running = False
        
        # 停止捕获线程
        self.capture_thread.stop()
        
        # 清空队列，确保线程能够退出
        while not self.capture_queue.empty():
            try:
                self.capture_queue.get(block=False)
            except queue.Empty:
                break
        while not self.inference_queue.empty():
            try:
                self.inference_queue.get(block=False)
            except queue.Empty:
                break
        while not self.tracking_queue.empty():
            try:
                self.tracking_queue.get(block=False)
            except queue.Empty:
                break
        
        # 等待所有线程退出
        self.logger.info("等待线程退出...")
        try:
            # 等待捕获线程退出
            self.capture_thread.join(timeout=10.0)
            # 等待推理线程退出
            if self.inference_thread.is_alive():
                self.inference_thread.join(timeout=10.0)
            # 等待跟踪线程退出
            if self.tracker_thread.is_alive():
                self.tracker_thread.join(timeout=10.0)
            # 等待逻辑线程退出
            if self.logic_thread.is_alive():
                self.logic_thread.join(timeout=10.0)
        except Exception as e:
            self.logger.warning(f"等待线程退出时出错: {e}")

        # 输出性能报告
        self.timer.report()
        
        # 重新创建队列，避免残留数据
        from queue import Queue
        self.capture_queue = Queue(maxsize=50)  # 增大队列大小
        self.inference_queue = Queue(maxsize=50)  # 增大队列大小
        self.tracking_queue = Queue(maxsize=50)  # 增大队列大小
        
        # 重置跟踪器配置
        self.cfg = IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True
        )
        self.logger.info("BYTETracker 配置重置成功")
        
        # 重新创建视频捕获线程，以便下次启动时能够重新打开摄像头
        try:
            # 确定视频源:优先使用 RTSP URL,否则使用摄像头设备
            video_source = self.settings.rtsp_url or self.settings.camera_device
            if self.settings.simulate_camera:
                video_source = None  # 模拟模式
            self.capture_thread = VideoCaptureThread(
                video_source,
                self.capture_queue,
                self.settings.capture_width,
                self.settings.capture_height,
                self.settings.capture_fps,
                on_running_change=self._set_capture_running,
                on_error=self._set_capture_error,
            )
            # 重新创建工作线程
            self.inference_thread = threading.Thread(
                target=self._inference_worker, daemon=True, name="InferenceThread"
            )
            self.tracker_thread = threading.Thread(
                target=self._tracker_worker, daemon=True, name="TrackerThread"
            )
            self.logic_thread = threading.Thread(
                target=self._logic_worker, daemon=True, name="LogicThread"
            )
            # 注意：模型资源会由垃圾回收器自动处理
            # 由于模型在推理线程中创建，必须在该线程中清理
            # 这里不进行显式清理，避免跨线程 CUDA 资源操作
            self.model = None
            self.tracker = None  # 重置跟踪器
            self.logger.info("线程资源重新初始化完成")
        except Exception as e:
            self.logger.error(f"重新初始化线程资源时出错: {e}")

    def _update_preview_cache(self, frame: np.ndarray) -> None:
        preview = np.ascontiguousarray(frame)
        try:
            ok, buf = cv2.imencode(".jpg", preview)
        except Exception as exc:
            with self._preview_lock:
                self._last_preview_frame = preview.copy()
                self._last_encode_error = f"imencode_exception:{exc}"
            return

        with self._preview_lock:
            self._last_preview_frame = preview.copy()
            if ok:
                self._last_jpeg = buf.tobytes()
                self._last_encode_error = None
            else:
                self._last_encode_error = "imencode_failed"

    def snapshot_jpeg(self) -> Optional[bytes]:
        with self._preview_lock:
            if self._last_jpeg:
                return self._last_jpeg
            if self._last_preview_frame is None:
                return None
            frame = self._last_preview_frame.copy()

        try:
            ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(frame))
        except Exception as exc:
            with self._preview_lock:
                self._last_encode_error = f"imencode_exception:{exc}"
            return None

        if not ok:
            with self._preview_lock:
                self._last_encode_error = "imencode_failed"
            return None

        jpeg = buf.tobytes()
        with self._preview_lock:
            self._last_jpeg = jpeg
            self._last_encode_error = None
        return jpeg

    def last_encode_error(self) -> Optional[str]:
        with self._preview_lock:
            return self._last_encode_error

    def _empty_results(self, frame: np.ndarray) -> Results:
        return Results(frame, np.array([]), np.empty((0, 4)), np.array([]), [])

    def _load_model(self):
        backend = self.yolo_backend

        if backend == "trt":
            if not self.engine_path.exists():
                self.logger.warning("TRT engine not found: %s", self.engine_path)
                return None, None
            try:
                from .algorithms.models.yolo_trt import TRTYOLO

                model = TRTYOLO(str(self.engine_path))
                self.logger.info("Loaded TRT model: %s", self.engine_path)
                return model, "trt"
            except Exception as exc:
                self.logger.error("Failed to load TRT model: %s", exc)
                return None, None

        try:
            from ultralytics import YOLO as UltralyticsYOLO
        except Exception as exc:
            self.logger.error("Failed to import ultralytics: %s", exc)
            return None, None

        model_path = self.pt_path
        if not model_path.exists():
            candidates = sorted(self.model_dir.glob("*.pt"))
            if not candidates:
                self.logger.warning("No YOLO .pt model found in %s", self.model_dir)
                return None, None
            model_path = candidates[0]

        try:
            model = UltralyticsYOLO(str(model_path))
            self.logger.info("Loaded Ultralytics PT model: %s", model_path)
            return model, "pt"
        except Exception as exc:
            self.logger.error("Failed to load Ultralytics model: %s", exc)
            return None, None

    def _infer_with_model(self, frame: np.ndarray) -> Results:
        if self.model is None:
            return self._empty_results(frame)

        model_kind = getattr(self, "model_kind", self.yolo_backend)
        if model_kind == "trt":
            outputs = self.model.infer(
                frame,
                conf_thres=self.settings.yolo_conf_thres,
                iou_thres=self.settings.yolo_iou_thres,
            )
            dets = outputs or []
        else:
            results = self.model.predict(
                source=frame,
                conf=self.settings.yolo_conf_thres,
                iou=self.settings.yolo_iou_thres,
                imgsz=self.settings.yolo_imgsz,
                verbose=False,
            )
            dets = extract_ultralytics_dets(results[0]) if results else []

        if not dets:
            return self._empty_results(frame)

        boxes = np.zeros((len(dets), 4), dtype=np.float32)
        cls = []
        confs = []
        keypoints = []

        for i, output in enumerate(dets):
            x1, y1, x2, y2 = output["bbox"]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            boxes[i] = [x, y, w, h]
            cls.append(output.get("class_id", 0))
            confs.append(output.get("score", 0.0) or 0.0)
            keypoints.append(output.get("keypoints") or [])

        return Results(frame, np.array(confs), boxes, np.array(cls), keypoints)

    def _update_capture_stats(self) -> None:
        state = getattr(self.algo_runner, "state", None)
        if state is None:
            return
        now_ms = int(time.time() * 1000)
        state.last_frame_ts = now_ms
        if self._capture_prev_ts_ms is not None:
            dt_ms = max(now_ms - self._capture_prev_ts_ms, 1)
            fps = 1000.0 / dt_ms
            prev = state.capture_fps_est
            state.capture_fps_est = round(fps if prev is None else prev * 0.8 + fps * 0.2, 2)
        self._capture_prev_ts_ms = now_ms

    def _set_capture_running(self, running: bool) -> None:
        state = getattr(self.algo_runner, "state", None)
        if state is None:
            return
        state.capture_running = running
        if not running:
            state.capture_fps_est = None

    def _set_capture_error(self, error: Optional[str]) -> None:
        state = getattr(self.algo_runner, "state", None)
        if state is None:
            return
        state.capture_error = error

    def _inference_worker(self):
        """
        ?????????

        ?????????????????YOLO ???????????????????????
        """
        local_model = None
        try:
            if self.model is None:
                local_model, self.model_kind = self._load_model()
                self.model = local_model

            start_time = time.time()
            while self.running:
                try:
                    frame = self.capture_queue.get(timeout=1.0)
                except queue.Empty:
                    if time.time() - start_time > 10:
                        self.logger.info("Inference worker waiting for frames...")
                        start_time = time.time()
                    continue

                if frame is None:
                    self.logger.info("Inference worker received shutdown sentinel")
                    self.inference_queue.put(None)
                    break

                self._update_capture_stats()
                self.timer.start("2_inference")
                try:
                    result = self._infer_with_model(frame)
                    self.inference_queue.put((frame, result))
                except Exception as e:
                    self.logger.error("Inference failed: %s", e)
                    self.inference_queue.put((frame, self._empty_results(frame)))
                finally:
                    self.timer.end("2_inference")
        finally:
            if local_model is not None:
                try:
                    if hasattr(local_model, 'cleanup'):
                        local_model.cleanup()
                    local_model = None
                    self.model = None
                    self.model_kind = None
                    self.logger.info("Inference worker model cleanup finished")
                except Exception as e:
                    self.logger.warning("Model cleanup failed: %s", e)

    def _tracker_worker(self):
        """
        跟踪工作线程
        
        从推理队列获取检测结果,使用 BYTETracker 进行多目标跟踪
        保持跟踪ID一致性,将跟踪结果放入跟踪队列
        """
        self.logger.info("启动跟踪线程")
        frame_count = 0
        # 在此线程中初始化跟踪器
        try:
            self.tracker = BYTETracker(args=self.cfg, frame_rate=self.settings.capture_fps)
            self.logger.info("BYTETracker 初始化成功")
        except Exception as e:
            self.logger.error(f"BYTETracker 初始化失败: {e}")
            self.running = False
            return

        while self.running:
            try:
                item = self.inference_queue.get(timeout=1.0)
            except queue.Empty:
                # 每10秒打印一次日志，确认线程还在运行
                if time.time() % 10 < 0.1:
                    self.logger.info("跟踪线程等待推理结果...")
                continue

            if item is None:
                self.logger.info("收到结束信号，退出跟踪线程")
                self.tracking_queue.put(None)
                break

            frame, result = item
            frame_count += 1
            # 每10帧打印一次跟踪日志，避免日志过多
            if frame_count % 10 == 0:
                self.logger.info("收到推理结果，开始跟踪")
            self.timer.start("3_tracking")

            try:
                # 使用 BYTETracker 进行跟踪
                # tracks 格式: [x1, y1, x2, y2, track_id, conf, cls, idx]
                tracks = self.tracker.update(result, frame)
                # 每10帧打印一次跟踪结果，避免日志过多
                if frame_count % 10 == 0:
                    self.logger.info(f"跟踪结果: {len(tracks)} 个目标")
                
                # 提取关键点,保持与 tracks 的对应关系
                keypoints = []
                if len(tracks) > 0:
                    # 获取原始检测结果的索引(最后一列)
                    idx = tracks[:, -1].astype(int)
                    for i in idx:
                        if 0 <= i < len(result.keypoints):
                            keypoints.append(result.keypoints[i])
                        else:
                            keypoints.append([])
                    # 移除索引列,得到标准格式 [x1, y1, x2, y2, track_id, conf, cls]
                    tracks = tracks[:, :-1]
                    tracker_result = TrackerResults(frame, tracks, keypoints)
                else:
                    tracker_result = TrackerResults(frame, np.array([]), [])

            except Exception as e:
                self.logger.error(f"跟踪错误: {e}")
                tracker_result = TrackerResults(frame, [], [])

            self.timer.end("3_tracking")
            # 每10帧打印一次跟踪完成日志，避免日志过多
            if frame_count % 10 == 0:
                self.logger.info("跟踪完成，放入跟踪队列")
            self.tracking_queue.put((frame, tracker_result))

    def _logic_worker(self):
        """
        ????????????

        ?????????????????????????????????????????
        """
        fps = 0.0
        prev_time = time.time()
        frame_count = 0
        window_created = False

        while self.running:
            try:
                item = self.tracking_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            frame, tracker_result = item
            frame_count += 1

            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = curr_time

            self.timer.start("4_business_logic")
            if self.algo_runner:
                if hasattr(self.algo_runner, "process_pipeline_result"):
                    self.algo_runner.process_pipeline_result(
                        frame, tracker_result, curr_time * 1000
                    )
                else:
                    self.algo_runner.process_frame(frame, curr_time * 1000)
            self.timer.end("4_business_logic")

            self.timer.start("5_drawing")
            try:
                annotated_frame = tracker_result.draw()
                annotated_frame = self._overlay_lane_guides(annotated_frame)
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (annotated_frame.shape[1] - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                if self.settings.display_mirror:
                    annotated_frame = cv2.flip(annotated_frame, 1)

                self._update_preview_cache(annotated_frame)

                if self.settings.display_preview:
                    if not window_created:
                        cv2.destroyAllWindows()
                        cv2.namedWindow("Edge Node Preview", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Edge Node Preview", 1280, 720)
                        window_created = True

                    cv2.imshow("Edge Node Preview", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        self.stop()
            except Exception as e:
                self.logger.debug("preview draw failed: %s", e)
            finally:
                self.timer.end("5_drawing")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
