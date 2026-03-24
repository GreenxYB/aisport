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
from .algorithms.models.yolo_trt import TRTYOLO as YOLO

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
        elif isinstance(idx, (list, np.ndarray)):
            if len(idx) == 0:
                # 空索引
                return Results(
                    orig_img=self.orig_img,
                    confs=np.array([]),
                    boxes=np.empty((0, 4)),
                    cls=np.array([]),
                    keypoints=[]
                )
            
            # 布尔掩码或整数索引数组
            try:
                boxes = self.boxes[idx]
                confs = self.confs[idx]
                cls = self.cls[idx]
                keypoints = [self.keypoints[i] for i in idx]
                
                return Results(
                    orig_img=self.orig_img,
                    confs=confs,
                    boxes=boxes,
                    cls=cls,
                    keypoints=keypoints
                )
            except IndexError as e:
                raise IndexError(f"Results.__getitem__ 中索引无效: {e}")
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
    
    def __init__(self, source, frame_queue: Queue, width: int, height: int, fps: int):
        """
        初始化视频捕获线程
        
        Args:
            source: 视频源,可以是设备索引、RTSP URL 或 None(模拟模式)
            frame_queue: 帧队列,用于传递捕获的帧
            width: 目标宽度
            height: 目标高度
            fps: 目标帧率
        """
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

        # 如果源为空,启用模拟模式
        if not source and source != 0:
            self.simulate = True

    def run(self):
        """线程主函数 - 捕获视频帧"""
        if self.simulate:
            # 模拟模式:生成虚拟帧
            logger.info("启动模拟视频捕获")
            self.running = True
            while self.running:
                self.timer.start("1_capture")
                # 生成黑色背景帧
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                # 添加时间戳文字
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

        # 真实摄像头模式
        try:
            logger.info(f"打开视频源: {self.source}")
            # 尝试将字符串数字转换为整数(设备索引)
            src = self.source
            if isinstance(src, str) and src.isdigit():
                src = int(src)

            # Windows 平台使用 DirectShow 后端
            if False:  # 如需 Windows 支持,改为: sys.platform == 'win32'
                self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(src)

            if not self.cap.isOpened():
                logger.error(f"无法打开视频源 {self.source}")
                return

            # 设置视频参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)

            self.running = True
            while self.running:
                self.timer.start("1_capture")
                ret, frame = self.cap.read()
                self.timer.end("1_capture")

                if not ret:
                    logger.warning("读取帧失败,重试中...")
                    time.sleep(0.1)
                    continue

                # 队列满时丢弃帧以保持实时性
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
        except Exception as e:
            logger.error(f"捕获线程错误: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.frame_queue.put(None)

    def stop(self):
        """停止捕获线程"""
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
        self.tracker = None

        # 确定模型路径
        self.model_dir = Path(self.settings.model_dir)
        self.model_path = str(self.model_dir / "yolo11n-pose.engine")
        
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

    def start(self):
        """启动流水线所有线程"""
        if self.running:
            return
        self.logger.info("启动 EdgePipeline...")
        self.running = True

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
        self.running = False
        self.capture_thread.stop()

        # 输出性能报告
        self.timer.report()
        cv2.destroyAllWindows()

    def _inference_worker(self):
        """
        推理工作线程
        
        从捕获队列获取帧,使用 YOLO 模型进行姿态检测
        将检测结果放入推理队列供跟踪线程使用
        """
        # 在此线程中加载模型以确保正确的 CUDA 上下文
        if self.model is None:
            try:
                self.logger.info(f"加载 TRT 模型: {self.model_path}")
                if Path(self.model_path).exists():
                    self.model = YOLO(self.model_path)
                    self.logger.info("TRT YOLO 模型加载成功")
                else:
                    self.logger.warning(f"模型文件不存在: {self.model_path}")
            except Exception as e:
                self.logger.error(f"加载 TRT 模型失败: {e}")

        while self.running:
            try:
                frame = self.capture_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:
                self.inference_queue.put(None)
                break

            self.timer.start("2_inference")
            try:
                if self.model:
                    # 执行模型推理
                    outputs = self.model.infer(frame)
                    # 转换输出格式为 Results
                    boxes = np.zeros((len(outputs), 4), dtype=np.float32)
                    cls = []
                    confs = []
                    keypoints = []

                    for i, output in enumerate(outputs):
                        x1, y1, x2, y2 = output["bbox"]
                        # 转换为 xywh 格式
                        x = (x1 + x2) / 2
                        y = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        boxes[i] = [x, y, w, h]
                        cls.append(output["class_id"])
                        confs.append(output["score"])
                        if "keypoints" in output:
                            keypoints.append(output["keypoints"])

                    result = Results(
                        frame, np.array(confs), boxes, np.array(cls), keypoints
                    )
                    self.inference_queue.put((frame, result))
                else:
                    # 模型未加载,传递空结果
                    self.inference_queue.put((frame, Results(frame, [], [], [], [])))

            except Exception as e:
                self.logger.error(f"推理错误: {e}")
            finally:
                self.timer.end("2_inference")

    def _tracker_worker(self):
        """
        跟踪工作线程
        
        从推理队列获取检测结果,使用 BYTETracker 进行多目标跟踪
        保持跟踪ID一致性,将跟踪结果放入跟踪队列
        """
        # 在此线程中初始化跟踪器
        try:
            self.tracker = BYTETracker(args=self.cfg, frame_rate=self.settings.capture_fps)
            logger.info("BYTETracker 初始化成功")
        except Exception as e:
            logger.error(f"BYTETracker 初始化失败: {e}")
            self.running = False
            return

        while self.running:
            try:
                item = self.inference_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                self.tracking_queue.put(None)
                break

            frame, result = item
            self.timer.start("3_tracking")

            try:
                # 使用 BYTETracker 进行跟踪
                # tracks 格式: [x1, y1, x2, y2, track_id, conf, cls, idx]
                tracks = self.tracker.update(result, frame)
                
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
                logger.error(f"跟踪错误: {e}")
                tracker_result = TrackerResults(frame, [], [])

            self.timer.end("3_tracking")
            self.tracking_queue.put((frame, tracker_result))

    def _logic_worker(self):
        """
        业务逻辑工作线程
        
        从跟踪队列获取跟踪结果,调用算法运行器进行处理
        负责可视化显示和性能统计
        """
        fps = 0.0
        prev_time = time.time()

        while self.running:
            try:
                item = self.tracking_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break

            frame, tracker_result = item

            # 计算 FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = curr_time

            self.timer.start("4_business_logic")

            # 调用算法运行器处理跟踪结果
            if self.algo_runner:
                if hasattr(self.algo_runner, "process_pipeline_result"):
                    self.algo_runner.process_pipeline_result(
                        frame, tracker_result, curr_time * 1000
                    )
                else:
                    # 回退到旧的帧处理方法
                    self.algo_runner.process_frame(frame, curr_time * 1000)

            self.timer.end("4_business_logic")

            # 可视化显示
            if self.settings.display_preview:
                self.timer.start("5_drawing")
                annotated_frame = tracker_result.draw()
                # 绘制 FPS
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (annotated_frame.shape[1] - 150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                # 镜像显示(如果需要)
                if self.settings.display_mirror:
                    annotated_frame = cv2.flip(annotated_frame, 1)

                cv2.imshow("Edge Node Preview", annotated_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC 键退出
                    self.stop()
                self.timer.end("5_drawing")
