# -*- coding: UTF-8 -*-
"""
  @Author: mpj
  @Date  : 2025/10/25 00:04
  @version V1.0
"""
import platform
import re
import threading
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import torch
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import os

from ultralytics.data import load_inference_source
from ultralytics.engine.results import Results
from ultralytics.trackers import BYTETracker, BOTSORT
from ultralytics.utils import ops, colorstr, MACOS, WINDOWS, IterableSimpleNamespace, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.files import increment_path

# 和cu文件保持同步
NUM_POSE_ELEMENT = 7 + 17 * 3  # left, top, right, bottom, confidence, class, keepflag, 17 keypoints * (x, y, score)
MAX_IMAGE_BOXES = 1024
NUM_KEYPOINTS = 17  # COCO dataset keypoint number
GPU_BLOCK_THREADS = 512


def grid_dims(num_jobs):
    block = min(num_jobs, GPU_BLOCK_THREADS)
    grid = (num_jobs + block - 1) // block
    return (grid, 1)


class TRTYOLO:
    def __init__(self, engine_path, device_id=0, cu_file="yolo_kernels.cu"):
        self.device_id = device_id
        cuda.init()
        if device_id >= cuda.Device.count():
            raise ValueError(f"Invalid device_id {device_id}, only {cuda.Device.count()} GPUs available.")
        self.device = cuda.Device(device_id)
        self.context_gpu = self.device.make_context()  # Switch to the specified GPU context

        try:
            self.logger = trt.Logger(trt.Logger.ERROR)
            self.runtime = trt.Runtime(self.logger)
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())  # type: trt.ICudaEngine
            self.context = self.engine.create_execution_context()  # type: trt.IExecutionContext

            # Check input/output bindings
            assert self.engine.num_io_tensors == 2, "Expected 1 input + 1 output"
            self.input_name = self.engine.get_tensor_name(0)
            self.output_name = self.engine.get_tensor_name(1)

            self.input_shape = self.engine.get_tensor_shape(self.input_name)  # [1, 3, H, W]
            self.output_shape = self.engine.get_tensor_shape(self.output_name)  # [1, N, C]

            assert self.input_shape[0] == 1, "Only batch=1 supported"
            assert len(self.output_shape) == 3 and self.output_shape[0] == 1, "Output must be [1, N, C]"

            self.model_h, self.model_w = self.input_shape[2], self.input_shape[3]
            self.num_boxes = self.output_shape[1]
            self.output_c_dim = self.output_shape[2]
            self.num_classes = self.output_c_dim - 4 - NUM_KEYPOINTS * 3

            # Allocate buffers
            self.stream = cuda.Stream()
            self.inputs, self.outputs = self._allocate_buffers()

            # Load CUDA kernels
            mod = self._load_cuda_kernels(cu_file)
            self.warp_kernel = mod.get_function("warp_affine_bilinear_and_normalize_plane_kernel")
            self.decode_kernel = mod.get_function("decode_kernel_Pose")
            self.nms_kernel = mod.get_function("nms_kernel_Pose")

            # Persistent GPU buffers
            self.d2i_gpu = cuda.mem_alloc(6 * 4)  # float32[6]
            self.parray_size = (1 + MAX_IMAGE_BOXES * NUM_POSE_ELEMENT) * 4  # bytes
            self.parray_gpu = cuda.mem_alloc(self.parray_size)

        except Exception as e:
            self.context_gpu.pop()  # Restore context on exception
            raise e

    def __del__(self):
        # 清理 CUDA 上下文
        if hasattr(self, 'context_gpu'):
            self.context_gpu.pop()

    def _load_cuda_kernels(self, cu_file):
        cu_file = os.path.join(os.path.dirname(__file__), cu_file)
        if not os.path.exists(cu_file):
            raise FileNotFoundError(f"Cannot find CUDA kernel file: {cu_file}")
        with open(cu_file, "r", encoding='utf-8') as f:
            code = f.read()

        # 使用 self.device_id 获取 compute capability
        cc = self.device.compute_capability()
        arch = f"sm_{cc[0]}{cc[1]}"

        options = [
            "-O3",
            "--use_fast_math",
            "-Xptxas", "-O3",
            "--std=c++14"
        ]

        return SourceModule(
            code,
            no_extern_c=True,
            arch=arch,
            options=options,
        )

    def _allocate_buffers(self):
        inputs, outputs = [], []
        for binding_name in self.engine:
            shape = self.engine.get_tensor_shape(binding_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            if self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem})
                print(f"Input {binding_name}: {shape}")
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
                print(f"Output {binding_name}: {shape}")
        return inputs, outputs

    def _compute_affine_matrix(self, from_wh, to_wh):
        src_w, src_h = from_wh
        dst_w, dst_h = to_wh
        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        scale = min(scale_x, scale_y)

        i2d = np.zeros(6, dtype=np.float32)
        i2d[0] = scale
        i2d[1] = 0.0
        i2d[2] = -scale * src_w * 0.5 + dst_w * 0.5 + scale * 0.5 - 0.5
        i2d[3] = 0.0
        i2d[4] = scale
        i2d[5] = -scale * src_h * 0.5 + dst_h * 0.5 + scale * 0.5 - 0.5

        a, b, tx = i2d[0], i2d[1], i2d[2]
        c, d, ty = i2d[3], i2d[4], i2d[5]
        det = a * d - b * c
        if abs(det) < 1e-12:
            d2i = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        else:
            inv_det = 1.0 / det
            d2i = np.array([
                d * inv_det,
                -b * inv_det,
                (b * ty - d * tx) * inv_det,
                -c * inv_det,
                a * inv_det,
                (c * tx - a * ty) * inv_det
            ], dtype=np.float32)
        return i2d, d2i

    def preprocess(self, image):
        orig_h, orig_w = image.shape[:2]
        _, d2i = self._compute_affine_matrix((orig_w, orig_h), (self.model_w, self.model_h))

        # Upload image to GPU
        image_gpu = cuda.mem_alloc(image.nbytes)
        cuda.memcpy_htod_async(image_gpu, np.ascontiguousarray(image), self.stream)

        # Upload d2i
        cuda.memcpy_htod_async(self.d2i_gpu, d2i, self.stream)

        # Launch warp kernel
        grid = ((self.model_w + 31) // 32, (self.model_h + 31) // 32)
        block = (32, 32, 1)
        self.warp_kernel(
            image_gpu,
            np.int32(image.strides[0]),
            np.int32(orig_w), np.int32(orig_h),
            self.inputs[0]['device'],
            np.int32(self.model_w), np.int32(self.model_h),
            np.uint8(114),
            self.d2i_gpu,
            block=block, grid=grid, stream=self.stream
        )
        image_gpu.free()

    def infer(self, image, conf_thres=0.25, iou_thres=0.45):
        self.context_gpu.push()  # ✅ 绑定当前线程到 GPU 上下文
        try:
            self.preprocess(image)

            # TRT inference
            self.context.set_tensor_address(self.input_name, self.inputs[0]['device'])
            self.context.set_tensor_address(self.output_name, self.outputs[0]['device'])
            self.context.execute_async_v3(stream_handle=self.stream.handle)

            # Postprocess on GPU
            cuda.memset_d32_async(self.parray_gpu, 0, 1, self.stream)  # Reset count

            grid = grid_dims(self.num_boxes)
            self.decode_kernel(
                self.outputs[0]['device'],
                np.int32(self.num_boxes),
                np.int32(self.num_classes),
                np.int32(self.output_c_dim),
                np.float32(conf_thres),
                self.d2i_gpu,
                self.parray_gpu,
                np.int32(MAX_IMAGE_BOXES),
                np.int32(NUM_KEYPOINTS),
                np.int32(NUM_POSE_ELEMENT),
                block=(GPU_BLOCK_THREADS, 1, 1),
                grid=grid,
                stream=self.stream
            )

            grid = grid_dims(MAX_IMAGE_BOXES)
            self.nms_kernel(
                self.parray_gpu,
                np.int32(MAX_IMAGE_BOXES),
                np.float32(iou_thres),
                np.int32(NUM_POSE_ELEMENT),
                block=(GPU_BLOCK_THREADS, 1, 1),
                grid=grid,
                stream=self.stream
            )

            # Copy final result to host
            host_parray = np.empty(1 + MAX_IMAGE_BOXES * NUM_POSE_ELEMENT, dtype=np.float32)
            cuda.memcpy_dtoh_async(host_parray, self.parray_gpu, self.stream)
            self.stream.synchronize()

            count = min(int(host_parray[0]), MAX_IMAGE_BOXES)
            results = []
            for i in range(count):
                box = host_parray[1 + i * NUM_POSE_ELEMENT: 1 + (i + 1) * NUM_POSE_ELEMENT]
                if box[6] > 0.5:  # keep flag
                    keypoints = [(box[j], box[j + 1], box[j + 2]) for j in range(7, NUM_POSE_ELEMENT, 3)]
                    results.append({
                        'bbox': box[:4].tolist(),
                        'score': float(box[4]),
                        'class_id': int(box[5]),
                        'keypoints': keypoints
                    })
            return results

        finally:
            self.context_gpu.pop()  # ✅ 确保上下文恢复

    def warmup(self, num_warmups=3):
        dummy_image = np.random.randint(0, 256, (self.model_h, self.model_w, 3), dtype=np.uint8)
        for _ in range(num_warmups):
            _ = self.infer(dummy_image, conf_thres=0.0, iou_thres=1.0)  # Lowest thresholds to ensure full process


def _hsv2bgr(h: float, s: float, v: float) -> tuple[int, int, int]:
    """
    将 HSV 颜色空间转换为 BGR（OpenCV 格式）。

    Args:
        h: 色调，范围 [0, 1)
        s: 饱和度，范围 [0, 1]
        v: 明度，范围 [0, 1]

    Returns:
        (B, G, R) 三个 uint8 值组成的元组
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q
    else:
        r, g, b = 1.0, 1.0, 1.0

    # 转为 [0, 255] 的整数，并确保不越界
    b = int(round(b * 255))
    g = int(round(g * 255))
    r = int(round(r * 255))
    return (b, g, r)  # OpenCV 是 BGR


def _random_color(id: int) -> tuple[int, int, int]:
    """
    根据 ID 生成确定性的随机颜色（与 C++ 版本行为一致）。

    Args:
        id: 类别或实例 ID

    Returns:
        (B, G, R) 颜色元组
    """
    # 模拟 C++ 中的位运算和取模
    h_plane = ((((id << 2) ^ 0x937151) % 100) / 100.0)
    s_plane = ((((id << 3) ^ 0x315793) % 100) / 100.0)
    return _hsv2bgr(h_plane, s_plane, 1.0)


def draw_detections(image, detections, class_names):
    """
    在图像上绘制检测结果，支持：
    - 普通 bbox（YOLOv8/V5/V7/X）
    - 分类（YOLOv8Cls）
    - 实例分割 mask（YOLOv8Seg）
    - 关键点（YOLOv8Pose）

    Args:
        image: BGR 图像 (H, W, 3) numpy array
        detections: list of dict，每个 dict 至少包含：
            - 'bbox': [x1, y1, x2, y2]
            - 'score': float
            - 'class_id': int
            - 可选: 'mask' -> (Hm, Wm) uint8, 值为 0~255
            - 可选: 'keypoints' -> list of [x, y, score]
        class_names: 类别名称列表
        is_show_mask: 是否显示分割 mask

    Returns:
        绘制后的图像
    """
    # 为每个类别生成固定颜色
    colors = [_random_color(i) for i in range(len(class_names))]

    # 普通检测 / Seg / Pose
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        class_id = det['class_id']
        keypoints = det['keypoints']
        color = colors[class_id]  # (B, G, R)

        # === 绘制边界框 ===
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)

        # === 绘制标签 ===
        label = f"{class_names[class_id]}: {score:.2f}"
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # 背景框
        cv2.rectangle(
            image,
            (int(x1), int(y1 - text_height - baseline)),
            (int(x1 + text_width), int(y1)),
            color,
            -1,
            cv2.LINE_AA
        )
        # 文字
        cv2.putText(
            image, label,
            (int(x1), int(y1 - baseline)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        # === 绘制关键点 ===
        if keypoints is not None:
            for i, (x, y, score) in enumerate(keypoints):
                if score > 0.5:
                    cv2.circle(image, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

    return image


class YOLO:
    def __init__(self, engine_path, names_path):
        self.done_warmup = False
        self.id2name = self._load_names(names_path)
        self.model = TRTYOLO(engine_path)
        self._lock = threading.Lock()
        self.callbacks = defaultdict(list)
        self.vid_writer = {}  # dict of {save_path: video_writer, ...}

    def _load_names(self, names_path):
        with open(names_path, 'r') as f:
            names = [line.strip() for line in f.readlines()]
        return {i: name for i, name in enumerate(names)}

    def _setup_source(self, source, **kwargs):
        vid_stride = kwargs.get('vid_stride', 1)
        stream_buffer = kwargs.get('stream_buffer', False)
        self.dataset = load_inference_source(
            source=source,
            batch=1,
            vid_stride=vid_stride,
            buffer=stream_buffer,
        )
        self.source_type = self.dataset.source_type
        if not getattr(self, "stream", True) and (
                self.source_type.stream
                or self.source_type.screenshot
                or len(self.dataset) > 1000  # many images
                or any(getattr(self.dataset, "video_flag", [False]))
        ):  # videos
            STREAM_WARNING = """
            WARNING ⚠️ inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
            errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

            Example:
                results = model(source=..., stream=True)  # generator of Results objects
                for r in results:
                    boxes = r.boxes  # Boxes object for bbox outputs
                    masks = r.masks  # Masks object for segment masks outputs
                    probs = r.probs  # Class probabilities for classification outputs
            """
            print(STREAM_WARNING)
        self.vid_writer = {}

    def predict(
            self,
            source: str = None,
            stream: bool = False,
            **kwargs
    ):
        return self(source=source, stream=stream, **kwargs)

    def track(
            self,
            source: str = None,
            stream: bool = False,
            persist: bool = False,
            **kwargs
    ):
        if not hasattr(self, "trackers"):
            _register_tracker(self, persist, **kwargs)
        return self(source=source, stream=stream, **kwargs)

    def __call__(self, source=None, stream=False, **kwargs):
        self.stream = stream
        if stream:
            return self._stream_inference(source, **kwargs)
        else:
            return list(self._stream_inference(source, **kwargs))

    def _stream_inference(self, source=None, **kwargs):
        if not self.model:
            raise RuntimeError("模型未初始化")

        with self._lock:
            self._setup_source(source=source)

            # Check save_dir
            save_dir = increment_path(Path(kwargs.get("project", "./runs/infer")),
                                      exist_ok=kwargs.get("exist_ok", False))
            self.save_dir = Path(save_dir)
            if kwargs.get("save", True):
                self.save_dir.mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(3)
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (ops.Profile(), ops.Profile())

            self._run_callbacks("on_predict_start")
            for self.batch in self.dataset:
                self._run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch
                n = len(im0s)
                assert n == 1, f"batch size {n} != 1"

                with profilers[0]:
                    outputs = self.model.infer(im0s[0], kwargs.get("conf", 0.25), kwargs.get("iou", 0.45))
                # 拼接results
                boxes = torch.empty((len(outputs), 6), dtype=torch.float32)
                keypoints = torch.empty((len(outputs), NUM_KEYPOINTS, 3), dtype=torch.float32)
                # f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
                for i, output in enumerate(outputs):
                    boxes[i][:4] = torch.Tensor(output['bbox'])
                    boxes[i][4] = torch.Tensor([output['score']])
                    boxes[i][5] = torch.Tensor([output['class_id']])
                    for j, (x, y, v) in enumerate(output['keypoints']):
                        keypoints[i][j] = torch.Tensor([x, y, v])
                result = Results(
                    orig_img=im0s[0],
                    path=paths[0],
                    names=self.id2name,
                    boxes=boxes,
                    keypoints=keypoints,
                )
                self.results = [result]
                with profilers[1]:
                    self._run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                for i in range(n):
                    self.seen += 1
                    self.results[i].speed = {
                        "infer": profilers[0].dt * 1e3 / n,
                        "track": profilers[1].dt * 1e3 / n,
                    }
                    if kwargs.get('verbose', True) or \
                            kwargs.get('save', True) or \
                            kwargs.get('save_txt', False) or \
                            kwargs.get('show', False):
                        s[i] += self.write_results(i, Path(paths[i]), im0s[0], s, **kwargs)

                # Print batch results
                if kwargs.get('verbose', True):
                    print("\n".join(s))

                self._run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        # Print final results
        if kwargs.get('verbose', True) and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            print(
                f"Speed: %.1fms infer, %.1fms track per image at shape "
                f"{(min(kwargs.get('batch', 1), self.seen), 3, *im0s[0].shape[:2])}" % t
            )
        if kwargs.get("save", True) or kwargs.get('save_txt', False) or kwargs.get('save_crop', False):
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" \
                if kwargs.get('save_txt', False) else ""
            print(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self._run_callbacks("on_predict_end")

    def write_results(self, i, p, im, s, **kwargs):
        """Write inference results to a file or directory."""
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[1:3])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['infer']:.1f}ms"

        # Add predictions to image
        if kwargs.get('save', True) or kwargs.get('show', False):
            self.plotted_img = result.plot(
                line_width=kwargs.get('line_width', None),
                boxes=kwargs.get('show_boxes', True),
                conf=kwargs.get('show_conf', True),
                labels=kwargs.get('show_labels', True),
            )

        # Save results
        if kwargs.get('save_txt', False):
            result.save_txt(f"{self.txt_path}.txt", save_conf=kwargs.get('save_conf', False))
        if kwargs.get('save_crop', False):
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if kwargs.get('show', False):
            self.show(str(p))
        if kwargs.get('save', True):
            self.save_predicted_images(str(self.save_dir / p.name), frame, **kwargs)

        return string

    def save_predicted_images(self, save_path="", frame=0, **kwargs):
        """Save video predictions as mp4 at specified path."""
        im = self.plotted_img

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = f'{save_path.split(".", 1)[0]}_frames/'
            if save_path not in self.vid_writer:  # new video
                if kwargs.get('save_frames', False):
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if kwargs.get('save_frames', False):
                cv2.imwrite(f"{frames_path}{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(str(Path(save_path).with_suffix(".jpg")), im)  # save to JPG for best support

    def show(self, p=""):
        """Display an image in a window using the OpenCV imshow function."""
        im = self.plotted_img
        if platform.system() == "Linux" and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(p, im.shape[1], im.shape[0])  # (width, height)
        cv2.imshow(p, im)
        cv2.waitKey(300 if self.dataset.mode == "image" else 1)  # 1 millisecond

    @property
    def names(self) -> Dict[int, str]:
        return self.id2name

    def _run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def _add_callback(self, event: str, func):
        """Add callback."""
        self.callbacks[event].append(func)


# A mapping of tracker types to corresponding tracker classes
TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}


def _on_predict_start(predictor: YOLO, persist: bool = False, **kwargs):
    if hasattr(predictor, "trackers") and persist:
        return

    tracker = check_yaml(kwargs.get('tracker', 'botsort.yaml'))
    cfg = IterableSimpleNamespace(**YAML.load(tracker))

    if cfg.tracker_type not in {"bytetrack", "botsort"}:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'")

    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
        if predictor.dataset.mode != "stream":  # only need one tracker for other modes.
            break
    predictor.trackers = trackers
    predictor.vid_path = [None] * predictor.dataset.bs  # for determining when to reset tracker on new video


def _on_predict_postprocess_end(predictor: YOLO, persist: bool = False, **kwargs):
    path, im0s = predictor.batch[:2]

    is_stream = predictor.dataset.mode == "stream"
    for i in range(len(im0s)):
        tracker = predictor.trackers[i if is_stream else 0]
        vid_path = predictor.save_dir / Path(path[i]).name
        if not persist and predictor.vid_path[i if is_stream else 0] != vid_path:
            tracker.reset()
            predictor.vid_path[i if is_stream else 0] = vid_path

        det = predictor.results[i].boxes.cpu().numpy()
        if len(det) == 0:
            continue
        tracks = tracker.update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]

        update_args = {"boxes": torch.as_tensor(tracks[:, :-1])}
        predictor.results[i].update(**update_args)


def _register_tracker(model: YOLO, persist: bool, **kwargs):
    model._add_callback("on_predict_start", partial(_on_predict_start, persist=persist, **kwargs))
    model._add_callback("on_predict_postprocess_end", partial(_on_predict_postprocess_end, persist=persist, **kwargs))
