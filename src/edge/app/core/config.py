from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.edge", env_file_encoding="utf-8")

    env: str = Field("dev", description="Runtime environment")
    node_id: int = Field(1, description="Edge node numeric ID")
    command_poll_interval_ms: int = 500
    cloud_api_base: str = Field("http://localhost:8000", description="Cloud endpoint")
    cloud_ws_url: str = Field("ws://localhost:8000/nodes/ws", description="Cloud websocket endpoint")
    camera_device: str = Field("0", description="Camera device index/path or RTSP URL")
    rtsp_url: str = Field("", description="Optional RTSP URL; overrides camera_device when set")
    simulate_camera: bool = Field(
        True,
        description="Use synthetic frames when no real camera is available; set false to use real device",
    )
    auto_start_capture: bool = Field(
        True, description="Start camera capture on service startup"
    )
    ws_enabled: bool = Field(False, description="Enable persistent websocket link to cloud")
    node_role: str = Field("START", description="Node role: START/FINISH/MID/ALL_IN_ONE")
    site_id: str = Field("local-dev", description="Logical site identifier")
    node_capabilities: str = Field("camera,speaker", description="Comma separated capability list")
    ws_reconnect_interval_sec: float = Field(3.0, description="Websocket reconnect interval seconds")
    ws_status_interval_sec: float = Field(2.0, description="Periodic status report interval seconds")
    display_preview: bool = Field(
        True,
        description="Show cv2.imshow live preview by default (requires GUI session)",
    )
    display_start_line: bool = Field(
        True, description="Draw start line on preview/snapshot for debugging"
    )
    display_mirror: bool = Field(
        False, description="Mirror preview display and snapshot horizontally"
    )
    capture_fps: int = Field(30, description="Target capture FPS")
    capture_width: int = Field(1280, description="Capture width")
    capture_height: int = Field(640, description="Capture height")
    model_dir: str = Field("./data/models", description="Model directory")
    simulate_events: bool = Field(
        True, description="Generate simulated events during monitoring"
    )
    event_interval_sec: float = Field(2.0, description="Simulated event interval seconds")
    simulate_finish_reports: bool = Field(
        True, description="Generate simulated finish reports during monitoring"
    )
    finish_interval_sec: float = Field(8.0, description="Simulated finish report interval seconds")
    report_enabled: bool = Field(
        False, description="Enable HTTP reporting of events to cloud service"
    )
    report_base_url: str = Field(
        "http://localhost:8000/nodes/reports",
        description="Cloud report base URL (violation/finish endpoints appended)",
    )
    report_timeout_sec: float = Field(2.0, description="HTTP report timeout seconds")
    report_retry_enabled: bool = Field(
        True, description="Retry failed reports in background"
    )
    report_retry_interval_sec: float = Field(
        3.0, description="Retry interval for failed reports"
    )
    report_retry_max: int = Field(5, description="Max retry attempts per event")
    algo_enabled: bool = Field(True, description="Enable algorithm runner")
    algo_target_fps: int = Field(5, description="Target FPS for algorithm processing")
    algo_log_path: str = Field("logs/alg_events.jsonl", description="Algorithm event log")
    yolo_backend: str = Field("pt", description="YOLO backend: pt or trt")
    yolo_engine_path: str = Field("yolo11n-pose-fp16.engine", description="YOLO TRT engine path")
    yolo_names_path: str = Field("pose.names", description="YOLO class names file")
    yolo_pt_path: str = Field("yolo11n-pose.pt", description="YOLO .pt path for ultralytics")
    yolo_conf_thres: float = Field(0.6, description="YOLO confidence threshold")
    yolo_iou_thres: float = Field(0.45, description="YOLO NMS IoU threshold")
    yolo_imgsz: int = Field(640, description="YOLO inference image size")
    start_line_y: int = Field(480, description="Start line Y for false start (based on 640px height)")
    finish_line_y: int = Field(520, description="Finish line Y (based on 640px height)")
    kps_conf_thres: float = Field(0.5, description="Keypoint confidence threshold")
    toe_proxy_scale: float = Field(
        0.45, description="Toe proxy extrapolation factor from knee->ankle direction"
    )
    # Unified visualization style for service preview and scripts
    viz_line_color: str = Field("0,0,255", description="Primary line color in B,G,R")
    viz_ready_color: str = Field("0,255,0", description="READY text color in B,G,R")
    viz_alert_color: str = Field("0,0,255", description="Alert color in B,G,R")
    viz_countdown_color: str = Field("0,255,255", description="Countdown/timer color in B,G,R")
    viz_box_color: str = Field("0,255,0", description="Debug bbox color in B,G,R")
    viz_toe_ankle_color: str = Field("255,0,0", description="Ankle point color in B,G,R")
    viz_toe_color: str = Field("0,255,255", description="Toe proxy point/link color in B,G,R")
    viz_hud_font_scale: float = Field(1.0, description="HUD font scale")
    viz_hud_font_thickness: int = Field(2, description="HUD font thickness")
    viz_line_thickness: int = Field(2, description="Primary line thickness")
    viz_box_thickness: int = Field(1, description="Debug bbox thickness")
    viz_toe_ankle_radius: int = Field(3, description="Ankle point radius")
    viz_toe_radius: int = Field(4, description="Toe proxy point radius")
    viz_toe_link_thickness: int = Field(1, description="Toe proxy link thickness")
    # Baidu face search
    baidu_app_id: str = Field("", description="Baidu AIP App ID")
    baidu_api_key: str = Field("", description="Baidu AIP API Key")
    baidu_secret_key: str = Field("", description="Baidu AIP Secret Key")
    baidu_group_id: str = Field("default", description="Baidu face group id")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
