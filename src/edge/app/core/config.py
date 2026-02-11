from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.edge", env_file_encoding="utf-8")

    env: str = Field("dev", description="Runtime environment")
    node_id: int = Field(1, description="Edge node numeric ID")
    command_poll_interval_ms: int = 500
    cloud_api_base: str = Field("http://localhost:8000", description="Cloud endpoint")
    camera_device: str = Field("0", description="Camera device index/path or RTSP URL")
    rtsp_url: str = Field("", description="Optional RTSP URL; overrides camera_device when set")
    simulate_camera: bool = Field(
        True,
        description="Use synthetic frames when no real camera is available; set false to use real device",
    )
    auto_start_capture: bool = Field(
        True, description="Start camera capture on service startup"
    )
    display_preview: bool = Field(
        True,
        description="Show cv2.imshow live preview by default (requires GUI session)",
    )
    display_mirror: bool = Field(
        True, description="Mirror preview display and snapshot horizontally"
    )
    capture_fps: int = Field(15, description="Target capture FPS")
    capture_width: int = Field(640, description="Capture width")
    capture_height: int = Field(480, description="Capture height")
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
    # Baidu face search
    baidu_app_id: str = Field("", description="Baidu AIP App ID")
    baidu_api_key: str = Field("", description="Baidu AIP API Key")
    baidu_secret_key: str = Field("", description="Baidu AIP Secret Key")
    baidu_group_id: str = Field("default", description="Baidu face group id")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
