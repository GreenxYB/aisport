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
        False,
        description="For local debug: show cv2.imshow live preview (requires GUI session)",
    )
    capture_fps: int = Field(15, description="Target capture FPS")
    capture_width: int = Field(640, description="Capture width")
    capture_height: int = Field(480, description="Capture height")
    model_dir: str = Field("./data/models", description="Model directory")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
