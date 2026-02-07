from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    env: str = Field("dev", description="Runtime environment")
    node_id: int = Field(1, description="Edge node numeric ID")
    command_poll_interval_ms: int = 500
    cloud_api_base: str = Field("http://localhost:8000", description="Cloud endpoint")
    camera_device: str = Field("/dev/video0", description="Camera device path")
    model_dir: str = Field("./data/models", description="Model directory")

    class Config:
        env_file = ".env.edge"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
