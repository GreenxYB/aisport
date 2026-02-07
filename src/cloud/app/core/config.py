from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    env: str = Field("dev", description="Runtime environment")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    db_url: str = Field(
        "sqlite+aiosqlite:///./data/cloud.db",
        description="SQLAlchemy compatible database URL",
    )
    broker_url: str = Field(
        "redis://localhost:6379/0", description="Message broker for node commands"
    )
    session_time_skew_ms: int = Field(
        50, description="Allowed absolute timestamp skew between cloud and edge nodes"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
