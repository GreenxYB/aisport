import logging
import os
from pathlib import Path

from fastapi import FastAPI

from .routers import health, commands, status, preview, face


def _setup_logging() -> None:
    root = Path(__file__).resolve().parents[4]
    log_dir = root / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / "edge.log"

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )


def create_app() -> FastAPI:
    _setup_logging()
    app = FastAPI(title="AI Sport Edge Node", version="0.1.0")
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(commands.router, prefix="/commands", tags=["commands"])
    app.include_router(status.router, prefix="/status", tags=["status"])
    app.include_router(preview.router, prefix="/preview", tags=["preview"])
    app.include_router(face.router, prefix="/face", tags=["face"])

    @app.on_event("startup")
    def _startup() -> None:
        # Initialize handler on startup to auto-open camera
        from .routers.commands import get_handler

        get_handler()

    return app


app = create_app()


@app.get("/")
def root():
    return {"status": "ok", "service": "edge"}
