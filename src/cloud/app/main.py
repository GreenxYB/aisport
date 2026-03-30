import logging
import os
from pathlib import Path

from fastapi import FastAPI

from .routers import sessions, nodes, health
from .routers.sessions import get_orchestrator_service, get_service
from .services.node_connection_manager import get_node_manager


def _setup_logging() -> None:
    root = Path(__file__).resolve().parents[3]
    log_dir = root / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir / "cloud.log"

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
    app = FastAPI(title="AI Sport Cloud", version="0.1.0")
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
    app.include_router(nodes.router, prefix="/nodes", tags=["nodes"])

    @app.on_event("startup")
    async def startup_event() -> None:
        orchestrator = get_orchestrator_service(get_service(), get_node_manager())
        await orchestrator.start()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        orchestrator = get_orchestrator_service(get_service(), get_node_manager())
        await orchestrator.stop()

    return app


app = create_app()


@app.get("/")
def root():
    return {"status": "ok", "service": "cloud"}
