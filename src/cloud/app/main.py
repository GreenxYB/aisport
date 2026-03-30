from fastapi import FastAPI

from .routers import sessions, nodes, health
from .routers.sessions import get_orchestrator_service, get_service
from .services.node_connection_manager import get_node_manager


def create_app() -> FastAPI:
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
