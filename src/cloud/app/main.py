from fastapi import FastAPI

from .routers import sessions, nodes, health


def create_app() -> FastAPI:
    app = FastAPI(title="AI Sport Cloud", version="0.1.0")
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
    app.include_router(nodes.router, prefix="/nodes", tags=["nodes"])
    return app


app = create_app()


@app.get("/")
def root():
    return {"status": "ok", "service": "cloud"}
