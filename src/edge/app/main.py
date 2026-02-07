from fastapi import FastAPI

from .routers import health, commands, status


def create_app() -> FastAPI:
    app = FastAPI(title="AI Sport Edge Node", version="0.1.0")
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(commands.router, prefix="/commands", tags=["commands"])
    app.include_router(status.router, prefix="/status", tags=["status"])
    return app


app = create_app()


@app.get("/")
def root():
    return {"status": "ok", "service": "edge"}
