from fastapi import APIRouter

router = APIRouter()


@router.get("/live")
def live():
    return {"status": "live"}


@router.get("/ready")
def ready():
    return {"status": "ready"}
