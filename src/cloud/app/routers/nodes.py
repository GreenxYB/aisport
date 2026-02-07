from fastapi import APIRouter
from ..models.schemas import ViolationReport, FinishReport

router = APIRouter()


@router.post("/reports/violation")
def receive_violation(report: ViolationReport):
    # TODO: persist and fan-out alerts
    return {"status": "accepted", "session_id": report.session_id}


@router.post("/reports/finish")
def receive_finish(report: FinishReport):
    # TODO: persist results and trigger scoring pipeline
    return {"status": "accepted", "session_id": report.session_id, "count": len(report.results)}
