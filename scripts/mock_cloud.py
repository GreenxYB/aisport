from pathlib import Path
import json

from fastapi import FastAPI, Request

app = FastAPI(title="Mock Cloud Receiver", version="0.1.0")

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "cloud_received.jsonl"


def _append(payload: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


@app.post("/nodes/reports/violation")
async def violation(req: Request):
    payload = await req.json()
    _append(payload)
    return {"status": "ok", "type": "violation"}


@app.post("/nodes/reports/finish")
async def finish(req: Request):
    payload = await req.json()
    _append(payload)
    return {"status": "ok", "type": "finish"}
