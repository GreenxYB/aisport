import argparse
import json
import urllib.request
from pathlib import Path


def send_event(base_url: str, event: dict, timeout: float) -> bool:
    msg_type = event.get("msg_type")
    if msg_type == "VIOLATION_EVENT":
        endpoint = "violation"
    elif msg_type == "FINISH_REPORT":
        endpoint = "finish"
    else:
        return True
    url = f"{base_url.rstrip('/')}/{endpoint}"
    data = json.dumps(event).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay failed edge events to cloud")
    parser.add_argument(
        "--file",
        default=str(Path("logs") / "events_failed.jsonl"),
        help="Path to events_failed.jsonl",
    )
    parser.add_argument(
        "--archive",
        default=str(Path("logs") / "events_failed.archived.jsonl"),
        help="Archive file for successfully replayed events",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/nodes/reports",
        help="Cloud report base URL",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="HTTP timeout seconds")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return

    ok = 0
    fail = 0
    ok_lines = []
    fail_lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            event = payload.get("event", payload)
        except Exception:
            fail += 1
            fail_lines.append(line)
            continue
        if send_event(args.base_url, event, args.timeout):
            ok += 1
            ok_lines.append(line)
        else:
            fail += 1
            fail_lines.append(line)

    # rewrite failed file to keep only unreplayed items
    if ok_lines:
        archive = Path(args.archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("a", encoding="utf-8") as f:
            for line in ok_lines:
                f.write(line + "\n")
    with path.open("w", encoding="utf-8") as f:
        for line in fail_lines:
            f.write(line + "\n")

    print(f"replay done: ok={ok} fail={fail} archived={len(ok_lines)}")


if __name__ == "__main__":
    main()
