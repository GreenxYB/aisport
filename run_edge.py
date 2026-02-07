import os
import sys

# Ensure src is on path when uvicorn reload spawns subprocesses (Windows spawn mode)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from edge.app.main import app  # noqa: E402


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run_edge:app",
        host="0.0.0.0",
        port=8100,
        reload=True,
    )
