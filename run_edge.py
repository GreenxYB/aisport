import os
import sys
import atexit

# Ensure src is on path when uvicorn reload spawns subprocesses (Windows spawn mode)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 全局 CUDA 上下文管理
_cuda_context = None

# 清理 CUDA 上下文的函数
def cleanup_cuda():
    global _cuda_context
    if _cuda_context is not None:
        try:
            _cuda_context.pop()
            _cuda_context = None
            print("CUDA 上下文清理完成")
        except Exception as e:
            print(f"清理 CUDA 上下文时出错: {e}")

# 注册退出处理程序
atexit.register(cleanup_cuda)

from edge.app.main import app  # noqa: E402


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "run_edge:app",
        host="0.0.0.0",
        port=8100,
        reload=False,
    )
