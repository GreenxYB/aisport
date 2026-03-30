from __future__ import annotations

from typing import Any


class NullPublisher:
    """发布器空实现。

    默认用于未接入 websocket/消息总线时的兜底，
    保持调用方逻辑统一（始终可调用 publish）。
    """

    def publish(self, message: dict[str, Any]) -> bool:
        """空实现固定返回 False，表示消息未实际发送。"""
        return False
