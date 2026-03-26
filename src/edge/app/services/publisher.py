from __future__ import annotations

from typing import Any


class NullPublisher:
    def publish(self, message: dict[str, Any]) -> bool:
        return False
