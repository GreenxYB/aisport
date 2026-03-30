from __future__ import annotations

from typing import Protocol, List, Dict

import numpy as np


class FrameAlgorithm(Protocol):
    """帧级算法接口协议。

    约定输入为单帧与时间戳，输出标准事件列表。
    """

    def process(self, frame: np.ndarray, ts_ms: float) -> List[Dict]:
        ...
