from __future__ import annotations

from typing import Protocol, List, Dict

import numpy as np


class FrameAlgorithm(Protocol):
    def process(self, frame: np.ndarray, ts_ms: float) -> List[Dict]:
        ...
