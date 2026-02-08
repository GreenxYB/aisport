from typing import List, Dict

import numpy as np


class ViolationAlgo:
    def __init__(self):
        self._counter = 0

    def process(self, frame: np.ndarray, ts_ms: float) -> List[Dict]:
        # Placeholder: no real detection
        self._counter += 1
        return []
