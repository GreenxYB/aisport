from typing import List, Dict
import base64
import logging

import cv2
import numpy as np

from ...core.config import get_settings

try:
    from aip import AipFace  # type: ignore
except Exception:  # pragma: no cover
    AipFace = None


logger = logging.getLogger("edge.face")


class FaceBindingAlgo:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = None
        if AipFace and self.settings.baidu_app_id and self.settings.baidu_api_key and self.settings.baidu_secret_key:
            self.client = AipFace(
                self.settings.baidu_app_id,
                self.settings.baidu_api_key,
                self.settings.baidu_secret_key,
            )
        elif not AipFace:
            logger.warning("baidu-aip not installed; face binding disabled")

    def process(self, frame: np.ndarray, ts_ms: float) -> List[Dict]:
        if not self.client:
            return []
        face_base64 = self._frame_to_base64(frame)
        if not face_base64:
            return []
        results = self.search_face_baidu(face_base64, self.settings.baidu_group_id)
        if not results:
            return []
        event = {
            "msg_type": "ID_REPORT",
            "timestamp": int(ts_ms),
            "data": results,
        }
        return [event]

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ok, buffer = cv2.imencode(".jpg", rgb)
            if not ok:
                return ""
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as exc:
            logger.error("face encode error: %s", exc)
            return ""

    def search_face_baidu(self, face_base64: str, group_id: str = "default") -> List[Dict]:
        try:
            res_list: List[Dict] = []
            res = self.client.search(face_base64, "BASE64", group_id)
            if res.get("error_code") == 0 and res.get("result", {}).get("user_list"):
                user = res["result"]["user_list"][0]
                res_list.append(
                    {
                        "face_token": res.get("result", {}).get("face_token"),
                        "confidence": user.get("score"),
                        "student_id": user.get("user_id"),
                        "name": user.get("user_info"),
                    }
                )
            return res_list
        except Exception as exc:
            logger.error("face search error: %s", exc)
            return []
