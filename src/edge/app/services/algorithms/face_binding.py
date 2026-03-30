from typing import List, Dict
import base64
import logging
import time

import cv2
import numpy as np

from ...core.config import get_settings

try:
    from aip import AipFace  # type: ignore
except Exception:  # pragma: no cover
    AipFace = None


logger = logging.getLogger("edge.face")


class FaceBindingAlgo:
    """人脸绑定算法封装。

    基于百度人脸检索接口，将候选目标映射到学员身份，输出 ID_REPORT。
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = None
        self._attempts_by_key: Dict[str, int] = {}
        self._confirmed_keys: set[str] = set()
        self._session_id: str | None = None
        self._last_search_ts: float = 0.0
        if (
            AipFace
            and self.settings.baidu_app_id
            and self.settings.baidu_api_key
            and self.settings.baidu_secret_key
        ):
            self.client = AipFace(
                self.settings.baidu_app_id,
                self.settings.baidu_api_key,
                self.settings.baidu_secret_key,
            )
        elif not AipFace:
            logger.warning("baidu-aip not installed; face binding disabled")

    def reset(self, session_id: str | None = None) -> None:
        """重置会话级缓存（尝试次数/已确认目标）。"""
        self._attempts_by_key.clear()
        self._confirmed_keys.clear()
        self._session_id = session_id

    def bind_session(self, session_id: str | None) -> None:
        """会话切换时自动清理历史识别状态。"""
        normalized = session_id or None
        if normalized != self._session_id:
            self.reset(normalized)

    def process(self, frame: np.ndarray, ts_ms: float) -> List[Dict]:
        """兼容旧接口：对整帧做人脸检索并输出 ID_REPORT。"""
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

    def process_candidates(self, candidates: List[Dict], ts_ms: float) -> List[Dict]:
        """对赛道候选目标逐个做人脸检索，并生成聚合 ID_REPORT。"""
        if not self.client:
            return []
        results: List[Dict] = []
        for candidate in candidates:
            binding_key = str(candidate.get("binding_key") or "")
            # 已确认目标不再重复识别，避免冗余请求。
            if binding_key and binding_key in self._confirmed_keys:
                continue
            attempts = self._attempts_by_key.get(binding_key, 0) if binding_key else 0
            # 单目标最多尝试 N 次，控制外部 API 调用量。
            if binding_key and attempts >= max(
                1, int(self.settings.face_search_max_attempts)
            ):
                continue
            lane = candidate.get("lane")
            image = candidate.get("image")
            if image is None:
                continue
            if binding_key:
                self._attempts_by_key[binding_key] = attempts + 1
            face_base64 = self._frame_to_base64(image)
            if not face_base64:
                continue
            matches = self.search_face_baidu(face_base64, self.settings.baidu_group_id)
            if not matches:
                continue
            top = matches[0]
            if binding_key:
                self._confirmed_keys.add(binding_key)
            enriched = {
                **top,
                "lane": int(lane) if isinstance(lane, int) else lane,
                "bbox": candidate.get("bbox"),
                "track_id": candidate.get("track_id"),
                "attempt": self._attempts_by_key.get(binding_key, 1)
                if binding_key
                else 1,
            }
            results.append(enriched)
        if not results:
            return []
        return [{"msg_type": "ID_REPORT", "timestamp": int(ts_ms), "data": results}]

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """将 BGR 图像编码为百度接口所需 BASE64 JPEG。"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ok, buffer = cv2.imencode(".jpg", rgb)
            if not ok:
                return ""
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as exc:
            logger.error("face encode error: %s", exc)
            return ""

    def search_face_baidu(
        self, face_base64: str, group_id: str = "default"
    ) -> List[Dict]:
        """调用百度人脸检索并标准化返回字段。"""
        try:
            self._respect_rate_limit()
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
            elif res.get("error_code") != 0:
                logger.warning(
                    "face search failed error_code=%s error_msg=%s",
                    res.get("error_code"),
                    res.get("error_msg"),
                )
            return res_list
        except Exception as exc:
            logger.error("face search error: %s", exc)
            return []

    def _respect_rate_limit(self) -> None:
        """限制连续调用速率，避免触发外部接口 QPS 限额。"""
        min_interval = float(
            getattr(self.settings, "face_search_min_interval_sec", 0.5) or 0.5
        )
        now = time.monotonic()
        delta = now - self._last_search_ts
        if delta < min_interval:
            time.sleep(min_interval - delta)
        self._last_search_ts = time.monotonic()
