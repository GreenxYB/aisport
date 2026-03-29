import json
import logging
import time
from pathlib import Path
from typing import Callable

from fastapi import HTTPException

from common.protocol import CommandPayload, NodeStatusReport
from ..core.config import get_settings
from ..core.state import NodePhase, NodeState
from .algorithms.lane_layout import binding_target_lanes, inspect_lane_layout
from .event_simulator import EventSimulator
from .algorithms import AlgorithmRunner
from .publisher import NullPublisher

try:
    from .pipeline import EdgePipeline
except Exception as exc:  # pragma: no cover
    EdgePipeline = None
    PIPELINE_IMPORT_ERROR = exc
else:
    PIPELINE_IMPORT_ERROR = None


class _NoopPipeline:
    def __init__(self):
        self.running = False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def snapshot_jpeg(self) -> bytes | None:
        return None

    def last_encode_error(self) -> str | None:
        return None


class CommandHandler:
    """
    鍛戒护澶勭悊鍣?- 澶勭悊鏉ヨ嚜浜戠鐨勬帶鍒跺懡浠?

    璐熻矗绠＄悊杈圭紭璁惧鐨勭敓鍛藉懆鏈?
    - 鍒濆鍖栦細璇?(CMD_INIT)
    - 鍚屾缁戝畾淇℃伅 (CMD_BINDING_SYNC)
    - 寮€濮嬬洃鎺?(CMD_START_MONITOR)
    - 鍋滄鐩戞帶 (CMD_STOP)
    - 蹇冭烦妫€娴?(CMD_HEARTBEAT)
    """

    def __init__(self):
        """鍒濆鍖栧懡浠ゅ鐞嗗櫒"""
        self.settings = get_settings()
        # 鐘舵€佹枃浠惰矾寰?- 鐢ㄤ簬鎸佷箙鍖栬妭鐐圭姸鎬?
        self.state_file = Path(__file__).resolve().parents[4] / "logs" / "state.json"
        # 鍔犺浇鎴栧垱寤哄垵濮嬬姸鎬?
        self.state = self._load_state()
        self.logger = logging.getLogger("edge.command")
        self.publisher = NullPublisher()

        # 鍒濆鍖栦簨浠舵ā鎷熷櫒 - 鐢ㄤ簬娴嬭瘯鐢熸垚妯℃嫙浜嬩欢
        self.event_sim = EventSimulator(self.state, publisher=self.publisher)
        # 鍒濆鍖栫畻娉曡繍琛屽櫒 - 澶勭悊瑙嗛甯у苟妫€娴嬩簨浠?
        self.algo = AlgorithmRunner(self.state, publisher=self.publisher)

        # 浣跨敤 EdgePipeline 杩涜澶氱嚎绋嬭棰戝鐞?
        # 鍖呭惈: 瑙嗛閲囬泦 -> YOLO鎺ㄧ悊 -> 鐩爣璺熻釜 -> 涓氬姟閫昏緫
        if EdgePipeline is None:
            self.pipeline = _NoopPipeline()
            self.logger.warning("EdgePipeline unavailable, falling back to no-op pipeline: %s", PIPELINE_IMPORT_ERROR)
        else:
            self.pipeline = EdgePipeline(algo_runner=self.algo)

        # 濡傛灉閰嶇疆浜嗚嚜鍔ㄥ惎鍔ㄩ噰闆?鍒欏惎鍔ㄨ棰戝鐞嗙閬?
        if self.settings.auto_start_capture:
            self.pipeline.start()
            self.state.capture_running = self.pipeline.running
            self.logger.info(f"鑷姩鍚姩閲囬泦: 绠￠亾杩愯鐘舵€?{self.pipeline.running}")

        # 瀹氫箟鍏佽鐨勫懡浠ら泦鍚?
        self.allowed_cmds = {
            "CMD_INIT",  # 鍒濆鍖栦細璇?
            "CMD_BINDING_SYNC",  # 鍚屾杩愬姩鍛樼粦瀹氫俊鎭?
            "CMD_START_MONITOR",  # 寮€濮嬬洃鎺?妫€娴?
            "CMD_STOP",  # 鍋滄鐩戞帶
            "CMD_RESET_ROUND",  # 杩濊鍚庨噸缃綋鍓嶈疆娆?
            "CMD_HEARTBEAT",  # 蹇冭烦妫€娴?
        }
        # 鍛戒护鍒嗗彂鏄犲皠琛?- 灏嗗懡浠ゆ槧灏勫埌瀵瑰簲鐨勫鐞嗘柟娉?
        self._dispatch_map: dict[str, Callable[[CommandPayload], None]] = {
            "CMD_INIT": self._handle_init,
            "CMD_BINDING_SYNC": self._handle_binding_sync,
            "CMD_START_MONITOR": self._handle_start_monitor,
            "CMD_STOP": self._handle_stop,
            "CMD_RESET_ROUND": self._handle_reset_round,
            "CMD_HEARTBEAT": self._handle_heartbeat,
        }

    def handle(self, payload: CommandPayload) -> None:
        """
        澶勭悊鎺ユ敹鍒扮殑鍛戒护

        Args:
            payload: 鍛戒护璐熻浇,鍖呭惈鍛戒护绫诲瀷銆佷細璇滻D銆佽妭鐐笽D鍜岄厤缃俊鎭?

        Raises:
            HTTPException: 褰撳懡浠や笉鏀寔鏃惰繑鍥?00閿欒
        """
        started = time.time()
        # 妫€鏌ュ懡浠ゆ槸鍚﹀湪鍏佽鍒楄〃涓?
        if payload.cmd not in self.allowed_cmds:
            self.logger.warning("鏈煡鍛戒护 %s", payload.cmd)
            raise HTTPException(status_code=400, detail="涓嶆敮鎸佺殑鍛戒护")

        # 鑾峰彇瀵瑰簲鐨勫懡浠ゅ鐞嗗櫒
        handler = self._dispatch_map[payload.cmd]
        self.logger.info(
            "cmd=%s session=%s node=%s summary=%s",
            payload.cmd,
            payload.session_id,
            payload.node_id,
            self._summarize_command(payload),
        )

        # 鎵ц鍛戒护澶勭悊
        handler(payload)
        # 鎸佷箙鍖栫姸鎬佸埌鏂囦欢
        self._persist_state()

        # 璁板綍澶勭悊鑰楁椂
        elapsed_ms = int((time.time() - started) * 1000)
        self.logger.info(
            "handled cmd=%s phase=%s elapsed_ms=%s",
            payload.cmd,
            self.state.phase,
            elapsed_ms,
        )

    # ==================== 鍛戒护澶勭悊鍣?====================

    def _handle_init(self, payload: CommandPayload) -> None:
        """
        澶勭悊鍒濆鍖栧懡浠?- 閲嶇疆鎵€鏈夌姸鎬?

        褰撲細璇濆彉鍖栨椂,閲嶇疆鎵€鏈夌姸鎬佷互寮€濮嬫柊鐨勬瘮璧涗細璇?
        """
        # 璁剧疆浼氳瘽ID
        self.state.session_id = payload.session_id
        # 杩涘叆缁戝畾闃舵 - 绛夊緟杩愬姩鍛樹俊鎭粦瀹?
        self.state.phase = NodePhase.BINDING
        # 淇濆瓨閰嶇疆淇℃伅
        self.state.config = payload.config or {}
        # 娓呯┖杩愬姩鍛樼粦瀹氬垪琛?
        self.state.bindings = []
        # 閲嶇疆鍚勭鐘舵€佸瓧娈?
        self.state.expected_start_time = None
        self.state.stop_reason = None
        self.state.events_generated = 0
        self.state.last_event_ts = None
        self.state.finish_reports_generated = 0
        self.state.last_finish_ts = None
        self.state.binding_confirmed_students = []
        self.state.binding_confirmed_lanes = []
        self.state.binding_assignments = []
        self.state.binding_confirmed_at_ms = None
        self.state.last_face_result = None
        self.state.last_face_ts = None
        self.algo.reset_binding_runtime()
        # 鍋滄浜嬩欢妯℃嫙
        self.event_sim.stop()
        self._touch(payload.cmd)

    def _handle_binding_sync(self, payload: CommandPayload) -> None:
        """
        澶勭悊缁戝畾鍚屾鍛戒护 - 鍚屾杩愬姩鍛樹笌璺戦亾鐨勭粦瀹氫俊鎭?

        鍦ㄦ瘮璧涘紑濮嬪墠,灏嗚繍鍔ㄥ憳ID涓庤窇閬撶紪鍙疯繘琛岀粦瀹?
        """
        # 楠岃瘉浼氳瘽涓€鑷存€?
        self._ensure_same_session(payload)
        # 楠岃瘉褰撳墠闃舵蹇呴』鏄疊INDING闃舵
        self._ensure_phase([NodePhase.BINDING])

        # 鑾峰彇缁戝畾淇℃伅
        bindings = payload.config.get("bindings") if payload.config else None
        self.state.bindings = bindings or []
        # 淇濇寔鍦˙INDING闃舵,绛夊緟寮€濮嬬洃鎺у懡浠?
        self.state.phase = NodePhase.BINDING
        self.algo.reset_binding_runtime()

        # 濡傛灉鍚敤浜嗕簨浠舵ā鎷?鍚姩妯℃嫙鍣?
        if self.settings.simulate_events:
            lane_count = int(self.state.config.get("lane_count", 1) or 1)
            self.event_sim.start(self.state.session_id or "UNKNOWN", lane_count)
        self._touch(payload.cmd)

    def _handle_start_monitor(self, payload: CommandPayload) -> None:
        """
        澶勭悊寮€濮嬬洃鎺у懡浠?- 鍚姩瑙嗛閲囬泦鍜岀畻娉曟娴?

        杩涘叆MONITORING闃舵,寮€濮嬪疄鏃舵娴嬭繚瑙勮涓哄拰鍐茬嚎浜嬩欢
        """
        # 楠岃瘉浼氳瘽涓€鑷存€?
        self._ensure_same_session(payload)
        # 楠岃瘉褰撳墠闃舵蹇呴』鏄疊INDING闃舵(鍙兘浠庣粦瀹氶樁娈佃繘鍏ョ洃鎺ч樁娈?
        self._ensure_phase([NodePhase.BINDING])

        # 鍒囨崲鍒扮洃鎺ч樁娈?
        self.state.phase = NodePhase.MONITORING
        # 淇濆瓨棰勮寮€濮嬫椂闂?
        self.state.expected_start_time = (payload.config or {}).get(
            "expected_start_time"
        )
        self.state.config["ready_ts"] = int(time.time() * 1000)
        # 婵€娲昏窡韪姛鑳?
        self.state.config["tracking_active"] = (payload.config or {}).get(
            "tracking_active", True
        )
        self.state.config["countdown_seconds"] = (payload.config or {}).get(
            "countdown_seconds", 3
        )

        # 濡傛灉瑙嗛绠￠亾鏈繍琛?鍒欏惎鍔ㄥ畠
        if not self.pipeline.running:
            try:
                self.pipeline.start()
            except Exception as exc:
                self.state.capture_error = str(exc)
                raise HTTPException(status_code=503, detail="鎽勫儚澶存墦寮€澶辫触")

        # 鏇存柊閲囬泦鐘舵€?
        self.state.capture_running = self.pipeline.running
        self.state.capture_error = None

        # 濡傛灉鍚敤浜嗕簨浠舵ā鎷?鍚姩妯℃嫙鍣?
        if self.settings.simulate_events:
            lane_count = int(self.state.config.get("lane_count", 1) or 1)
            self.event_sim.start(self.state.session_id or "UNKNOWN", lane_count)
        self._touch(payload.cmd)

    def _handle_stop(self, payload: CommandPayload) -> None:
        """
        澶勭悊鍋滄鍛戒护 - 鍋滄鐩戞帶骞舵竻鐞嗚祫婧?

        杩涘叆STOPPED闃舵,鍋滄瑙嗛閲囬泦鍜屼簨浠剁敓鎴?
        """
        # 楠岃瘉浼氳瘽涓€鑷存€?
        self._ensure_same_session(payload)
        # 鍒囨崲鍒板仠姝㈤樁娈?
        self.state.phase = NodePhase.STOPPED
        # 淇濆瓨鍋滄鍘熷洜
        self.state.stop_reason = (payload.config or {}).get("reason")
        reason = str(self.state.stop_reason or "")
        # 鍖哄垎鈥滃仠姝笟鍔♀€濅笌鈥滃叧闂瑙?閲囬泦鈥濓細
        # - 榛樿鎯呭喌涓嬶細鍋滄涓氬姟骞跺仠姝㈢閬擄紙涓庡巻鍙茶涓轰竴鑷达級
        # - BINDING_TIMEOUT锛氶粯璁や粎鍋滄涓氬姟锛屼繚鐣欓瑙堢獥鍙ｇ敤浜庣幇鍦烘帓鏌?
        # - 鍙€氳繃 payload.config.stop_capture 鏄惧紡瑕嗙洊
        stop_capture = bool(
            (payload.config or {}).get(
                "stop_capture",
                False if reason == "BINDING_TIMEOUT" else True,
            )
        )
        if reason == "BINDING_TIMEOUT":
            try:
                status = self.build_status_report().data
                self.logger.warning(
                    "binding timeout diagnostics session=%s stage=%s ready=%s required=%s "
                    "configured_lanes=%s observed_lanes=%s confirmed_lanes=%s pending_lanes=%s "
                    "last_face_ts=%s last_lane_obs_ts=%s camera_ready=%s",
                    self.state.session_id,
                    status.get("session_stage"),
                    status.get("binding_ready"),
                    status.get("binding_required"),
                    status.get("binding_configured_lanes", status.get("binding_target_lanes")),
                    status.get("binding_observed_lanes"),
                    status.get("binding_confirmed_lanes"),
                    status.get("binding_pending_lanes"),
                    status.get("last_face_ts"),
                    status.get("last_lane_observation_ts"),
                    status.get("camera_ready"),
                )
            except Exception as exc:
                self.logger.warning("binding timeout diagnostics failed: %s", exc)
        # 鍏抽棴璺熻釜鍔熻兘
        self.state.config["tracking_active"] = False
        # 鍙€夊仠姝㈣棰戝鐞嗙閬擄紙浼氬叧闂彲瑙嗗寲绐楀彛锛?
        if stop_capture:
            self.pipeline.stop()
            self.state.capture_running = False
            self.state.capture_error = None
        else:
            self.logger.info(
                "stop cmd=%s reason=%s stop_capture=%s -> keep capture and preview running",
                payload.cmd,
                reason or "-",
                stop_capture,
            )
            self.state.capture_running = self.pipeline.running
        # 鍋滄浜嬩欢妯℃嫙
        self.event_sim.stop()
        self._touch(payload.cmd)

    def _handle_reset_round(self, payload: CommandPayload) -> None:
        """
        澶勭悊杞閲嶇疆鍛戒护銆?

        淇濈暀缁戝畾淇℃伅锛屽皢鑺傜偣鎭㈠鍒扮瓑寰呴噸鏂拌捣璺戠殑鍑嗗鎬侊紝
        閫傜敤浜庢姠璺戠瓑杩濊鍚庣殑蹇€熼噸寮€銆?
        """
        self._ensure_same_session(payload)
        self._ensure_phase([NodePhase.BINDING, NodePhase.MONITORING, NodePhase.STOPPED])

        self.state.phase = NodePhase.BINDING
        self.state.expected_start_time = None
        self.state.stop_reason = None
        self.state.config["tracking_active"] = False
        self.state.binding_confirmed_students = []
        self.state.binding_confirmed_lanes = []
        self.state.binding_assignments = []
        self.state.binding_confirmed_at_ms = None
        self.state.last_face_result = None
        self.state.last_face_ts = None
        self.state.last_false_start_event = None
        self.state.last_false_start_ts = None
        self.state.last_toe_proxy_debug = None
        self.state.last_toe_proxy_ts = None
        self.algo.reset_binding_runtime()

        self.event_sim.stop()
        self._touch(payload.cmd)

    def _handle_heartbeat(self, payload: CommandPayload) -> None:
        """
        澶勭悊蹇冭烦鍛戒护 - 淇濇寔浼氳瘽娲昏穬

        浜戠瀹氭湡鍙戦€佸績璺充互纭杈圭紭璁惧鍦ㄧ嚎
        """
        # 楠岃瘉浼氳瘽涓€鑷存€?鍏佽绌轰細璇?鐢ㄤ簬鍒濆杩炴帴)
        self._ensure_same_session(payload, allow_empty=True)
        self._touch(payload.cmd)

    # ==================== 杈呭姪鏂规硶 ====================

    def _ensure_same_session(
        self, payload: CommandPayload, allow_empty: bool = False
    ) -> None:
        """
        楠岃瘉鍛戒护鐨勪細璇滻D涓庡綋鍓嶇姸鎬佷竴鑷?

        Args:
            payload: 鍛戒护璐熻浇
            allow_empty: 鏄惁鍏佽褰撳墠浼氳瘽涓虹┖(鐢ㄤ簬鍒濆杩炴帴)

        Raises:
            HTTPException: 浼氳瘽涓嶅尮閰嶆椂杩斿洖409閿欒
        """
        if allow_empty and not self.state.session_id:
            return
        if self.state.session_id and self.state.session_id != payload.session_id:
            raise HTTPException(status_code=409, detail="node session mismatch")

    def _ensure_phase(self, allowed: list[NodePhase]) -> None:
        """
        楠岃瘉褰撳墠闃舵鏄惁鍦ㄥ厑璁哥殑鍒楄〃涓?

        Args:
            allowed: 鍏佽鐨勯樁娈靛垪琛?

        Raises:
            HTTPException: 闃舵涓嶅尮閰嶆椂杩斿洖409閿欒
        """
        if self.state.phase not in allowed:
            raise HTTPException(
                status_code=409,
                detail=f"褰撳墠闃舵 {self.state.phase} 鏃犳晥; 鍏佽鐨勯樁娈? {[p.value for p in allowed]}",
            )

    def _touch(self, cmd: str) -> None:
        """
        鏇存柊鐘舵€佹椂闂存埑鍜屾渶鍚庡懡浠?

        姣忔澶勭悊鍛戒护鍚庤皟鐢?璁板綍鎿嶄綔鏃堕棿
        """
        self.state.last_command = cmd
        self.state.last_updated_ms = int(time.time() * 1000)


    @staticmethod
    def _summarize_command(payload: CommandPayload) -> str:
        config = payload.config or {}
        if payload.cmd == "CMD_INIT":
            return (
                f"project_type={config.get('project_type')} "
                f"lane_count={config.get('lane_count')}"
            )
        if payload.cmd == "CMD_BINDING_SYNC":
            bindings = config.get("bindings") or []
            return f"bindings={len(bindings)}"
        if payload.cmd == "CMD_START_MONITOR":
            return (
                f"start_at={config.get('expected_start_time')} "
                f"countdown={config.get('countdown_seconds', 3)} "
                f"tracking={config.get('tracking_active', True)}"
            )
        if payload.cmd in {"CMD_STOP", "CMD_RESET_ROUND"}:
            return (
                f"reason={config.get('reason')} "
                f"stop_capture={config.get('stop_capture')}"
            )
        return "-"

    def build_status_report(self) -> NodeStatusReport:
        lane_count = int(self.state.config.get("lane_count", 0) or 0)
        configured_target_lanes = binding_target_lanes(self.state.bindings, lane_count)

        observed_lanes: list[int] = []
        lane_debug = self.state.lane_layout_debug
        if isinstance(lane_debug, dict):
            observations = lane_debug.get("observations") or []
            if isinstance(observations, list):
                for item in observations:
                    if isinstance(item, dict) and isinstance(item.get("lane"), int):
                        observed_lanes.append(int(item["lane"]))
        observed_lanes = list(dict.fromkeys(observed_lanes))

        # 缁戝畾灏辩华浼樺厛鎸夆€滃綋鍓嶇敾闈㈤噷瀹為檯鍑虹幇鐨勮窇閬撯€濆垽瀹氾紝
        # 浣嗕粎闄愪簬宸查厤缃粦瀹氱殑璺戦亾锛岄伩鍏嶆棤鍏冲尯鍩熷共鎵般€?
        if configured_target_lanes and observed_lanes:
            observed_set = set(observed_lanes)
            target_lanes = [lane for lane in configured_target_lanes if lane in observed_set]
            if not target_lanes:
                target_lanes = configured_target_lanes
        else:
            target_lanes = configured_target_lanes

        binding_students_by_lane: dict[int, str] = {}
        for item in self.state.bindings:
            if not isinstance(item, dict):
                continue
            lane = item.get("lane")
            student_id = item.get("student_id")
            if isinstance(lane, int) and student_id:
                binding_students_by_lane[int(lane)] = str(student_id)

        binding_target_students = [
            binding_students_by_lane[lane]
            for lane in target_lanes
            if lane in binding_students_by_lane
        ]
        confirmed_students = list(dict.fromkeys(self.state.binding_confirmed_students))
        confirmed_lanes = list(dict.fromkeys(self.state.binding_confirmed_lanes))
        pending_students = [
            student_id for student_id in binding_target_students if student_id not in confirmed_students
        ]
        pending_lanes = [
            lane for lane in target_lanes if lane not in confirmed_lanes
        ]
        binding_required = bool(binding_target_students) and self.settings.node_role.upper() in {"START", "ALL_IN_ONE"}
        if self.settings.node_role.upper() in {"START", "ALL_IN_ONE"}:
            binding_required = bool(target_lanes)
        binding_ready = (not binding_required) or not pending_lanes
        now_ms = int(time.time() * 1000)
        session_stage = self.state.phase.value
        if self.state.phase == NodePhase.BINDING:
            session_stage = "WAIT_BINDING" if binding_required and not binding_ready else "BOUND"
        elif self.state.phase == NodePhase.MONITORING:
            if self.state.expected_start_time and now_ms < int(self.state.expected_start_time):
                session_stage = "COUNTDOWN"
            else:
                session_stage = "RUNNING"
        elif self.state.phase == NodePhase.STOPPED:
            session_stage = "STOPPED"

        lane_layout_status = inspect_lane_layout(
            frame_width=int(self.settings.capture_width),
            frame_height=int(self.settings.capture_height),
            target_lanes=target_lanes,
            lane_ranges_text=self.settings.lane_x_ranges,
            lane_polygons_text=self.settings.lane_polygons,
            lane_layout_file=self.settings.lane_layout_file,
        )

        return NodeStatusReport(
            node_id=self.state.node_id,
            session_id=self.state.session_id or "",
            timestamp=int(time.time() * 1000),
            data={
                "session_stage": session_stage,
                "phase": self.state.phase.value,
                "last_command": self.state.last_command,
                "capture_running": self.state.capture_running,
                "capture_fps_est": self.state.capture_fps_est,
                "last_frame_ts": self.state.last_frame_ts,
                "capture_error": self.state.capture_error,
                "binding_required": binding_required,
                "binding_ready": binding_ready,
                "binding_target_count": len(target_lanes),
                "binding_target_lanes": target_lanes,
                "binding_configured_lanes": configured_target_lanes,
                "binding_observed_lanes": observed_lanes,
                "binding_confirmed_count": len(confirmed_lanes),
                "binding_confirmed_lanes": confirmed_lanes,
                "binding_pending_count": len(pending_lanes),
                "binding_pending_lanes": pending_lanes,
                "binding_confirmed_students": confirmed_students,
                "binding_pending_students": pending_students,
                "binding_assignments": self.state.binding_assignments,
                "binding_confirmed_at_ms": self.state.binding_confirmed_at_ms,
                "last_face_ts": self.state.last_face_ts,
                "lane_layout_status": lane_layout_status,
                "lane_layout_debug": self.state.lane_layout_debug,
                "last_lane_observation_ts": self.state.last_lane_observation_ts,
                "camera_ready": self.state.capture_running and not self.state.capture_error,
                "tracking_active": bool(self.state.config.get("tracking_active", False)),
                "expected_start_time": self.state.expected_start_time,
                "ready_ts": self.state.config.get("ready_ts"),
                "countdown_seconds": self.state.config.get("countdown_seconds"),
                "binding_timeout_sec": self.state.config.get("binding_timeout_sec"),
                "race_timeout_sec": self.state.config.get("race_timeout_sec"),
                "last_false_start_ts": self.state.last_false_start_ts,
                "node_role": self.settings.node_role,
            },
        )

    def set_publisher(self, publisher) -> None:
        self.publisher = publisher
        self.event_sim.publisher = publisher
        self.algo.publisher = publisher

    def snapshot(self) -> dict:
        """
        鑾峰彇褰撳墠鐘舵€佺殑蹇収

        Returns:
            褰撳墠鐘舵€佺殑瀛楀吀琛ㄧず
        """
        return self.state.model_dump()

    # 娉ㄦ剰: 浣跨敤 EdgePipeline 鏃朵笉闇€瑕?_on_frame 鍥炶皟
    # EdgePipeline 鐨?_logic_worker 鐩存帴璋冪敤 algo.process_frame() 鎴?algo.process_pipeline_result()

    # ==================== 鐘舵€佹寔涔呭寲 ====================

    def _persist_state(self) -> None:
        """
        灏嗗綋鍓嶇姸鎬佹寔涔呭寲鍒癑SON鏂囦欢

        鐢ㄤ簬鏈嶅姟閲嶅惎鍚庢仮澶嶇姸鎬?
        """
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("w", encoding="utf-8") as f:
            json.dump(self.state.model_dump(), f, ensure_ascii=False, indent=2)

    def _load_state(self) -> NodeState:
        """
        浠嶫SON鏂囦欢鍔犺浇鐘舵€?

        Returns:
            鍔犺浇鐨勮妭鐐圭姸鎬?濡傛灉鏂囦欢涓嶅瓨鍦ㄦ垨鎹熷潖鍒欒繑鍥炴柊鐘舵€?
        """
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return NodeState(**data)
        except FileNotFoundError:
            # 棣栨杩愯,鍒涘缓鏂扮姸鎬?
            return NodeState(node_id=self.settings.node_id)
        except Exception as exc:  # 鐘舵€佹枃浠舵崯鍧?
            logging.getLogger("edge.command").warning("鍔犺浇 state.json 澶辫触: %s", exc)
            return NodeState(node_id=self.settings.node_id)
