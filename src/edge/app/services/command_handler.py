import json
import logging
import time
from pathlib import Path
from typing import Callable

from fastapi import HTTPException

from common.protocol import CommandPayload, NodeStatusReport
from ..core.config import get_settings
from ..core.state import NodePhase, NodeState
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
    命令处理器 - 处理来自云端的控制命令

    负责管理边缘设备的生命周期:
    - 初始化会话 (CMD_INIT)
    - 同步绑定信息 (CMD_BINDING_SYNC)
    - 开始监控 (CMD_START_MONITOR)
    - 停止监控 (CMD_STOP)
    - 心跳检测 (CMD_HEARTBEAT)
    """

    def __init__(self):
        """初始化命令处理器"""
        self.settings = get_settings()
        # 状态文件路径 - 用于持久化节点状态
        self.state_file = Path(__file__).resolve().parents[4] / "logs" / "state.json"
        # 加载或创建初始状态
        self.state = self._load_state()
        self.logger = logging.getLogger("edge.command")
        self.publisher = NullPublisher()

        # 初始化事件模拟器 - 用于测试生成模拟事件
        self.event_sim = EventSimulator(self.state, publisher=self.publisher)
        # 初始化算法运行器 - 处理视频帧并检测事件
        self.algo = AlgorithmRunner(self.state, publisher=self.publisher)

        # 使用 EdgePipeline 进行多线程视频处理
        # 包含: 视频采集 -> YOLO推理 -> 目标跟踪 -> 业务逻辑
        if EdgePipeline is None:
            self.pipeline = _NoopPipeline()
            self.logger.warning("EdgePipeline unavailable, falling back to no-op pipeline: %s", PIPELINE_IMPORT_ERROR)
        else:
            self.pipeline = EdgePipeline(algo_runner=self.algo)

        # 如果配置了自动启动采集,则启动视频处理管道
        if self.settings.auto_start_capture:
            self.pipeline.start()
            self.state.capture_running = self.pipeline.running
            self.logger.info(f"自动启动采集: 管道运行状态={self.pipeline.running}")

        # 定义允许的命令集合
        self.allowed_cmds = {
            "CMD_INIT",  # 初始化会话
            "CMD_BINDING_SYNC",  # 同步运动员绑定信息
            "CMD_START_MONITOR",  # 开始监控/检测
            "CMD_STOP",  # 停止监控
            "CMD_RESET_ROUND",  # 违规后重置当前轮次
            "CMD_HEARTBEAT",  # 心跳检测
        }
        # 命令分发映射表 - 将命令映射到对应的处理方法
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
        处理接收到的命令

        Args:
            payload: 命令负载,包含命令类型、会话ID、节点ID和配置信息

        Raises:
            HTTPException: 当命令不支持时返回400错误
        """
        started = time.time()
        # 检查命令是否在允许列表中
        if payload.cmd not in self.allowed_cmds:
            self.logger.warning("未知命令 %s", payload.cmd)
            raise HTTPException(status_code=400, detail="不支持的命令")

        # 获取对应的命令处理器
        handler = self._dispatch_map[payload.cmd]
        self.logger.info(
            "cmd=%s session=%s node=%s summary=%s",
            payload.cmd,
            payload.session_id,
            payload.node_id,
            self._summarize_command(payload),
        )

        # 执行命令处理
        handler(payload)
        # 持久化状态到文件
        self._persist_state()

        # 记录处理耗时
        elapsed_ms = int((time.time() - started) * 1000)
        self.logger.info(
            "handled cmd=%s phase=%s elapsed_ms=%s",
            payload.cmd,
            self.state.phase,
            elapsed_ms,
        )

    # ==================== 命令处理器 ====================

    def _handle_init(self, payload: CommandPayload) -> None:
        """
        处理初始化命令 - 重置所有状态

        当会话变化时,重置所有状态以开始新的比赛会话
        """
        # 设置会话ID
        self.state.session_id = payload.session_id
        # 进入绑定阶段 - 等待运动员信息绑定
        self.state.phase = NodePhase.BINDING
        # 保存配置信息
        self.state.config = payload.config or {}
        # 清空运动员绑定列表
        self.state.bindings = []
        # 重置各种状态字段
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
        # 停止事件模拟
        self.event_sim.stop()
        self._touch(payload.cmd)

    def _handle_binding_sync(self, payload: CommandPayload) -> None:
        """
        处理绑定同步命令 - 同步运动员与跑道的绑定信息

        在比赛开始前,将运动员ID与跑道编号进行绑定
        """
        # 验证会话一致性
        self._ensure_same_session(payload)
        # 验证当前阶段必须是BINDING阶段
        self._ensure_phase([NodePhase.BINDING])

        # 获取绑定信息
        bindings = payload.config.get("bindings") if payload.config else None
        self.state.bindings = bindings or []
        # 保持在BINDING阶段,等待开始监控命令
        self.state.phase = NodePhase.BINDING

        # 如果启用了事件模拟,启动模拟器
        if self.settings.simulate_events:
            lane_count = int(self.state.config.get("lane_count", 1) or 1)
            self.event_sim.start(self.state.session_id or "UNKNOWN", lane_count)
        self._touch(payload.cmd)

    def _handle_start_monitor(self, payload: CommandPayload) -> None:
        """
        处理开始监控命令 - 启动视频采集和算法检测

        进入MONITORING阶段,开始实时检测违规行为和冲线事件
        """
        # 验证会话一致性
        self._ensure_same_session(payload)
        # 验证当前阶段必须是BINDING阶段(只能从绑定阶段进入监控阶段)
        self._ensure_phase([NodePhase.BINDING])

        # 切换到监控阶段
        self.state.phase = NodePhase.MONITORING
        # 保存预计开始时间
        self.state.expected_start_time = (payload.config or {}).get(
            "expected_start_time"
        )
        self.state.config["ready_ts"] = int(time.time() * 1000)
        # 激活跟踪功能
        self.state.config["tracking_active"] = (payload.config or {}).get(
            "tracking_active", True
        )
        self.state.config["countdown_seconds"] = (payload.config or {}).get(
            "countdown_seconds", 3
        )

        # 如果视频管道未运行,则启动它
        if not self.pipeline.running:
            try:
                self.pipeline.start()
            except Exception as exc:
                self.state.capture_error = str(exc)
                raise HTTPException(status_code=503, detail="摄像头打开失败")

        # 更新采集状态
        self.state.capture_running = self.pipeline.running
        self.state.capture_error = None

        # 如果启用了事件模拟,启动模拟器
        if self.settings.simulate_events:
            lane_count = int(self.state.config.get("lane_count", 1) or 1)
            self.event_sim.start(self.state.session_id or "UNKNOWN", lane_count)
        self._touch(payload.cmd)

    def _handle_stop(self, payload: CommandPayload) -> None:
        """
        处理停止命令 - 停止监控并清理资源

        进入STOPPED阶段,停止视频采集和事件生成
        """
        # 验证会话一致性
        self._ensure_same_session(payload)
        # 切换到停止阶段
        self.state.phase = NodePhase.STOPPED
        # 保存停止原因
        self.state.stop_reason = (payload.config or {}).get("reason")
        # 关闭跟踪功能
        self.state.config["tracking_active"] = False
        # 停止视频处理管道
        self.pipeline.stop()
        # 更新采集状态
        self.state.capture_running = False
        self.state.capture_error = None
        # 停止事件模拟
        self.event_sim.stop()
        self._touch(payload.cmd)

    def _handle_reset_round(self, payload: CommandPayload) -> None:
        """
        处理轮次重置命令。

        保留绑定信息，将节点恢复到等待重新起跑的准备态，
        适用于抢跑等违规后的快速重开。
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

        self.event_sim.stop()
        self._touch(payload.cmd)

    def _handle_heartbeat(self, payload: CommandPayload) -> None:
        """
        处理心跳命令 - 保持会话活跃

        云端定期发送心跳以确认边缘设备在线
        """
        # 验证会话一致性(允许空会话,用于初始连接)
        self._ensure_same_session(payload, allow_empty=True)
        self._touch(payload.cmd)

    # ==================== 辅助方法 ====================

    def _ensure_same_session(
        self, payload: CommandPayload, allow_empty: bool = False
    ) -> None:
        """
        验证命令的会话ID与当前状态一致

        Args:
            payload: 命令负载
            allow_empty: 是否允许当前会话为空(用于初始连接)

        Raises:
            HTTPException: 会话不匹配时返回409错误
        """
        if allow_empty and not self.state.session_id:
            return
        if self.state.session_id and self.state.session_id != payload.session_id:
            raise HTTPException(status_code=409, detail="节点会话不匹配")

    def _ensure_phase(self, allowed: list[NodePhase]) -> None:
        """
        验证当前阶段是否在允许的列表中

        Args:
            allowed: 允许的阶段列表

        Raises:
            HTTPException: 阶段不匹配时返回409错误
        """
        if self.state.phase not in allowed:
            raise HTTPException(
                status_code=409,
                detail=f"当前阶段 {self.state.phase} 无效; 允许的阶段: {[p.value for p in allowed]}",
            )

    def _touch(self, cmd: str) -> None:
        """
        更新状态时间戳和最后命令

        每次处理命令后调用,记录操作时间
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
            return f"reason={config.get('reason')}"
        return "-"

    def build_status_report(self) -> NodeStatusReport:
        binding_target_lanes = [
            int(item.get("lane"))
            for item in self.state.bindings
            if isinstance(item, dict) and isinstance(item.get("lane"), int)
        ]
        if not binding_target_lanes:
            lane_count = int(self.state.config.get("lane_count", 0) or 0)
            binding_target_lanes = list(range(1, lane_count + 1))

        binding_target_students = [
            str(item.get("student_id"))
            for item in self.state.bindings
            if isinstance(item, dict) and item.get("student_id")
        ]
        confirmed_students = list(dict.fromkeys(self.state.binding_confirmed_students))
        confirmed_lanes = list(dict.fromkeys(self.state.binding_confirmed_lanes))
        pending_students = [
            student_id for student_id in binding_target_students if student_id not in confirmed_students
        ]
        pending_lanes = [
            lane for lane in binding_target_lanes if lane not in confirmed_lanes
        ]
        binding_required = bool(binding_target_students) and self.settings.node_role.upper() in {"START", "ALL_IN_ONE"}
        if self.settings.node_role.upper() in {"START", "ALL_IN_ONE"}:
            binding_required = bool(binding_target_lanes)
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
                "binding_target_count": len(binding_target_lanes),
                "binding_target_lanes": binding_target_lanes,
                "binding_confirmed_count": len(confirmed_lanes),
                "binding_confirmed_lanes": confirmed_lanes,
                "binding_pending_count": len(pending_lanes),
                "binding_pending_lanes": pending_lanes,
                "binding_confirmed_students": confirmed_students,
                "binding_pending_students": pending_students,
                "binding_assignments": self.state.binding_assignments,
                "binding_confirmed_at_ms": self.state.binding_confirmed_at_ms,
                "last_face_ts": self.state.last_face_ts,
                "camera_ready": self.state.capture_running and not self.state.capture_error,
                "tracking_active": bool(self.state.config.get("tracking_active", False)),
                "expected_start_time": self.state.expected_start_time,
                "ready_ts": self.state.config.get("ready_ts"),
                "countdown_seconds": self.state.config.get("countdown_seconds"),
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
        获取当前状态的快照

        Returns:
            当前状态的字典表示
        """
        return self.state.model_dump()

    # 注意: 使用 EdgePipeline 时不需要 _on_frame 回调
    # EdgePipeline 的 _logic_worker 直接调用 algo.process_frame() 或 algo.process_pipeline_result()

    # ==================== 状态持久化 ====================

    def _persist_state(self) -> None:
        """
        将当前状态持久化到JSON文件

        用于服务重启后恢复状态
        """
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("w", encoding="utf-8") as f:
            json.dump(self.state.model_dump(), f, ensure_ascii=False, indent=2)

    def _load_state(self) -> NodeState:
        """
        从JSON文件加载状态

        Returns:
            加载的节点状态,如果文件不存在或损坏则返回新状态
        """
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            return NodeState(**data)
        except FileNotFoundError:
            # 首次运行,创建新状态
            return NodeState(node_id=self.settings.node_id)
        except Exception as exc:  # 状态文件损坏
            logging.getLogger("edge.command").warning("加载 state.json 失败: %s", exc)
            return NodeState(node_id=self.settings.node_id)
