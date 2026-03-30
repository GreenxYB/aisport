# 云边通信协议草案（v0.2）

## 1. 总体原则

当前建议的协议方向是：

- `WebSocket`：云端与边缘节点的主链路
- `HTTP`：调试、管理、快照查看等辅助接口
- `MQTT`：后续设备规模提升后再评估

当前仓库中的边缘命令接口仍以 HTTP 为主，但协议设计以 WebSocket 长连接为目标。

基础约定：

- 编码：UTF-8
- 时间戳：毫秒级 Unix epoch
- `session_id`：一轮任务唯一标识
- `node_id`：设备唯一标识

## 2. 设备身份与连接

`task_mode` 不用于标识设备身份。

设备身份建议通过连接阶段完成，使用：

- `node_id`：设备编号
- `node_role`：设备角色

推荐角色：

- `START`
- `FINISH`
- `MID`
- `ALL_IN_ONE`

连接阶段建议消息：

```json
{
  "node_id": 1,
  "node_role": "START",
  "site_id": "school_a",
  "capabilities": ["camera", "speaker"],
  "token": "optional"
}
```

当前阶段可以先只做：

- `node_id` 白名单校验

后续再升级为：

- `node_id + token`
- 或签名认证

## 3. Cloud -> Edge 指令格式

统一命令格式：

```json
{
  "cmd": "CMD_INIT",
  "session_id": "RUN_20260326_100000_001",
  "node_id": 1,
  "task_mode": "TRACK_RACE",
  "config": {
    "project_type": "200m",
    "lane_count": 8,
    "sync_time": 1773000000000
  }
}
```

字段说明：

- `cmd`：命令类型
- `session_id`：本轮任务 ID，建议用yyyymmdd_hhmmss_00x的格式
- `node_id`：目标设备 ID
- `task_mode`：任务模式，不是设备身份
- `config`：具体业务参数

## 4. `task_mode` 的作用

`task_mode` 是“任务模式字段”，用于标识本轮任务是什么类型。

推荐理解：

- `task_mode` 回答“这轮任务是什么”
- `node_role` 回答“这个节点负责哪里”

示例：

- `TRACK_RACE`
- `SUNSHINE_RUN`
- `FREE_RUN_ANALYSIS`

当前边缘端真正决定处理流程的，仍然主要是：

- `cmd`
- 当前 `phase`
- `config`

也就是说，`task_mode` 已进入协议，但更多是为后续多模式扩展预留。

## 5. 指令列表

### `CMD_INIT`

开始一轮新的任务，初始化 session。

示例：

```json
{
  "cmd": "CMD_INIT",
  "session_id": "RUN_20260326_100000_001",
  "node_id": 1,
  "task_mode": "TRACK_RACE",
  "config": {
    "project_type": "200m",
    "lane_count": 8,
    "sync_time": 1773000000000
  }
}
```

说明：

- 正常开始下一轮时，应重新发送 `CMD_INIT`
- `CMD_STOP` 更适合人工中止或异常终止

### `CMD_BINDING_SYNC`

同步跑道与学生绑定信息。

示例：

```json
{
  "cmd": "CMD_BINDING_SYNC",
  "session_id": "RUN_20260326_100000_001",
  "node_id": 1,
  "task_mode": "TRACK_RACE",
  "config": {
    "bindings": [
      { "lane": 1, "student_id": "S101", "feature_id": "F001" },
      { "lane": 2, "student_id": "S102", "feature_id": "F002" }
    ]
  }
}
```

### `CMD_START_MONITOR`

开始监控。建议在所有关键节点 ready 后再下发，不建议在 `CMD_BINDING_SYNC` 后立即盲发。

示例：

```json
{
  "cmd": "CMD_START_MONITOR",
  "session_id": "RUN_20260326_100000_001",
  "node_id": 1,
  "task_mode": "TRACK_RACE",
  "config": {
    "expected_start_time": 1773000005000,
    "false_start_check": true,
    "tracking_active": true,
    "countdown_seconds": 3,
    "audio_plan": "START_321_GO"
  }
}
```

说明：

- `expected_start_time` 是全局统一的官方起跑时间
- 起点节点本地负责倒计时与喇叭播报
- 所有节点都按同一个 `expected_start_time` 判定正式起跑

### `CMD_RESET_ROUND`

一轮中发生违规后，重置当前轮次，回到等待重新起跑状态，建议三次以上云端直接发送CMD_STOP指令，完善退出机制，避免重复CMD_RESET_ROUND。

示例：

```json
{
  "cmd": "CMD_RESET_ROUND",
  "session_id": "RUN_20260326_100000_001",
  "node_id": 1,
  "task_mode": "TRACK_RACE",
  "config": {
    "reason": "FALSE_START"
  }
}
```

说明：

- 保留绑定信息
- 清空本轮运行态
- 不必重新走一遍完整初始化

### `CMD_STOP`

人工中止或异常终止任务。

示例：

```json
{
  "cmd": "CMD_STOP",
  "session_id": "RUN_20260326_100000_001",
  "node_id": 1,
  "task_mode": "TRACK_RACE",
  "config": {
    "reason": "MANUAL_ABORT"
  }
}
```

### `CMD_HEARTBEAT`

保活或链路探测，可选。

## 6. 云端何时下发 `CMD_START_MONITOR`

建议的简单流程：

1. 云端发送 `CMD_INIT`
2. 云端发送 `CMD_BINDING_SYNC`
3. 各节点上报 ready 状态
4. 云端确认起点、终点等必需节点都 ready
5. 云端发送 `CMD_START_MONITOR`

这里的 ready 可以先通过 `NODE_STATUS` 上报。

例如：

```json
{
  "msg_type": "NODE_STATUS",
  "node_id": 1,
  "session_id": "RUN_20260326_100000_001",
  "timestamp": 1773000004000,
  "data": {
    "phase": "BINDING",
    "camera_ready": true,
    "binding_ready": true,
    "speaker_ready": true
  }
}
```

## 7. 起跑与喇叭控制

建议规则：

- 官方起跑时间以 `expected_start_time` 为准
- 起点节点收到 `CMD_START_MONITOR` 后，本地执行倒计时
- 起点节点本地控制喇叭播放 `3 2 1 跑`

这样做的好处是：

- 减少云端二次控制带来的延迟
- 起点检测与播报在同一个节点完成，更稳定

如果发生抢跑，建议：

- 起点节点本地立即播报“检测到违规，请重新准备”
- 同时上报 `VIOLATION_EVENT`
- 云端再下发 `CMD_RESET_ROUND`

## 8. 成绩计算

第一版建议按下面方式计算：

```text
score = finish_ts - expected_start_time
```

前提：

- 所有节点使用同一官方起跑时间
- 节点间时钟误差可接受

时钟同步问题可以先放在后续阶段再强化。

## 9. Edge -> Cloud 上报格式

统一原则：

- 列表型数据统一放在 `data: []`
- 对象型数据统一放在 `data: {}`
- 不再使用 `results`

### `ID_REPORT`

```json
{
  "msg_type": "ID_REPORT",
  "node_id": 1,
  "session_id": "RUN_20260326_100000_001",
  "timestamp": 1773000002000,
  "data": [
    {
      "lane": 1,
      "student_id": "S101",
      "confidence": 0.98,
      "face_token": "xxx",
      "name": "张三"
    }
  ]
}
```

### `VIOLATION_EVENT`

```json
{
  "msg_type": "VIOLATION_EVENT",
  "node_id": 1,
  "session_id": "RUN_20260326_100000_001",
  "timestamp": 1773000004800,
  "data": [
    {
      "event": "FALSE_START",
      "lane": 3,
      "track_id": 8,
      "bbox": [100, 200, 300, 400],
      "evidence_frame": null
    }
  ]
}
```

### `FINISH_REPORT`

```json
{
  "msg_type": "FINISH_REPORT",
  "node_id": 7,
  "session_id": "RUN_20260326_100000_001",
  "timestamp": 1773000025000,
  "data": [
    {
      "lane": 1,
      "track_id": 11,
      "rank": 1,
      "finish_ts": 1773000025450
    }
  ]
}
```

### `NODE_STATUS`

```json
{
  "msg_type": "NODE_STATUS",
  "node_id": 1,
  "session_id": "RUN_20260326_100000_001",
  "timestamp": 1773000003000,
  "data": {
    "phase": "BINDING",
    "camera_ready": true,
    "binding_ready": true,
    "last_cmd": "CMD_BINDING_SYNC"
  }
}
```

## 10. 一轮结束后如何开始下一轮

建议：

- 正常完成一轮后，下一轮重新发送 `CMD_INIT`
- 不需要先发送 `CMD_STOP`
- `CMD_STOP` 只用于人工停止或异常中断

## 11. 状态机建议

简化版建议状态流：

`IDLE -> BINDING -> READY -> MONITORING -> FINISHED`

如果发生违规：

`MONITORING -> RESET_ROUND -> BINDING/READY`

当前仓库代码里仍是简化状态机，但协议层建议已经向这个方向靠拢。

## 12. 时钟同步

当前阶段建议先不实现复杂时钟同步，仅保留：

- `sync_time`
- `server_time`

作为时间参考字段。

后续如果成绩精度成为瓶颈，再补：

- 时钟偏移测量
- 多次校时
- 发令确认时间
