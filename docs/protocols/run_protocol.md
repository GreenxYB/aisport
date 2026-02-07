# 云端⇄边缘 通信协议草案 (v0.1)

## 总体
- 传输：HTTP/HTTPS JSON（后续可切换 MQTT/AMQP，保持消息体一致）。
- 时间戳：毫秒级 Unix epoch（int64）。
- 编码：UTF-8。
- 会话号：`RUN_YYYYMMDD_HHMMSS_mmm`（云端生成，唯一）。

## Cloud → Edge 指令集
所有指令共用字段：
```json
{
  "cmd": "CMD_INIT",
  "session_id": "RUN_20260207_120000_123",
  "node_id": 1,
  "task_mode": "IDENTITY_BINDING",
  "config": { "project_type": "200m", "lane_count": 8, "sync_time": 1738416000000 }
}
```
- `cmd`: 指令名称（见下）。
- `node_id`: 目标节点编号。
- `task_mode`: 可选，标识节点内部模式。
- `config`: 业务配置，按指令不同携带。

指令列表：
- `CMD_INIT`：任务初始化/身份绑定。config 必含 `project_type`、`lane_count`、`sync_time`。
- `CMD_BINDING_SYNC`：广播跑道绑定名单。config 示例：`{"bindings":[{"lane":1,"student_id":"S101","feature_id":"F001"}]}`。
- `CMD_START_MONITOR`：起跑监测启动。config 示例：`{"expected_start_time":1738416005000,"false_start_check":true,"tracking_active":true}`。
- `CMD_STOP`：紧急终止。config 示例：`{"reason":"FALSE_START_LANE_3"}`。
- `CMD_HEARTBEAT`：可选，云端对节点状态轮询/保活。

状态机约束（Edge）：
- 允许：`IDLE -> CMD_INIT -> BINDING -> CMD_BINDING_SYNC -> BINDING -> CMD_START_MONITOR -> MONITORING -> CMD_STOP -> STOPPED`
- 其他顺序将返回 `409 Invalid phase`。

## Edge → Cloud 回传
### 身份绑定
`msg_type: "ID_REPORT"`
```json
{
  "msg_type": "ID_REPORT",
  "node_id": 1,
  "session_id": "RUN_20260207_120000_123",
  "results": [
    {"lane":1,"student_id":"S101","confidence":0.98,"face_img":"base64..."},
    {"lane":2,"status":"VACANT"}
  ]
}
```

### 违规事件
`msg_type: "VIOLATION_EVENT"`
```json
{
  "msg_type": "VIOLATION_EVENT",
  "node_id": 1,
  "session_id": "RUN_20260207_120000_123",
  "event": "FALSE_START",
  "lane": 3,
  "timestamp": 1738416004850,
  "evidence_frame": "url_or_base64"
}
```

### 冲线成绩
`msg_type: "FINISH_REPORT"`
```json
{
  "msg_type": "FINISH_REPORT",
  "node_id": 7,
  "session_id": "RUN_20260207_120000_123",
  "results": [
    {"lane":1,"finish_ts":1738416025450},
    {"lane":3,"finish_ts":1738416026120}
  ]
}
```

### 节点心跳/状态
`msg_type: "NODE_STATUS"`
```json
{
  "msg_type": "NODE_STATUS",
  "node_id": 3,
  "session_id": "RUN_20260207_120000_123",
  "uptime_sec": 3600,
  "temp_c": 52.3,
  "last_cmd": "CMD_START_MONITOR",
  "last_cmd_ts": 1738416005000
}
```

## 错误与应答
- HTTP 200 + `{"status":"accepted","phase":...,"last_updated_ms":...}` 代表节点已收并更新状态。
- 常见错误码：
  - 400：参数错误/不支持的指令
  - 404：会话/节点未找到（预留）
  - 409：状态冲突（例如未 INIT 就 START_MONITOR，或 session_id 不一致）
  - 500：节点内部错误

## 安全
- 鉴权预留：HTTP Header `Authorization: Bearer <token>`；token 由云端 IAM 发放。
- 所有消息需带 `session_id` 防串话；云端校验 `node_id` 与任务绑定关系。

## 后续待定
- MQTT 主题规范（例如 `run/{session_id}/{node_id}/cmd`）。
- 文件/大图传输：采用对象存储 URL，上报仅传引用。
- 精度校准：NTP/PPs 或云端时间同步指令的对时流程细节。
