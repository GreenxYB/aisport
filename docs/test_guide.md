# AI Sport 跑步测试全流程测试文档

## 目录

1. [环境准备](#环境准备)
2. [前置条件](#前置条件)
3. [测试配置](#测试配置)
4. [完整测试流程](#完整测试流程)
5. [验证检查点](#验证检查点)
6. [常见问题解决](#常见问题解决)

---

## 环境准备

### 1. 依赖安装

确保已安装所有 Python 依赖：

```bash
cd /Users/luoweibin/Desktop/code/aisport
pip install -r requirements.txt
```

### 2. 目录结构检查

确保项目目录结构完整：

```
aisport/
├── configs/
│   ├── cloud.example.env
│   └── edge.example.env
├── src/
│   ├── cloud/
│   ├── common/
│   └── edge/
├── tests/
├── docs/
├── scripts/
├── logs/              # 日志目录（自动创建）
└── data/
    └── models/        # 模型文件目录
```

---

## 前置条件

### 1. 创建配置文件

在项目根目录创建 `.env.edge` 文件：

```bash
cd /Users/luoweibin/Desktop/code/aisport
cp configs/edge.example.env .env.edge
```

### 2. 配置测试参数

编辑 `.env.edge` 文件，设置以下参数（使用模拟模式）：

```env
# 环境设置
ENV=dev
NODE_ID=1

# 摄像头设置 - 使用模拟模式
SIMULATE_CAMERA=true
# 如果使用真实摄像头，请设置:
# SIMULATE_CAMERA=false
# RTSP_URL=rtsp://username:password@ip:port/path
# 或 CAMERA_DEVICE=0

# 显示设置
DISPLAY_PREVIEW=true
DISPLAY_MIRROR=true

# 算法设置
ALGO_ENABLED=true
ALGO_TARGET_FPS=5

# 事件模拟设置 - 启用模拟事件
SIMULATE_EVENTS=true
EVENT_INTERVAL_SEC=2.0
SIMULATE_FINISH_REPORTS=true
FINISH_INTERVAL_SEC=8.0

# 上报设置 - 暂时禁用上报（先不启动云端）
REPORT_ENABLED=false
# 如果启用上报，需要设置:
# REPORT_ENABLED=true
# REPORT_BASE_URL=http://localhost:8000/nodes/reports
# REPORT_TIMEOUT_SEC=2.0
# REPORT_RETRY_ENABLED=true
# REPORT_RETRY_INTERVAL_SEC=3.0
# REPORT_RETRY_MAX=5

# 模型设置
MODEL_DIR=./data/models
```

### 3. 创建必要目录

```bash
mkdir -p logs
mkdir -p data/models
```

---

## 测试配置

### 测试方案选择

| 测试模式 | 说明 | 适用场景 |
|---------|------|---------|
| **模拟模式** | 不需要真实摄像头，使用生成的假帧 | 快速测试、调试 |
| **真实摄像头** | 使用 USB 摄像头 | 本地测试 |
| **RTSP 流** | 使用网络摄像头流 | 实际部署测试 |

### 本次测试：模拟模式

本次测试使用**模拟模式**，可以快速验证完整流程。

---

## 完整测试流程

### 测试步骤总览

```
1. 启动边缘服务
   ↓
2. 检查服务健康状态
   ↓
3. 发送 CMD_INIT - 初始化会话
   ↓
4. 发送 CMD_BINDING_SYNC - 同步绑定信息
   ↓
5. 发送 CMD_START_MONITOR - 开始监控
   ↓
6. 观察事件生成和日志
   ↓
7. 发送 CMD_STOP - 停止监控
   ↓
8. 验证测试结果
```

---

### 测试步骤详解

#### 步骤 1：启动边缘服务

**操作**：

打开终端 1，运行边缘服务：

```bash
cd /Users/luoweibin/Desktop/code/aisport
PYTHONPATH="src" python run_edge.py
```

**预期输出**：

```
INFO:edge.command:Auto-start capture: pipeline running=True
INFO:edge.pipeline:启动 EdgePipeline...
INFO:edge.camera:Starting simulated capture
INFO:edge.pipeline:加载 TRT 模型: data/models/yolo11n-pose.engine
WARNING:edge.pipeline:模型文件不存在: data/models/yolo11n-pose.engine
INFO:edge.pipeline:BYTETracker 配置加载成功
INFO:edge.pipeline:BYTETracker 初始化成功
```

**检查点**：

- ✅ 服务在 `http://0.0.0.0:8100` 启动
- ✅ 没有 ERROR 级别的日志
- ✅ Pipeline 启动成功

---

#### 步骤 2：检查服务健康状态

**操作**：

打开终端 2，使用 curl 或浏览器访问：

```bash
curl http://localhost:8100/health
```

**预期输出**：

```json
{"status":"healthy"}
```

**检查点**：

- ✅ HTTP 状态码 200
- ✅ 返回健康状态

---

#### 步骤 3：发送 CMD_INIT - 初始化会话

**操作**：

在终端 2 中执行：

```bash
curl -X POST http://localhost:8100/commands/ \
  -H "Content-Type: application/json" \
  -d '{
    "cmd": "CMD_INIT",
    "session_id": "TEST_SESSION_001",
    "node_id": 1,
    "task_mode": "IDENTITY_BINDING",
    "config": {
      "lane_count": 4,
      "project_type": "100m"
    }
  }'
```

**预期输出**：

```json
{
  "session_id": "TEST_SESSION_001",
  "node_id": 1,
  "cmd": "CMD_INIT",
  "phase": "BINDING",
  "last_updated_ms": 1711345678000
}
```

**检查点**：

- ✅ HTTP 状态码 200
- ✅ 返回 `phase: "BINDING"`
- ✅ 返回正确的 `session_id`

**查看终端 1 日志**：

```
INFO:edge.command:recv cmd=CMD_INIT session=TEST_SESSION_001 node=1 config=...
INFO:edge.command:state updated phase=BINDING last_cmd=CMD_INIT elapsed_ms=12
```

---

#### 步骤 4：发送 CMD_BINDING_SYNC - 同步绑定信息

**操作**：

在终端 2 中执行：

```bash
curl -X POST http://localhost:8100/commands/ \
  -H "Content-Type: application/json" \
  -d '{
    "cmd": "CMD_BINDING_SYNC",
    "session_id": "TEST_SESSION_001",
    "node_id": 1,
    "config": {
      "bindings": [
        {"lane": 1, "athlete_id": "A001", "name": "张三"},
        {"lane": 2, "athlete_id": "A002", "name": "李四"},
        {"lane": 3, "athlete_id": "A003", "name": "王五"},
        {"lane": 4, "athlete_id": "A004", "name": "赵六"}
      ]
    }
  }'
```

**预期输出**：

```json
{
  "session_id": "TEST_SESSION_001",
  "node_id": 1,
  "cmd": "CMD_BINDING_SYNC",
  "phase": "BINDING",
  "last_updated_ms": 1711345678500
}
```

**检查点**：

- ✅ HTTP 状态码 200
- ✅ 保持 `phase: "BINDING"`
- ✅ 绑定信息已保存

**查看终端 1 日志**：

```
INFO:edge.command:recv cmd=CMD_BINDING_SYNC session=TEST_SESSION_001 node=1 config=...
INFO:edge.command:state updated phase=BINDING last_cmd=CMD_BINDING_SYNC elapsed_ms=8
INFO:edge.event-sim:Starting simulated capture
```

---

#### 步骤 5：发送 CMD_START_MONITOR - 开始监控

**操作**：

在终端 2 中执行：

```bash
curl -X POST http://localhost:8100/commands/ \
  -H "Content-Type: application/json" \
  -d '{
    "cmd": "CMD_START_MONITOR",
    "session_id": "TEST_SESSION_001",
    "node_id": 1,
    "config": {
      "expected_start_time": 1711345679000
    }
  }'
```

**预期输出**：

```json
{
  "session_id": "TEST_SESSION_001",
  "node_id": 1,
  "cmd": "CMD_START_MONITOR",
  "phase": "MONITORING",
  "last_updated_ms": 1711345679000
}
```

**检查点**：

- ✅ HTTP 状态码 200
- ✅ `phase` 变为 `"MONITORING"`
- ✅ `capture_running` 为 true

**查看终端 1 日志**：

```
INFO:edge.command:recv cmd=CMD_START_MONITOR session=TEST_SESSION_001 node=1 config=...
INFO:edge.command:state updated phase=MONITORING last_cmd=CMD_START_MONITOR elapsed_ms=15
```

---

#### 步骤 6：观察事件生成和日志

**操作**：

在终端 1 中观察日志输出，应该看到：

**模拟违规事件**（每 2 秒）：
```
INFO:edge.command:生成违规事件 VIOLATION_EVENT lane=2
```

**模拟冲线报告**（每 8 秒）：
```
INFO:edge.command:生成冲线报告 FINISH_REPORT
```

**检查事件日志文件**：

打开新终端 3，查看事件日志：

```bash
cd /Users/luoweibin/Desktop/code/aisport
tail -f logs/events.jsonl
```

**预期事件内容**：

违规事件：
```json
{
  "msg_type": "VIOLATION_EVENT",
  "node_id": 1,
  "session_id": "TEST_SESSION_001",
  "event": "FALSE_START",
  "lane": 2,
  "timestamp": 1711345680000,
  "evidence_frame": null
}
```

冲线报告：
```json
{
  "msg_type": "FINISH_REPORT",
  "node_id": 1,
  "session_id": "TEST_SESSION_001",
  "results": [
    {"lane": 1, "finish_ts": 1711345685000},
    {"lane": 2, "finish_ts": 1711345686000},
    {"lane": 3, "finish_ts": 1711345684000},
    {"lane": 4, "finish_ts": 1711345687000}
  ]
}
```

**检查点**：

- ✅ 每 2 秒生成 1 个违规事件
- ✅ 每 8 秒生成 1 个冲线报告
- ✅ `logs/events.jsonl` 文件被正确写入
- ✅ 事件包含正确的 `session_id` 和 `node_id`

---

#### 步骤 7：发送 CMD_STOP - 停止监控

**操作**：

在终端 2 中执行：

```bash
curl -X POST http://localhost:8100/commands/ \
  -H "Content-Type: application/json" \
  -d '{
    "cmd": "CMD_STOP",
    "session_id": "TEST_SESSION_001",
    "node_id": 1,
    "config": {
      "reason": "测试完成"
    }
  }'
```

**预期输出**：

```json
{
  "session_id": "TEST_SESSION_001",
  "node_id": 1,
  "cmd": "CMD_STOP",
  "phase": "STOPPED",
  "last_updated_ms": 1711345690000
}
```

**检查点**：

- ✅ HTTP 状态码 200
- ✅ `phase` 变为 `"STOPPED"`
- ✅ `capture_running` 为 false

**查看终端 1 日志**：

```
INFO:edge.command:recv cmd=CMD_STOP session=TEST_SESSION_001 node=1 config=...
INFO:edge.command:state updated phase=STOPPED last_cmd=CMD_STOP elapsed_ms=10
INFO:edge.pipeline:停止 EdgePipeline...
INFO:edge.pipeline:===============================================
INFO:edge.pipeline:流水线各阶段平均耗时 (ms)
INFO:edge.pipeline:1_capture           :    12.34 ms (n=150)
INFO:edge.pipeline:2_inference         :     5.67 ms (n=150)
INFO:edge.pipeline:3_tracking          :     2.12 ms (n=150)
INFO:edge.pipeline:4_business_logic     :     0.89 ms (n=150)
INFO:edge.pipeline:===============================================
```

---

#### 步骤 8：验证测试结果

**检查状态文件**：

```bash
cat logs/state.json
```

**预期内容**：

```json
{
  "node_id": 1,
  "session_id": "TEST_SESSION_001",
  "phase": "STOPPED",
  "last_command": "CMD_STOP",
  "events_generated": 10,
  "finish_reports_generated": 3,
  "reports_sent": 0,
  "reports_failed": 0,
  "algo_events_generated": 0
}
```

**检查点**：

- ✅ `phase` 为 `"STOPPED"`
- ✅ `events_generated` > 0
- ✅ `finish_reports_generated` > 0
- ✅ `last_command` 为 `"CMD_STOP"`

**查看算法日志**（如果启用）：

```bash
cat logs/alg_events.jsonl
```

---

## 验证检查点

### 测试通过标准

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 服务启动 | ✅ | 边缘服务成功启动在 8100 端口 |
| 健康检查 | ✅ | /health 端点返回正常 |
| CMD_INIT | ✅ | 成功初始化，phase 变为 BINDING |
| CMD_BINDING_SYNC | ✅ | 成功保存绑定信息 |
| CMD_START_MONITOR | ✅ | 成功开始监控，phase 变为 MONITORING |
| 事件生成 | ✅ | 生成违规事件和冲线报告 |
| 日志记录 | ✅ | 事件正确写入 logs/events.jsonl |
| CMD_STOP | ✅ | 成功停止，phase 变为 STOPPED |
| 状态持久化 | ✅ | logs/state.json 正确保存 |

### 性能指标

在模拟模式下，预期性能指标：

| 指标 | 预期值 | 说明 |
|------|--------|------|
| 捕获帧率 | ~15 FPS | 模拟模式稳定 |
| 推理延迟 | < 10ms | 无真实模型时 |
| 跟踪延迟 | < 5ms | BYTETracker 性能 |
| 事件间隔 | ~2s | 违规事件 |
| 报告间隔 | ~8s | 冲线报告 |

---

## 常见问题解决

### 问题 1：服务启动失败

**症状**：
```
Error: No module named 'xxx'
```

**解决方案**：
```bash
pip install -r requirements.txt
```

### 问题 2：健康检查失败

**症状**：
```
curl: (7) Failed to connect to localhost port 8100
```

**解决方案**：
1. 检查服务是否真正启动
2. 查看 `logs/edge.log` 获取错误信息
3. 确认 8100 端口未被占用：
```bash
lsof -i :8100
```

### 问题 3：事件没有生成

**症状**：
```
logs/events.jsonl 文件为空
```

**解决方案**：
1. 确认 `SIMULATE_EVENTS=true`
2. 确认 `CMD_START_MONITOR` 命令成功执行
3. 检查 `logs/edge.log` 中的事件生成日志

### 问题 4：Pipeline 初始化失败

**症状**：
```
ERROR:edge.pipeline:BYTETracker 初始化失败
```

**解决方案**：
1. 确认 `ultralytics` 已正确安装
2. 如果没有 `bytetrack.yaml`，会使用默认配置，这是正常的
3. 查看完整错误堆栈

### 问题 5：状态文件损坏

**症状**：
```
WARNING:edge.command:加载 state.json 失败
```

**解决方案**：
```bash
rm logs/state.json
```
系统会自动创建新的状态文件。

---

## 扩展测试

### 测试真实摄像头

1. 修改 `.env.edge`：
```env
SIMULATE_CAMERA=false
CAMERA_DEVICE=0  # 或使用 RTSP_URL
```

2. 重新启动服务

### 测试云端上报

1. 启动模拟云端接收器：
```bash
cd /Users/luoweibin/Desktop/code/aisport
PYTHONPATH="src" python -m uvicorn scripts.mock_cloud:app --host 0.0.0.0 --port 8000
```

2. 修改 `.env.edge`：
```env
REPORT_ENABLED=true
REPORT_BASE_URL=http://localhost:8000/nodes/reports
```

3. 重新测试，查看 `logs/cloud_received.jsonl`

---

## 总结

按照本文档的步骤，您应该能够完成：

✅ 完整的 4 阶段流程测试
✅ 验证命令处理
✅ 验证事件生成
✅ 验证状态持久化
✅ 理解数据流和业务逻辑

如果遇到问题，请参考"常见问题解决"部分，或查看日志文件获取更多信息。

祝您测试顺利！
