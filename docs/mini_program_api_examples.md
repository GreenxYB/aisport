# 小程序接口示例

## 1. 创建任务

### 请求

`POST /sessions`

```json
{
  "project_type": "200m",
  "start_node_id": 7,
  "auto_start": true,
  "binding_timeout_sec": 15,
  "start_delay_ms": 5000,
  "countdown_seconds": 3,
  "race_timeout_sec": 120
}
```

### 成功响应

```json
{
  "session_id": "RUN_20260331_120000_123",
  "status": "CREATED",
  "created_at": "2026-03-31T12:00:00.123456",
  "project_type": "200m",
  "lane_count": 8,
  "start_node_id": 7,
  "finish_node_id": 6,
  "tracking_node_ids": [],
  "bindings": [],
  "candidate_lanes": [1, 2, 3, 4, 5, 6, 7, 8],
  "active_lanes": [],
  "binding_mode": "DISCOVER",
  "sync_time_ms": null,
  "require_bindings": true,
  "auto_start": true,
  "binding_timeout_sec": 15,
  "start_delay_ms": 5000,
  "countdown_seconds": 3,
  "race_timeout_sec": 120,
  "audio_plan": "START_321_GO",
  "tracking_active": true,
  "expected_start_time": null,
  "finished_at_ms": null,
  "terminal_reason": null
}
```

### 非法起点机位响应

例如 `200m` 不允许选 `start_node_id=1`：

```json
{
  "detail": "invalid start_node_id=1 for project_type=200m; allowed: [7, 8]"
}
```

---

## 2. 查询任务诊断

### 请求

`GET /sessions/{session_id}/diagnostics`

### 关键字段说明

- `session.status`
  - 当前任务状态
- `workflow.init_sent_to`
  - 已收到初始化命令的节点
- `workflow.binding_sent_to`
  - 已收到绑定命令的节点
- `workflow.start_sent`
  - 是否已下发开始监控命令
- `readiness.all_ready`
  - 所有必要节点是否 ready
- `nodes[].last_status.data.binding_target_lanes`
  - 起点节点当前认为需要关注的跑道
- `nodes[].last_status.data.binding_observed_lanes`
  - 起点节点当前观察到有人的跑道
- `nodes[].last_status.data.binding_confirmed_lanes`
  - 已经识别成功的跑道
- `warnings`
  - 标定文件、起跑线、终点线等 warning

---

## 3. 查询结果

### 请求

`GET /sessions/{session_id}/results`

### 响应示例

```json
{
  "session_id": "RUN_20260331_120000_123",
  "status": "FINISHED",
  "expected_start_time": 1774929605000,
  "finished_at_ms": 1774929712000,
  "terminal_reason": "All target lanes reached a terminal result",
  "results": [
    {
      "lane": 1,
      "student_id": "20230001",
      "feature_id": null,
      "recognized_name": "张三",
      "recognized_confidence": 96.5,
      "finish_ts": 1774929613200,
      "expected_start_time": 1774929605000,
      "elapsed_ms": 8200,
      "rank": 1,
      "false_start": false,
      "false_start_detail": null,
      "result_status": "OK"
    },
    {
      "lane": 2,
      "student_id": "20230002",
      "feature_id": null,
      "recognized_name": "李四",
      "recognized_confidence": 94.2,
      "finish_ts": null,
      "expected_start_time": 1774929605000,
      "elapsed_ms": null,
      "rank": null,
      "false_start": true,
      "false_start_detail": {
        "event": "FALSE_START",
        "lane": 2
      },
      "result_status": "FALSE_START"
    }
  ],
  "report_counts": {
    "id_reports": 1,
    "violations": 1,
    "finishes": 1
  }
}
```

### `result_status` 含义

- `OK`
- `FALSE_START`
- `DNF`
- `UNBOUND`
- `WAIT_BINDING`
- `RUNNING`

---

## 4. 当前小程序应遵守的请求约束

小程序请求中不应再传：

- `lane_count`
- `finish_node_id`
- `tracking_node_ids`
- `bindings`
- `student_id`
- `feature_id`
- `sync_time_ms`

这些都应由云端编排或边缘节点识别结果生成。
