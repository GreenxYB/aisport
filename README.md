# AI Sport Project (Python Edge/Cloud Scaffold)

## 快速启动 Edge
```
cd e:\computer\CursorProject\AISportProject\python
$env:PYTHONPATH="src"
python run_edge.py
# 或 uvicorn run_edge:app --host 0.0.0.0 --port 8100 --reload
```

日志输出：`logs/edge.log`
状态持久化：`logs/state.json`（自动保存/恢复）
摄像头：默认 `SIMULATE_CAMERA=true` 生成假帧；有设备时设为 false 并配置 `CAMERA_DEVICE=0` 或 `RTSP_URL=...`。服务启动即打开采集并默认 `DISPLAY_PREVIEW=true` 弹出 cv2.imshow 画面（无 GUI 时可设为 false）。`DISPLAY_MIRROR=true` 会将预览和快照左右镜像。
事件模拟：默认 `SIMULATE_EVENTS=true`，在 `CMD_START_MONITOR` 后按 `EVENT_INTERVAL_SEC` 生成模拟违规事件，落盘到 `logs/events.jsonl`。可选 `SIMULATE_FINISH_REPORTS=true`，按 `FINISH_INTERVAL_SEC` 生成模拟冲线报告。
事件上报：上报 URL 在 `REPORT_BASE_URL` 配置（默认 `http://localhost:8000/nodes/reports`），事件会 POST 到 `/violation` 与 `/finish`。未启用云端时保持 `REPORT_ENABLED=false`，失败记录写入 `logs/events_failed.jsonl`。
本地接收器：运行 `python -m uvicorn scripts.mock_cloud:app --host 0.0.0.0 --port 8000`，收到的事件写入 `logs/cloud_received.jsonl`。
失败重放：`python scripts/replay_failed_events.py --base-url http://localhost:8000/nodes/reports`（成功会归档到 `logs/events_failed.archived.jsonl`，失败仍保留在原文件）
算法框架：`src/edge/app/services/algorithms` 提供占位算法接口与 `AlgorithmRunner`。离线回放：`python scripts/offline_playback.py --video <path>`
人脸绑定（Baidu AIP）：在 `.env.edge` 或环境变量设置 `BAIDU_APP_ID / BAIDU_API_KEY / BAIDU_SECRET_KEY / BAIDU_GROUP_ID`，并确保安装 `baidu-aip`。
测试 Baidu 接口：`python scripts/test_baidu_face.py --image <path> --group <group_id>`

## 手动发指令（Postman）
导入 `docs/protocols/postman_collection.json`，包含：
- CMD_INIT / CMD_BINDING_SYNC / CMD_START_MONITOR / CMD_STOP
- STATUS 查询

## 协议
详见 `docs/protocols/run_protocol.md`（云↔边指令与回报）。

## 测试
```
cd e:\computer\CursorProject\AISportProject\python
$env:PYTHONPATH="src"
pytest tests/unit/test_edge_commands.py
```

## 分支策略
- `main`: 生产分支
- `dev-syq`: 开发分支（当前工作分支）
