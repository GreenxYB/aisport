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
摄像头：默认 `SIMULATE_CAMERA=true` 生成假帧；有设备时设为 false 并配置 `CAMERA_DEVICE=0` 或 `RTSP_URL=...`。服务启动即打开采集；本地调试可设 `DISPLAY_PREVIEW=true` 看到 cv2.imshow 画面。

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
