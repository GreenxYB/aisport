# AI Sport Project

智慧校园 AI 跑步项目的 Python 原型仓库。当前仓库聚焦三件事：边缘节点服务、云边通信协议、以及起跑/冲线/人脸绑定等业务能力的调试脚本。

## 项目背景

这个项目面向校园跑步场景，目标不是单纯做视频识别，而是跑通一轮完整业务：

- 云端创建一轮跑步任务
- 开始节点、结束节点等边缘设备接入
- 云端下发初始化、绑定、开始监控等指令
- 边缘节点执行身份绑定、起跑检测、违规检测、冲线判定
- 边缘节点回传状态和事件，云端汇总成绩

当前代码属于“工程原型”阶段，特点是：

- 边缘端主流程已经能跑
- 协议和结构在持续收敛
- 真实云端、音响控制、多设备长连接还在逐步补齐
- 算法采用“先打通流程，再替换真实模型”的策略

## 第一次看项目，建议怎么读

建议按下面顺序阅读：

1. 先看本文，建立项目全貌
2. 再看协议文档 [docs/protocols/run_protocol.md](E:\computer\CursorProject\AISportProject\python\docs\protocols\run_protocol.md)
3. 看边缘入口 [run_edge.py](E:\computer\CursorProject\AISportProject\python\run_edge.py)
4. 看边缘应用入口 [src/edge/app/main.py](E:\computer\CursorProject\AISportProject\python\src\edge\app\main.py)
5. 看命令处理核心 [src/edge/app/services/command_handler.py](E:\computer\CursorProject\AISportProject\python\src\edge\app\services\command_handler.py)
6. 看协议模型 [src/common/protocol.py](E:\computer\CursorProject\AISportProject\python\src\common\protocol.py)
7. 再看算法主入口 [src/edge/app/services/algorithms/runner.py](E:\computer\CursorProject\AISportProject\python\src\edge\app\services\algorithms\runner.py)

如果你要直接调效果，优先看这些脚本：

- 起跑测试 [scripts/test_start_monitor.py](E:\computer\CursorProject\AISportProject\python\scripts\test_start_monitor.py)
- 冲线测试 [scripts/test_finish_line.py](E:\computer\CursorProject\AISportProject\python\scripts\test_finish_line.py)
- 百度人脸测试 [scripts/test_baidu_face.py](E:\computer\CursorProject\AISportProject\python\scripts\test_baidu_face.py)

## 当前系统结构

### 云端

位置：`src/cloud`

作用：

- 创建 session
- 生成初始化命令
- 接收边缘节点上报的违规、冲线、状态数据

当前状态：

- 有最小服务骨架
- 更多联调目前仍可用 Postman 或 mock 替代

### 边缘节点

位置：`src/edge`

作用：

- 接收云端命令
- 管理节点状态机
- 打开摄像头或视频源
- 执行算法与业务逻辑
- 输出预览和事件

### 公共协议

位置：`src/common`

作用：

- 定义命令模型和上报模型
- 让云端、边缘、脚本使用一致字段

## 目录说明

```text
.
├─ configs/                配置模板
├─ docs/                   协议、说明、测试文档
├─ scripts/                独立调试脚本
├─ src/
│  ├─ cloud/               云端最小骨架
│  ├─ common/              协议与共享模型
│  └─ edge/                边缘节点服务
├─ tests/                  单元测试
├─ run_edge.py             边缘服务启动入口
└─ requirements.txt        依赖列表
```

## 通信方案说明

当前仓库里为了调试方便，边缘命令接口还是 HTTP。

但协议设计方向已经明确调整为：

- `WebSocket`：作为云端和边缘节点的主链路
- `HTTP`：保留给调试接口、管理后台、快照查看
- `MQTT`：后续设备规模上来再考虑

也就是说，现阶段代码实现和长期协议目标不是完全一致的。阅读项目时请把 HTTP 理解成“调试落地方案”，把 WebSocket 理解成“下一阶段主链路”。

## 快速启动

建议环境：`conda` + Python 3.10

安装依赖：

```powershell
cd E:\computer\CursorProject\AISportProject\python
pip install -r requirements.txt
```

启动边缘节点：

```powershell
cd E:\computer\CursorProject\AISportProject\python
$env:PYTHONPATH="src"
python run_edge.py
```

服务启动后可以访问：

- `GET http://127.0.0.1:8100/health`
- `GET http://127.0.0.1:8100/status`
- `GET http://127.0.0.1:8100/preview/snapshot`

配置模板见 [configs/edge.example.env](E:\computer\CursorProject\AISportProject\python\configs\edge.example.env)

## 最常见的几种调试方式

### 1. 只验证边缘服务能否启动

```powershell
cd E:\computer\CursorProject\AISportProject\python
$env:PYTHONPATH="src"
python run_edge.py
```

### 2. 用 Postman 模拟云端发命令

导入：

- [docs/protocols/postman_collection.json](E:\computer\CursorProject\AISportProject\python\docs\protocols\postman_collection.json)

建议顺序：

1. `CMD_INIT`
2. `CMD_BINDING_SYNC`
3. `CMD_START_MONITOR`
4. `CMD_RESET_ROUND` 或 `CMD_STOP`

### 3. 单独测试起跑检测

```powershell
conda run -n torchpy10 python scripts/test_start_monitor.py `
  --input "data/samples/起跑/起跑/起跑-踩线-抢跑.mp4" `
  --output "data/sample_res/起跑/起跑/起跑-踩线-抢跑_start_test.mp4" `
  --model data/models/yolo26n-pose.pt
```

### 4. 单独测试冲线检测

```powershell
conda run -n torchpy10 python scripts/test_finish_line.py `
  --input "data/samples/冲线/冲线/多人-冲线-违规.mp4" `
  --output "data/sample_res/冲线/冲线/多人-冲线-违规_finish_test.mp4" `
  --model data/models/yolo26n-pose.pt
```

## 当前协议约定

详细说明见 [docs/protocols/run_protocol.md](E:\computer\CursorProject\AISportProject\python\docs\protocols\run_protocol.md)。

这里先抓住几个关键点：

- `task_mode` 表示任务模式，不表示设备身份
- 设备身份建议使用 `node_role`，例如 `START`、`FINISH`
- 起跑统一以 `expected_start_time` 为官方起点
- 成绩先按 `finish_ts - expected_start_time` 计算
- 上报结构统一使用 `data`
- 违规后不建议直接结束整轮，优先使用 `CMD_RESET_ROUND`

## `task_mode` 是什么

`task_mode` 是协议层的“业务模式字段”，定义在 [src/common/protocol.py](E:\computer\CursorProject\AISportProject\python\src\common\protocol.py)。

它的作用是表达“这轮任务是什么类型”，例如：

- `TRACK_RACE`
- `SUNSHINE_RUN`
- `FREE_RUN_ANALYSIS`

它不适合表达“这个终端节点是起点还是终点”。

节点身份更适合用 `node_role`：

- `START`
- `FINISH`
- `MID`
- `ALL_IN_ONE`

## 事件上报结构

当前协议统一约定：

- 列表类消息用 `data: []`
- 对象类消息用 `data: {}`

例如：

- `ID_REPORT` 使用 `data: []`
- `VIOLATION_EVENT` 使用 `data: []`
- `FINISH_REPORT` 使用 `data: []`
- `NODE_STATUS` 使用 `data: {}`

这样服务端更容易构建统一处理逻辑。

## 当前建议的业务流程

一轮正常流程建议这样走：

1. 云端创建 session
2. 节点连上云端并完成设备身份校验
3. 云端下发 `CMD_INIT`
4. 云端下发 `CMD_BINDING_SYNC`
5. 节点上报 ready 状态
6. 云端确认必需节点都 ready
7. 云端下发 `CMD_START_MONITOR`
8. 起点节点本地执行倒计时和喇叭播报
9. 结束后云端汇总成绩

如果发生违规：

1. 起点节点本地先播报违规提示
2. 节点上报 `VIOLATION_EVENT`
3. 云端下发 `CMD_RESET_ROUND`
4. 节点回到等待重新起跑状态

## 日志与输出

常见位置：

- `logs/edge.log`：边缘服务日志
- `logs/state.json`：节点状态快照
- `logs/alg_events.jsonl`：算法事件日志
- `data/sample_res/`：测试脚本输出视频

## 分支说明

- `main`：主分支，作为相对稳定版本
- `dev-syq`：开发分支
