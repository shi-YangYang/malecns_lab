# M4 实施报告 — 新脑桌面接缝

状态：工程实现与定向轻量检查完成，待新的独立验收。2026-10-23。风险 **S3**（跨进程身份、时钟、IPC 和控制生命周期）。未运行完整脑、GPU VM、ROS、仿真、训练或设备；P3/P5 未通过，不将工程检查称为多任务能力证据。

## 实际修改

- `learning/x2_behavior_session.py`：独立 transport schema、主机单次 nonce 观测交换、唯一 `NeuralSession`、真实三头及 DN 特征归档、持久神经状态、主机 `DecisionGate`。用户任务改变时旧推理输出仍归档，旧结果拒绝执行，已推进神经状态不回滚。没有教师、任务到动作表或旧策略回退。
- `learning/x2_behavior_telemetry.py`：复用旧只读 spike 采样/设备归约/有界写入算法，使用新的 behavior manifest/batch/summary schema，明确来源为同一 `X2BehaviorPolicy._step` 的实际 MPSLIF spikes。关联 run/session/request/observation/decision/checkpoint/input_map/已有执行；保留 50 步一批、22 群组、32 输出神经元预算与丢批统计。未声称新策略为 FrozenInference。
- `experiments/x2_behavior_bridge.py`：仿真专用 ROS bridge；实际 odom/MC/关节反馈，独立 remote gate 与 `SkillExecutor`，完整 `ExecutionEvent` 包装。保留原始命令归属、`fact`、`finish_decision_seq`/`finish_request_id`、实际 MC 发布及原始反馈。HOLD 收束、普通 END 和紧急停止分开；wave 无服务调用路径，能力未开放。
- `experiments/x2_behavior_desktop.py`：固定候选配置、完整脑单锁、受限子进程与 GPU 仿真生命周期。复用返工后的 `SkillRuntime` 独立清理，而非让日志失败跳过关闭。renderer 只有受限 task/continue/stop；配置/CLI 从主机选择真实新 checkpoint，并在启动脑/VM前校验。没有候选立即报错，允许固定候选以明确 qualification 模式收集仿真证据，不形成“先通过仿真才允许验证”的循环门槛。
- `apps/desktop`：保留左仿真右脑布局，增加五个简体中文任务按钮、候选验证/计算/拒绝/执行状态和独立“紧急停止”。没有恢复已删除的大标题、底部播放条或历史小字。新旧遥测 schema 分别核验，旧记录仍可回放；过期批次显示“等待新批次”。
- `tests/test_x2_behavior_desktop.py` 与 Electron controller 测试覆盖上述纯软件边界。没有修改 M1/M2/M3 拥有文件、历史数值源、SDK、旧 schema 或科研资产，没有 commit/push。

## 时钟与执行边界

主机发唯一 nonce，远端在自身 monotonic 上签发观测并保留原始 odom/MC 年龄。主机只接收一次匹配的 run/session/request/nonce/sequence，用自身发出至收到的 RTT 作为传输上界（最多 2 秒），来源年龄加 RTT 不得超过 0.75 秒。读取旧文件不会刷新 nonce、时间戳或来源年龄。

原始脑观测使用主机本地接收时间，host gate TTL 为 30 秒。回送决策必须绑定原 nonce；VM 在原 nonce 的 Linux 签发时刻起检查 30 秒预算，包含传输及脑计算，绝不比较 macOS/Linux 单调时钟。MC 动作另使用本地接收后的既有 1 秒发布/最多 2 秒目标上限及 12 秒停稳观察。任务请求另记录主机接收时间，最多 120 秒，不因连续决策续期。

远端事件传入 host gate 时将 `received_monotonic` 转为主机真实收到时间，Linux 原值留在 `raw_feedback.remote_received_monotonic`。同一状态的 skill admission、preset ACK、finish 请求均保留 fact，不把 ACK 当完成。

推理后会先取得新反馈以排空实际完成事件；这次采样不会替换原神经输入或给原决策续期。若执行归属在决策传输期间已改变，远端拒绝并保护退出，不能把原本针对在途技能的 FINISH 重新解释为一个新的技能。这种保守退出会降低任务成功率，必须计入真实 P5 故障/失败，不能静默重试或补答案。

## 资源与记录

默认新运行预算：总墙钟 600 秒、VM 共驻留 240 秒、35 秒清理预留；最多 50 次决策，每请求最多 8 次/120 秒。父进程检查实际 RSS 与至少 1 GiB 系统预留；RSS/MPS 封顶取 6 GiB 与启动时可用内存减预留的较小者。MPS 分配器限制与周期采样、主机 RSS CSV、脑原始事件、控制请求/执行事件、源码与官方资产哈希及最终神经状态均归档。

只有本次启动的私有 GPU 容器/VM/脑子进程归本运行清理。保留旧 CPU 仿真关闭状态，不改默认 Docker context，不停止 Noria。命令损坏、管道断开、资源超限、观测过期、任务过期、遥测失败及显式紧急停止均进入有界保护/清理；未收到实际停稳确认不会声明停止成功。

## 已执行检查

1. 新 Python 定向检查最终为 **11 passed in 0.65s**，exit 0。覆盖独立时钟、旧文件/nonce 重放、来源年龄、未开放 wave、新任务使旧输出失效且保持神经状态、事件原始/finish 归属、无副作用 preview、任务输入白名单、缺候选/路径限制、日志失败仍清理，以及 phase 字段组合。只使用显式 CPU 测试对象，没有创建图或运行 MPS。
2. `node --test electron/controller.test.mjs electron/boundaries.test.mjs`：**12 passed**，exit 0。覆盖五枚举任务和普通 END、固定 CLI、来源/路径/大小边界、停止与实际子进程退出、旧记录读取。未改动这些检查通过后的 Electron IPC 代码，未重复运行。
3. `npm run build`：通过（包含该桌面的 assets、TypeScript 与 Vite）。随后增加遥测等待状态，因此仅重建受影响桌面一次，最终 `built in 407ms`，exit 0。
4. `git diff --check` 通过。没有全仓 pytest、全量 lint、旧数值等价重跑或任何设备动作。

Python 首次 9 项通过后，加入完整事件包装、独立资源记录、请求期限与执行归属竞态保护；最后修正启动 phase 与 qualification 字段重复组合，并新增直接回归，所以相应小文件随相关代码变化重跑。以上是实际轻量检查，不是运行资格。

## 下一阶段与限制

- 尚未选取/创建桌面候选配置，`data/x2_behavior_desktop.json` 缺失会明确报错。真实候选、预算登记和运行由协调者按 M2/M3 的实际准入安排。
- 当前入口只提供 candidate qualification，不提供已合格部署模式。持续速度技能也不能因这里可调用而被称为 P2 通过；wave 的紧急取消资格仍缺失，默认无能力。故当前五任务不能全部完成，主线缺口明确保留。
- 无相机时视觉为缺失；桥接只填真实 odom yaw/vx/wz/MC，不以任务名称伪造方向目标、完成进度或成功。`heading_error_rad=0` 代表当前没有外部方向目标输入；专项学习/实际评估必须对齐该观察分布或登记真实交互目标扩展。
- “脑选择 HOLD 后等待新任务”只记录 `brain_hold` 和 `task_success=null`；每请求达到上限同样不记成功。真实 25 段任务、任务顺序/方向/实际动作、失败恢复和同步视频归档均需后续实测。
- 采集的是选定群组/32 个神经元，不是全脑逐步记录；脑时间、GUI 刷新和控制节奏分别记录，不承诺 60 FPS 脑推理或硬实时。
