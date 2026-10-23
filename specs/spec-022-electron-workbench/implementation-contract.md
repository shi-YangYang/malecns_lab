# P1/P2 正式接入协议

状态：已完成并独立 PASS，2026-10-23。本文件为实施前冻结的 P1/P2 协议，实际证据见 actual-integration.md，最终审查见 p1-p2-acceptance.md；不覆盖历史实测结论。

## 交付与边界

沿用左侧仿真、右侧脑图。移除原型素材驱动：默认仅静态真实胞体图；开始实验后显示同一次冻结 MPS 控制脑的活动及 GPU 官方仿真。保留群组选择、脑图操作、分栏。必要操作放在现有标题栏/面板内，不恢复用户已删除的大标题、底部播放条与历史小字。

正式入口管理已有 Spec021 的两轮纵向离散反馈；不增加训练、动作轴、连续控制、真实相机或真机。仍使用 odom 虚拟标记，不把 GUI 帧率或批次显示频率称为脑计算频率。仅 macOS 本地开发运行，安装包/签名/跨平台另定。

开发和离线检查时关闭重资源。当前实施包含必要有界联调：一个完整脑，系统至少 1 GiB 预留；一次独立遥测开关性能/等价配对及一次桌面两轮集成。只有发生具体失败才定向重试。每次结束即关闭自己启动的 GPU 容器与私有 VM，不启动旧 CPU 仿真，不停止 Noria，不重启 Docker、不改默认 context。

## 真实遥测 v1

来源是 `FrozenInference._step` 的实际 spike 张量，旁路 observer 不修改张量或运算顺序；在既有 50 步检查边界批量取回。实际 dt=0.1 ms，50 步为 5 ms 神经时间，不是墙钟时间。

只使用已验证完整图 `identity.reference_provenance.group_mapping` 内的 22 个群组。manifest 按名字排序，包含真实 body_id 字符串和全部组成员分母；缺坐标成员仍参与统计。默认从真实 33 个输出神经元中按 ID 升序选前 32 个记录逐步脉冲，明确 32/33 的范围，不假称全脑逐步归档。

每个新 run 写入 `telemetry/`：

- `manifest.json`：schema=`malecns_lab.x2_telemetry_manifest.v1`，run_id、execution_id、checkpoint_id、图/来源身份、dt_ms、groups=[{id,name,body_ids,neuron_count}]、selected_neurons=[{body_id,graph_index,group_id}]、预算与采样范围。
- `batches/000001.json` 起：schema=`malecns_lab.x2_telemetry_batch.v1`，run_id、execution_id、batch_sequence、phase（precondition/forward）、observation_sequence、decision_sequence、step_start（含）、step_end（不含）、dt_ms、host_monotonic_ns、group_rates_hz（manifest 顺序）、spike_events=[{body_id,step}]、dropped_batches。采集时尚未生成决策，decision_sequence=null，不伪称已作出动作；决策从已有 decision 文件关联。
- `latest.json`：原子替换的完整最新 batch，供显示端限频读取。
- `summary.json`：记录写入数、丢批、错误、是否完整，不能将部分档案标成完整。

采样 GPU 缓冲 50×32 逐步脉冲及群组计数；后台落盘队列上限 8 批，满时不阻塞控制线程，丢批显式计数。单批不超过 256 KiB、manifest 不超过 1 MiB、单 run 遥测不超过 16 MiB。显示端最多 10 Hz 拉取，默认 250 ms；记录显示端跳批与采集端丢批的不同含义。回放仅读已封存记录，缺批留空并显示，禁止插值生成脉冲。

冻结数值/性能门槛：同机同输入遥测开关配对，全部递归边界状态与动作输出逐元素精确一致，计算耗时增幅不超过 15%，峰值 RSS 增量不超过 128 MiB。单节点注入只验证局部旁路行为，不替代完整图证据；性能门槛失败需报告并限定优化，不能自动放宽。

## 生命周期与进程协议

Electron 主进程只启动固定本仓库 `.venv-train/bin/python -u -m malecns_lab.experiments.x2_desktop --execute --sim-only --output <固定运行根/desktop-日期时间-随机hex>`。运行目录由主进程生成，渲染器无 shell、路径或任意参数能力。Python 编排再调用 `x2_neural --telemetry`。

stdin NDJSON 仅允许 `{command:"continue"}` 与 `{command:"stop"}`。EOF、父进程失联、SIGTERM 等均走停止路径。stdout NDJSON：

```json
{"schema":"malecns_lab.desktop_event.v1","type":"status","state":"preparing","message":"正在准备实验","output":"absolute path","run_id":null,"simulator_ready":false,"motion_started":false,"stop_confirmed":null,"resources_stopped":false,"host_monotonic_ns":0}
```

state 为 preparing / awaiting_simulation / running / stopping / completed / cancelled / failed。message 为简短中文，stage 可细化；仅变化时发出。原始子进程输出落日志，不混入协议。仅在资源清理完毕后发终态，清理失败必须 failed 并保留资源状态。进程 exit=0 本身不证明停稳或成功。

步骤为：独占锁及 CPU/私有 VM 已停止预检 → 完整脑真实 ready → 启动私有 VM/固定 GPU 容器/Sim/MC/官方 SD → awaiting_simulation → 用户通过嵌入的官方 noVNC 界面 Reset/Run，点开始 → 启动两轮 ROS 桥。没有假就绪或跳过姿态门槛。

停止先写封印 host_abort，等待桥实际发布零速并记录 final_stopped=true、active_forward=false，再有界关闭本应用拥有的子进程、容器和 VM。桥未启动时 stop_confirmed=null，不冒称已测得停稳。停车超时保留失败证据，之后终止独立仿真资源；不能将强制关闭资源等同于停车确认。总预算 240 s，VM 开启至关闭预算 120 s（预留停车与清理时间）。不接管已有运行的 VM/容器，不清理无所有权的资源。

## Electron 隔离与显示

主窗口仍 nodeIntegration=false、contextIsolation=true、sandbox=true；contextBridge 只提供准备、开始、停止、状态读取、限定记录回放、左视图边界更新等具体能力，验证 IPC 的主 frame 与来源。重入准备被拒绝，关闭或崩溃先停止，不能直接杀控制进程绕过停车。

noVNC 使用独立内存 session 的 WebContentsView，仅固定 `http://127.0.0.1:6081` 与对应 WebSocket，无 preload/Node/实验 IPC；拒绝外部导航、弹窗、下载、权限请求。随左画布和窗口改变 bounds，回放或离线时隐藏。无任意URL输入。

前端按 manifest 的 body_id 映射真实胞体；群组均值着色和选定神经元的真实脉冲栅格分开展示。区分实时、回放、无数据、延迟、丢批与无坐标。神经时间、宿主单调时钟、接收时钟不混用；Electron 用宿主单调时钟计算传输延迟，页面单独记录接收时间。

## 分工、验证与停止

风险 S3：涉及冻结数值旁路、新版本遥测、Electron IPC 与系统进程生命周期。验证仅覆盖这些子系统，不跑全仓套件。

1. Python 遥测实施者负责采集、注册、离线注入测试与完整图 probe 入口。
2. 生命周期实施者负责新 x2_desktop 编排及隔离故障/清理测试。
3. 桌面实施者负责 Electron、呈现、有限文件/IPC/网络边界与定向 Node 检查；集成后一次桌面 build。
4. 协调者维护本协议，执行有界真实遥测 probe 和桌面联调。独立新验收者审阅结果/差异；失败限定新实施者返工、再由新验收者复核。未通过部分如实保留，不冒称完成。

- [x] AC1 真脑批量采集与版本身份完整，数值等价和预算有真实记录。
- [x] AC2 有界写入/背压/丢批、缺失与回放均正确，旧CPU数据不进入实时视图。
- [x] AC3 noVNC独立隔离，有限IPC与路径读取，不暴露其他系统能力。
- [x] AC4 准备/开始/停止/取消/关闭/失败均有真实生命周期，保护现有服务。
- [x] AC5 实际 Electron 观察到同一run脑活动与仿真，两轮结束并确认停车、资源关闭。
- [x] AC6 必要定向检查通过，独立验收通过，无未处理活动任务。

本次没有新的 commit/push 请求，不提交或推送。完成上述验收后停止，不自动扩展科学实验或训练。

API依据：[WebContentsView](https://www.electronjs.org/docs/latest/api/web-contents-view)、[contextBridge](https://www.electronjs.org/docs/latest/api/context-bridge)、[Electron安全指南](https://www.electronjs.org/docs/latest/tutorial/security)，2026-10-23 核对。
