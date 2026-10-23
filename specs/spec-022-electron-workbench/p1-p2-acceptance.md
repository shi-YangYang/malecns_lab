# P1/P2 独立验收

2026-10-23。验收者为本轮新独立代理；只审查，不修改业务实现。风险 S3，范围限定 Spec022 的遥测、冻结旁路、桌面生命周期、Electron IPC/noVNC 与回放。未重复实施者已通过的检查，未启动完整脑、VM、容器、训练或设备。

## 当前结论

**PASS：Spec022 本轮约定的 macOS 本地 Electron 正式接入完成独立验收，AC1–AC6 通过。R1 和官方 SD 启动限定返工均已关闭，没有本次范围内未处理的验收问题。**

返工后的独立验收者审阅限定源代码、已有定向检查报告、完整图配对和真实原生桌面记录，并实际查看保存的原生截图；没有重复任何测试、build、typecheck 或完整脑 probe，没有操作 UI 或启动重资源。成功结论仅覆盖已冻结的两轮纵向离散反馈与真实遥测接入，不延伸为训练效果、连续导航或真机成功。

| 条件 | 最终状态与依据 |
|---|---|
| AC1 真脑采集、身份、精确等价及预算 | PASS；完整图开关配对 4 个边界及 3 个输出精确一致，计算增幅 0.166587%，RSS 增量 62.4375 MiB |
| AC2 有界归档、真实显示、完整性与回放 | PASS；R1 修复通过；成功 run 32/32 批完整、零丢批，实际离线回放第 28 批与原始群组均值及脉冲一致；旧 CPU 活动不进入正式视图 |
| AC3 IPC/noVNC/路径隔离 | PASS；限定源码与 Node 定向检查通过，原生窗口已实际连接独立 noVNC 视图 |
| AC4 准备、停止、失败、关闭与资源保护 | PASS；生命周期源码及隔离故障检查通过，真实失败记录保留且清理资源；成功 run 两轮均 STOPPED，最终停车与资源关闭均确认 |
| AC5 同一 run 原生双侧、两轮及停车 | PASS；`desktop-20261023-034602-5d90cd78` 原生观察与封存记录一致，真实位移 0.381063869 m / 0.446197222 m，运动后新观测成立 |
| AC6 必要检查及独立验收 | PASS；沿用已通过的相关子系统检查，R1、SD 限定返工分别独立复核，最终真实证据复核通过；本验收未引入新的实施任务 |

## 原生联调最终证据（2026-10-23）

成功目录为 `data/x2_sim/agibot/data/var/desktop-20261023-034602-5d90cd78`，`run_id=e2e5ba357a674e8895c8279ce58264e9`。

- `bridge_summary.json`：`success=true`、`new_observation_after_motion=true`，两轮均为真实 `FORWARD`，位移分别 0.38106386918153634 m、0.4461972223433889 m，分别获得 `owned_core_STOPPED`。最终 `final_stopped=true`、`active_forward=false`、`cleanup_completed=true`，无 core failure。
- `worker_result.json` 与遥测 manifest：同一 run 和执行身份，完成两帧，`parameters_unchanged=true`，原始连接计数哈希前后一致。群组统计取自 22 个真实群组，逐步脉冲范围为输出神经元 32/33；没有将群组均值宣称为全脑逐步脉冲。
- `telemetry/summary.json`：32 批全部归档，`complete=true`、`dropped_batches=0`、`errors=[]`，总计 428612 bytes；与预算及 batch 身份一致。
- `desktop_session.json`：`completed`、`motion_started=true`、`stop_confirmed=true`、`resources_stopped=true`。运行根及 `logs/desktop_runs/<run>/summary.json` 一致记录宿主 87.507982 秒、VM 80.332176 秒，未放宽原 240/120 秒预算。
- `logs/desktop_runs/<run>/sd-attempt-01.json` 保留首次官方拒绝，第二次 attempt 保留返回码 0 和 `Robot mode set successfully`。这次真实执行证明有限重试可继续至官方明确就绪；不倒推旧未留响应记录的根因。
- 已查看 `logs/desktop_workbench/live-running.jpg`：同一 Electron 窗口左侧为实际 noVNC 官方仿真，右侧为实时预置第 12 批，0 Hz、无选定脉冲，与 `000012.json` 一致。协调者报告结束时观察到第 32 批 13.3 Hz 与脉冲；原始 `000032.json` 确为 13.333333 Hz、1 个真实 spike。
- 已查看 `logs/desktop_workbench/replay.jpg`：资源关闭后原生 UI 选择成功记录并切换至第 28/32 批，DNg02_L 显示 53.3 Hz、栅格有 5 个脉冲；`000028.json` 对应 53.333333 Hz、5 个带真实 body_id/step 的事件，神经时间 135–140 ms。左侧明确无仿真视频，没有将当前或历史截图当作对应回放画面。

先前取消、UI 准备超时、SD 未确认及 `desktop-20261023-034424-ffa085b8` 姿态门槛失败记录均保留，不计作闭环成功。已审阅后者宿主汇总：`motion_started=false`、`stop_confirmed=false`、`resources_stopped=true`，证明失败清理而非实际停车。停止、关闭和崩溃保护还包含隔离注入与源码证据，本次没有另开重资源逐个强制故障。

交付边界：仅 macOS 本地开发运行；安装包、签名和跨平台发布不在本轮。脑活动可逐批回放，仿真视频归档/同步视频回放未实现。仍沿官方 Sim + MC 路线，使用 odom 虚拟标记和两轮有界纵向控制；连续控制、真实相机、真机与新训练均不在本轮。每批 5 ms 是神经模拟时间，不是实时控制周期或渲染帧率。

## R1 与完整图配对复核（2026-10-23）

- **R1 PASS**：实时快照现在消费 `summary`，校验 schema、run_id、execution_id、状态、错误和丢批字段；切换 run 或回放时重置，避免串用旧汇总。即使最新批次未变化，也刷新错误、最终完整性与累计丢批。运行中的 `complete=false` 本身不误报；终态或回放的不完整记录可见。脑活动采集异常与控制状态分开，不把遥测错误等同于停车失败。
- **栅格样式 PASS**：`.chart-grid` 已设置 `fill:none`，未闭合的坐标路径不再采用 SVG 默认填充。该修改没有改变数据、动作或已认可的布局。
- **AC1 PASS**：审阅 `logs/desktop_workbench/telemetry-pair.json`、`actual-integration.md`，以及 `data/x2_telemetry_off_20261023_r1` 与 `data/x2_telemetry_on_20261023_r1` 内的原始 `comparison.json`、`result.json` 和 worker 记录。两次使用相同输入、运行环境与 wrapper 源码，各自 4 个递归边界和 3 个完整输出均与同一 baseline 逐元素精确一致。计算耗时为 9.882742 秒 / 9.899205 秒，增加 0.166587%；峰值 RSS 增量 62.4375 MiB，满足预定 15% / 128 MiB 门槛。开启遥测记录 22 个群组、32 个选定神经元，42/42 批、零丢批、无采集错误且 `complete=true`，总计 435440 bytes。该证据是一次同机顺序配对，不是统计性能或实时性保证。
- **当时联调待完成，现已由上节关闭**：准备超时或官方 SD 未确认记录仍保留，已有 noVNC 与真实脑预置批次的观察没有被单独当作两轮闭环。没有动作的清理记录不计作实际停车成功。

以下保留首次独立审查事实及当时状态，已由上述 R1/AC1 复核更新的条目不再作为当前未解决问题。

## 首次静态审查结论

**整体暂不通过：1 项具体显示完整性问题待返工，真实完整图配对与原生桌面联调仍待证据。**

已完成的源码审阅没有发现阻断受保护的离线完整图 on/off probe，或协议约定有界 GPU 仿真联调的数值/停车/资源所有权问题。此结论只允许继续收集真实证据，不等于 AC1、AC4、AC5 已通过。

## 必须修复

### R1：实时视图隐藏采集错误及最终不完整状态

`apps/desktop/src/main.ts` 的 `poll()` 读取 `Snapshot.summary` 后完全不消费它；只有打开历史记录时才检查 `opened.summary.complete`。实时数据质量仅使用 `current.dropped_batches`，而采样线程发生错误或最后若干批被丢弃时，`latest.json` 可以保持较早批次，真正的最终错误/丢批数只在 `telemetry/summary.json` 与 worker result 中。

因此一次实验的控制逻辑正常结束、采集失败或尾批丢失时，当前窗口仍可显示“实时/最后一批”和旧的丢批数，并把实验标成完成；用户必须重新选择回放才看到“不完整”。这违反冻结协议要求的真实错误、丢批与不完整可见性。

返工范围限定实时展示：消费与当前 manifest/run 相匹配的 summary；即时呈现采集错误；在终态呈现不完整档案；显示累计/最终丢批数量而非仅依赖陈旧 latest batch。`running` summary 的 `complete=false` 本身是正常的，不能因此在运行中误报失败。不要求改变冻结数值或将显示故障与实际停车混为一谈。

## 已审阅的实现事实

- 数值旁路：`FrozenInference._step` 的 `spikes` 仅作为只读输入，私有计数与选样缓冲不覆盖脑状态；50 步既有错误检查后一次 CPU 传输，保留既有数值算式和运算顺序。是否逐元素精确等价、耗时增加是否不超过 15%、RSS 增量是否不超过 128 MiB，必须由完整图配对记录确认。
- 映射与归档：群组取自验证后的完整参考图映射，按名排序，全部成员为分母；选定输出神经元最多 32 个。记录携带 run、execution、checkpoint、图和源码身份，实际 step 和 dt，动作尚未形成时 decision_sequence 为 null。队列和字节预算有界，归档丢批及错误不标为完整。
- 生命周期：固定 Python 入口和输出根；完整脑真实 ready 后启动 GPU VM；官方 Reset/Run 后再启桥。stdin EOF、非法命令、父进程丢失、信号和资源时限进入停止路径。停止先封印 abort，再等待当前 run 的实际零速发布、STOPPED 和 summary，最后清理自有资源。未开桥时不声称确认停车；清理失败保留 failed 和 resources_stopped=false。
- 所有权：运行中的私有 VM/容器不接管，固定 CPU 仿真只读预检；不改默认 Docker context、不操作 Noria。容器或 VM 启动中失败时保留已获得的所有权供清理。相关故障路径目前只有源码及隔离注入证据。
- Electron：主 renderer 和 noVNC 均无 Node；主 renderer 使用隔离 preload，IPC 校验具体主 webContents、主 frame 和精确应用 URL。noVNC 单独内存 session、无 preload，仅固定 localhost HTTP/WebSocket 端口，拒绝导航/弹窗/下载/权限。
- 读取边界：记录 ID 固定格式，封存状态要求资源清理确认；目录真实路径和归档字节数受限，JSON 单次读取固定上限，回放不补造缺失批次。左右图保留真实胞体按 body_id 映射，群组均值和逐步脉冲栅格分开；旧 CPU 活动文件不再作为正式数据源。
- 关闭：窗口关闭先请求停止，只有资源确认和子进程实际退出后正常退出；意外 exit=0 不当成完成；主 renderer 或 noVNC renderer 崩溃请求停止。实际原生窗口行为仍需联调。

## 首次检查证据与当时待验收状态

采用实施者报告中已完成的定向结果：P1 首次 40 passed，summary 改动后局部 10 passed；runtime 20 passed；Electron Node 11 passed；桌面一次 build（素材准备、TypeScript、Vite）通过。独立验收没有机械重跑这些检查。

| 条件 | 当前状态 |
|---|---|
| AC1 实际完整图数值等价与性能/RSS预算 | 待协调者 probe |
| AC2 有界归档、真实显示、完整性和回放 | 静态主要路径合理；R1 待修复 |
| AC3 IPC/noVNC/路径隔离 | 静态及定向检查可接受；原生连接待观察 |
| AC4 准备/停止/失败/关闭与资源保护 | 静态及注入检查可接受；真实停车清理待观察 |
| AC5 同一 run 原生 Electron 双侧显示、两轮和停车 | 待实际联调 |
| AC6 必要检查与独立验收 | 检查已有证据；整体未通过 |

修复 R1 后由新独立验收者复核限定变更，并结合协调者的真实配对与原生联调证据形成最终结论。不要把本次静态可执行判断写成全部验收 PASS。

## 官方 SD 限定修复复核（2026-10-23）

新独立验收者只审阅 `x2_desktop.py` 的可选命令记录和 `wait_for_official_sd`、新增的五种隔离场景以及实施报告末段。**限定修复 PASS，可继续协议内的有界真实联调；不代表实际 SD、两轮闭环或整体验收通过。**

- 重试最多五次，局部截止取当前活动截止和 25 秒的较早者；单次容器内 `timeout` 与宿主命令截止按剩余时间收缩，预留容器命令终止余量。失败间隔最多 1.5 秒。240 秒宿主、120 秒 VM、30 秒停车清理预留及 65 秒 UI 准备截止保持原值，未通过增加总体运行时间绕过失败。
- 命令等待与重试间隔均以约 0.1 秒粒度检查原有停止、父进程失联、内存和活动时限保护；间隔与下一次调用前也检查完整脑是否退出。取消或超时会终止本次宿主命令并进入原有停车/清理路径，不继续下一次 SD。
- 成功仍要求官方命令返回码为 0 且响应包含 `Robot mode set successfully`。只有此条件满足，调用方才设置 `simulator_ready`；没有以服务发现、启动进程或重试次数代替成功证据。
- 每次实际发起的 SD 调用正常返回时保留 stdout、stderr、returncode；启动异常保留 null 返回码和异常，命令中止后收集已有输出，均经 `finally` 写入独立 attempt JSON。未发起调用的预算拒绝不伪造 attempt。此日志足以区分官方拒绝、命令错误和超时，仍需后续真实响应确定失败原因。
- 采用实施者已完成的定向结果：`5 passed, 20 deselected`。本次未重复运行测试、build 或 typecheck，未启动完整脑、容器、VM、ROS 或设备，未修改业务代码。

保留 `desktop-20261023-031503-c92ea250` 的失败事实：旧实现没有保存三次 SD 的官方响应，不能声称已经证实为 MC 启动时序问题。下一次联调成功或失败都应保留新 attempt 日志及最终资源关闭记录，由最终验收者结合实际证据更新整体结论。
