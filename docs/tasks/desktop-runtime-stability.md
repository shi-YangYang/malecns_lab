# 桌面启动与运行失败排查

2026-10-26。用户授权先本地提交脑图修复（`5b40c43`），再定位并修复启动或运行期间的报错。不训练、不更换模型或动作策略、不修改官方仿真参数、不关闭无关服务。

## 已有证据

- `desktop-20261026-080505-03e8bd4c`：控制桥记录 `JSONDecodeError:Expecting value: line 1 column 473 (char 472)` 后退出；最终 `observation_request.json` 为完整 477 字节对象，第 472 字节开始是 `null}`。原异常未保存文件名及当时原始字节，不能据此断言是哪一个文件或虚拟文件系统故障。当前写入已有临时文件替换，但读取没有短暂不完整读取恢复或文件名诊断。
- 同场桥摘要的 `status=COMPLETED` 是最后动作状态，`reason` 却是 JSON 异常；主机只报告 `Behavior bridge exited`，次要错误收集又因状态为 COMPLETED 丢掉原因。
- `desktop-20261026-080448-d6a81782`：脑初始化仅获 1,519,517,696 字节（1.42 GiB）MPS 上限，分配失败；相邻已加载场次采样驱动内存峰值为 1,525,497,856 字节。当前启动准入只要求预算大于零，无法提前阻止明显不足的加载；界面未显示子进程已记录的内存原因。

## 限定修复与验收

完善有界通信文件读写：完整写入同步后替换；只对 JSON/编码的瞬态读取错误做极短、有次数上限的重读，持续损坏仍失败并带文件名，不缓存旧动作、不更改 TTL/身份校验。子进程退出时在停车前读取当前运行的具体原因，保留动作保护首因。完整脑启动至少需要 2 GiB 进程预算，同时保留既有 1 GiB 系统余量；不足时提前说明，不取消 MPS 上限。

按受影响的通信/运行管理子系统验证：定向覆盖瞬态及持续损坏、大小/链接限制、错误首因和内存准入；独立只读审查。修复后做一次有界 Electron 启动—单次前进—停止回归（资源准入允许时），不进行动作矩阵或长期实验。软件检查与实际仿真结果分别记录。

## 实施与验证结果

- `learning/x2_behavior_session.py::atomic/read_json`：写入先 flush/fsync 再替换；读取绑定单个文件描述符、fstat 与实际读取长度均限制 256 KiB，拒绝符号链接及非普通文件。JSON/UTF-8 解析失败最多读取三次、两次各等待 2 ms；不缓存旧包，持续损坏保留文件名、字节数及解析原因并停止。既有 nonce、会话及 TTL 校验未放宽。
- `runtime/x2/behavior.py::child_exit_reason/collect_secondary_errors`：子进程退出时在清理前提取原因；动作保护仍优先；桥最后动作即使 COMPLETED 也收集进程异常。脑初始化的 MPS 错误直接转为可读内存提示。
- `experiments/x2_behavior_bridge.py::execute`：进程 `process_status` 与最后动作 `status` 分开；异常退出不再仅因停车成功就返回 0。正常用户取消且停车确认仍返回 0。桥与脑摘要增加运行身份。
- `brain_resource_budget`：低于 2 GiB 脑预算提前拒绝，仍预留 1 GiB 系统内存，原 6 GiB 上限及 MPS 分配器保护保留。2 GiB 是基于现有固定脑记录的最低准入，不代表所有后续内存需求均有保证。

定向 pytest 首轮 86 通过、5 失败。其中 4 项为已有启动测试 fixture 缺 `automatic_start` 且仍断言旧英文异常；只更新该 fixture/错误契约后，单独重跑 4 项全部通过（30/12/2 秒截止和守护断言保留）。剩余 1 项 `test_preview_has_no_heavy_imports_or_startup` 为既有 `control/__init__.py → adapters/body.py → numpy` 导入与“预览不导入 NumPy”断言冲突；这些模块本轮未改，未扩大范围重构。合计 90 项有通过记录，包括本次 14 个新增回归；不声称整套全绿。独立只读审查 PASS，未重复通过项，未运行前端 build。

实际 Electron 点击开始产生 `desktop-20261026-082316-41dd9d5b`：可用内存 2.56 GiB，加载脑及启动仿真前按新门槛拒绝；界面完整显示所需/当前内存及重试提示，开始按钮恢复可用。主机摘要 `resources_stopped=true`、`errors=[]`、`secondary_errors=[]`，未产生脑就绪或桥记录。截图 `logs/desktop-runtime-memory-check.png`。

本次没有强制关闭其他程序释放内存、没有关闭分配器保护、没有训练或启动物理仿真。因此运行中跨虚拟机通信的实际改善以及前进—停止回归仍未实测；不能将有界重读测试等同于已复现并解决底层共享文件系统根因。既有动作保护（如 `wave_yaw_limit`）和未通过的动作精度记录不在本轮中改写或放宽。

## 用户要求继续实测

用户随后明确要求实测并授权临时退出 Chrome。正常退出该浏览器窗口后可用内存达到 3.57 GiB，未更改内存保护或停止其他服务。从真实 Electron 点击开始，产生 `desktop-20261026-083912-8fad4108`：完整脑初始化成功并写出 `brain_ready.json`；首条官方 SD 准备命令的容器 15 秒 timeout 尚未返回时，主机 17 秒预算先超时，`sd-01.json` 保留完整命令和空 stdout/stderr。没有进入行为桥或发送前进请求，不能计作动作通过。

该场清理时 `lighter stop` 未在 6 秒内完成，紧随的状态检查未确认，原摘要如实为 `resources_stopped=false`。后续只读检查确认同一私有 VM 已停止、socket 消失；不改写旧摘要。需修正 SD 冷启动等待和迟到清理确认的有界处理，再复测；不放宽实际站稳或动作保护条件。

针对该实际失败，SD 客户端改用原有完整 25 秒准备窗口，主机命令超时注明对应记录；VM 只请求停止一次，随后在原 35 秒清理期限内有界确认，预留 8 秒清理子进程，同时要求停止提示和私有 socket 消失。`vm-shutdown.json` 保留命令超时及最终确认，两者分别记录。针对启动与清理的定向检查 13 项通过；独立只读审查通过。

复测 `desktop-20261026-090054-c796d3d7`：脑成功加载，官方 SD 客户端正常退出但明确返回 `Failed (code=0)`；未发送前进请求。此前只启动一次客户端的改动未覆盖该启动阶段明确失败，需要恢复有界重试。清理已实际正常完成，`resources_stopped=true`、`errors=[]`，界面开始按钮恢复。官方客户端内部本身会重发 ROS 请求，因此这里的客户端启动次数不等于网络请求次数。

最终 SD 逻辑：固定原 25 秒总窗口，最多 5 次客户端启动；只在客户端正常退出且明确返回 `Failed (code=...)` 时允许下一次。主机超时、非零退出、响应不明均停止，不重发未确认请求。新实施者完成局部修复，4 项 SD 定向检查通过（包含新失败后恢复用例）；独立只读复审 PASS，未重复其他已通过检查。

`desktop-20261026-090721-b68c9708`：第三次 SD 成功，自动 Pause/Reset/Run 成功，随后在 Electron 真实画面采集检查中失败，npm 日志为 `Current display surface not available for capture`。20 秒后前端主动停止，主机摘要因此记为 `user_emergency_stop`，这不是人工点击的证据。清理成功，未发送动作。该错误的界面原因传播以及原生显示层问题尚未修复。

## 最终实际回归结果

将 Electron 窗口置前后，从界面点击开始产生 `desktop-20261026-091042-38a5a08e`，沿完整脑、官方仿真和控制桥实际运行：

- `brain_ready.json`、`sim_startup.json(status=ready)`、`bridge_ready.json` 均存在；第二次 SD 成功。未训练或改变模型、官方参数与动作保护。
- 实际点击一次“前进 1 米”，`task_requests.jsonl` 关联请求 `request-637f9be8445f4eb5814490929d2aba19`。`behavior_events.jsonl` 有 4 次神经输入与原始决策、执行接收、完成应答；最终 `request_waiting.task_success=true`。
- `bridge_summary.json`：`status=COMPLETED`、`advance_target_reached=true`、`motion_state=STOPPED`，最终前向距离 **1.2284028493 m**、横向距离 **-0.0375289778 m**。这是达标后停稳回归，存在约 0.23 m 超调，不是精确一米控制的证据。
- 界面显示“动作已完成并确认停稳”，其他动作按钮恢复；随后实际点击顶部“停止”。主机运行约 138.44 秒，最终 `state=cancelled`（主动停止）、`stop_confirmed=true`、`resources_stopped=true`、`errors=[]`、`secondary_errors=[]`。桥 `process_status=cancelled`、`base_stop_confirmed=true`、`wave_stop_confirmed=true`；后者只表示停机确认，本次没有执行挥手。
- `simulation_capture/summary.json` 封存 192 帧、采样失败 0；`simulation_video/summary.json` 视频封存完整。采集帧为真实 MuJoCo 画面，构图有裁切，不据此声称展示效果完整。`logs/desktop-runtime-action-completed.png` 保存主界面的动作完成状态与脑活动。
- 本轮未出现 JSON 解析异常或非预期控制桥退出；停止后另外只读确认私有 lighter 未运行。默认 Docker/Noria 未操作。

边界：只完成一次前台启动—前进—停稳—停止，不代表全动作、后台/遮挡窗口或长时间稳定性通过。窗口置前后成功支持显示状态相关性，但尚不能证明原生显示层失败的底层原因；没有取消真实采集检查或凭推测增加恢复代码。通信瞬态重读的底层共享文件系统成因仍未证实。既有 NumPy 导入检查失败仍保留。

## 显示层启动恢复（用户随后授权）

用户要求继续解决原生显示层无法采集。按 S2 限定桌面启动与显示生命周期：仅识别 `Current display surface not available for capture`，在当前启动仍有效时最多一次恢复并置前现有窗口，随后必须重新通过真实 `capturePage`；保留 20 秒源确认、总启动期限和运行阶段停车。停止、关闭、会话替换后不得恢复或继续。新运行记录恢复是否触发与最终结果；不把仅靠前台启动通过当成已验证恢复分支。

验收范围：相关启动/媒体编排检查、独立只读审查；实际 Electron 在后台准备时复现并检查恢复记录，能够启动后停止并确认清理。不重训、不发额外动作、不更改官方仿真和默认 Docker。

已实现：`electron/main.mjs::confirmCaptureSource` 仅为该确切原生错误附 `DISPLAY_SURFACE_UNAVAILABLE`；`startup.mjs::finishStartup` 每次启动最多调用一次恢复，原 deadline 与 stop/close/run 检查保留，恢复后必须再次通过真实非空采集。`recoverStartupSurface` 只显示、置前当前窗口，不重新加载页面、不修改运行期停车。失败原因通过当前 run 的 `startupSnapshot` 持续显示，后端摘要不改，停车和资源清理字段原样保留。`desktop_startup.json` 有 run_id、至多 32 个合并事件，异步写入，写失败不改变控制结果。

`node --test electron/startup.test.mjs electron/media-window.test.mjs`：24 项通过；覆盖一次恢复后成功、持续失败仍超时、停止/关闭/换 run/截止后不恢复、运行期错误停车及终态展示不改原证据。独立只读审查 PASS；没有运行完整测试或前端构建。

实际场次 `desktop-20261026-093252-3e3cd60c`：重启修复后的 Electron，点击开始并将现有 Chrome 置前遮挡；约 29.6 秒后源检查成功，完整脑、仿真和桥进入 running，UI 显示真实 MuJoCo 画面，未发送动作。`desktop_startup.json` 为 `continue_requested`，没有 `surface_recovery_requested`，因此本轮原错误未复现，**不能称为自动恢复分支已实际触发成功**。点击停止后 `stop_confirmed=true`、`resources_stopped=true`、`errors=[]`、`secondary_errors=[]`；243 帧封存，采样失败 0。截图 `logs/desktop-surface-background-start.png`。

完全遮挡的第二种复现没有执行启动：切换 Chrome 窗口时用户正在使用该浏览器，未继续改变其状态。没有反复启动仿真追求复现。当前结论是恢复机制与防误恢复的软件检查通过、一次普通后台启动实际通过；原生故障的确切触发原因及真实故障后的自动恢复仍缺触发证据。Electron 官方同类问题 [PR 52844](https://github.com/electron/electron/pull/52844) 已合入 45 分支，不能仅凭相同错误字符串断言本机仍是该缺陷；本次未升级依赖或修改图形驱动。
