# P2 生命周期实施记录

日期：2026-10-23。风险 S3；本记录只覆盖 Python 编排，真实联调与独立验收由协调者执行。

## 修改

- 新增 `src/malecns_lab/experiments/x2_desktop.py`，将 Spec021 一次性编排转成桌面管理的固定 CLI；默认 preview/help 不导入数值模型、不启动进程。
- 限定新输出为项目共享数据根内 `desktop-YYYYMMDD-HHMMSS-hex`，使用独占桌面锁；完整脑仍由既有 `x2_neural --telemetry` 与共享完整脑锁管理。
- 完整脑真实 ready 后重新核对私有 GPU VM 与 CPU 仿真已停止，再启动固定 runtime、GPU 容器、官方 Sim、MC、SD。默认 Docker context 只读取；不切换 context，不操作 Noria 或 CPU 容器。
- stdin 仅接受 continue/stop；EOF、父进程失联、信号、内存不足和超时进入停止路径。提前发送的 continue 不当作用户完成官方 Reset/Run，必须在 awaiting_simulation 之后确认。
- 状态通过固定 NDJSON 发送，并原子替换 `desktop_session.json`。原始 host/bridge 输出写到 `logs/desktop_runs/<run>/`；慢 stdout 不阻塞控制线程，写出失败请求停止。
- 停止先封印 `host_abort.json`，等待当前桥的真实零速发布与 STOPPED 反馈，同时核对 `bridge_summary` 的当前 run_id、final_stopped 和 active_forward。未启动桥时 stop_confirmed=null；超时不能宣称停稳。
- 仅关闭从停止状态启动的固定容器、私有 VM 与所拥有的进程树。资源关闭失败保留 failed 和 resources_stopped=false。若发现运行中的 VM/容器则不接管。
- 总预算 240 秒，VM 预算 120 秒；运行阶段预留最后 30 秒用于至多 12 秒停车与资源关闭。系统至少保留 1 GiB，遥测/推理本身仍沿用既有资源保护。

## 实际检查

仅运行隔离的 `tests/test_x2_desktop.py`。首次 19 通过、1 失败：入口顶层导入控制协议会间接导入 NumPy。改为执行实例初始化时导入后重跑同一文件，**20 passed in 0.08s**。

覆盖固定输出路径、单实例锁、惰性入口、EOF/非法命令/停止、父进程失联、内存预留、运行中资源不接管、零速与停稳证据、先 abort 后清理、关闭失败保留状态、未开桥取消、封存状态与终态顺序、实际命令发布标志、stdout 断连。

检查未启动 Docker、VM、ROS、完整脑、训练或真机。没有运行全仓测试、修改现有数值/控制模块或提交。

## 限制与待验收

- 未实测新 CLI 的完整生命周期，20 项隔离检查不是仿真成功或真实停车证明。
- 等待用户 Reset/Run 同样计入 VM 时间；90 秒活动窗口不足时提前停车并报告失败，不自动延长。
- motion_started 表示桥确实发布过非零指令；真实位移由既有桥记录，不由该标志推断。
- 遥测输出与 `--telemetry` 由另一实施者提供；桌面状态消费与 noVNC 隔离由 Electron 实施者提供。

## 官方 SD 启动限定返工

2026-10-23，新实施者，局部风险 S1。只修改 `x2_desktop.py` 的官方启动等待及该调用使用的可选日志捕获；相关隔离检查追加在 `test_x2_desktop.py`。

证据：`desktop-20261023-031503-c92ea250/desktop_summary.json` 记录 `Official SD did not confirm success`，VM 持续 11.0398 秒，`motion_started=false`、`resources_stopped=true`。原实现连续三次调用后失败，没有保存各次 stdout/stderr/returncode；因此不能从此记录判定本次失败根因。Spec021 r2 实施记录及 r4 `lifecycle.json` 保留了首次未成功、重试后官方明确成功的启动经验；本次修复依据该时序容错需要，不声称已经确认本次同因。

- 将官方 SD 的明确成功响应作为就绪判据，最多 5 次、局部至多 25 秒，失败间隔 1.5 秒。每次仍要求返回码 0 且包含 `Robot mode set successfully`，不以进程启动或服务发现代替动作成功。
- 局部截止取既有活动截止与 25 秒的较早者；单次官方 `timeout` 与外层命令时限按剩余预算缩短。间隔按 0.1 秒片段检查停止、父进程/内存/总时限及脑子进程退出。240 秒宿主、120 秒 VM、30 秒停车清理预留、65 秒 UI 准备截止均未放宽。
- 每次实际 SD 调用在 `logs/desktop_runs/<run>/sd-attempt-XX.json` 保存参数、stdout、stderr、returncode、异常和单调时间。命令正常返回、超时或取消终止均经过日志写出；启动失败返回码为 null，保留异常，不伪造成功。其他命令不启用此可选日志。
- 一次最小检查：`.venv/bin/python -m pytest -q tests/test_x2_desktop.py -k official_sd_readiness`，**5 passed, 20 deselected in 0.04s**。覆盖第三次就绪、始终不就绪、间隔取消、命令超时仍保留输出、停车预留不足时不启动；只使用假时钟与 mock 子进程。

未启动完整脑、Docker、VM、ROS、仿真或设备；没有改 SDK、模型、动作/姿态阈值、Electron 或预算常量。下一步交新独立验收，再由协调者决定有界实测；此记录不把 mock 检查称为实际 SD 已成功。
