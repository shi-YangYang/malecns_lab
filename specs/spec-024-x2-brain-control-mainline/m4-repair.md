# M4 限定返工 — R1–R4

日期：2026-10-23。状态：实施者完成限定修复，等待新的独立验收；不自行把 M4/P5 改为通过。本轮按 **S2** 处理三个相关 Python 模块的加载、跨进程观测和会话反馈接缝，仅运行直接 CPU 检查。没有启动完整脑、MPS 计算、VM、ROS、仿真、训练或硬件，没有 commit/push。

## 修复结果

- **R1**：worker 通过 `load_candidate_policy()` 使用真实的 `learning/x2_inference.py` 加载历史 anchors、参数和完整参考图路径。保留 `read_anchors` / `load_parameters` 原校验及 `restore_checkpoint` 严格校验。该加载模块已经在 `source_inventory()` 的实际清单中，定向检查核对了源码哈希。
- **R2**：唯一 `NeuralSession` 维护最后实际完成策略调用的请求 ID。观测查询携带该事实，远端据此产生请求边沿，主机复核；真正入脑前也以会话事实生成边沿。gate-only refresh、执行轮询及新任务失效检查均不消费边沿。同任务的新请求经过多次刷新后，首次实际入脑仍收到 `request_changed=True`，下一次为 False。旧请求造成的神经状态推进继续保留。
- **R3**：实际执行器 `REJECTED` 通过完整事件到达主机会话并排队；host gate 拒绝另记来源并排队。只有下一次实际策略调用才消费一次。固定 M1 反馈槽中的 status、behavior、request 和 execution 全部取自被拒绝决策；decision 归属留在同条证据，身体量来自该次新鲜真实观测。完整当前反馈与活动执行 ID 独立归档，gate / executor 的 active 状态不因投影而改写。新任务和 refresh 不清空/消费队列。修复了 HOLD 被执行层拒绝后仍直接结束请求的问题：只有实际正常终态及 `stop_confirmed=True` 才转入待命等待；拒绝和保护不计为模型正确，`task_success=None` 继续保留。
- **R4**：主机注册透传候选的初始化与完整 backward 构造身份；worker 在读取模型资产前用当前 `backward_identity()` 严格核验，按候选的 1.0 / 0.1 构造。缺少旧 backward 字段只表示历史 gain=1.0；显式 null、未知 gain 或身份内容不符均不静默回退。`restore_checkpoint` 的完整 policy/input/source 等式未修改。旧源码 checkpoint 若与当前身份不符仍明确拒绝，不能把“默认 gain=1 数值路径兼容”说成“当前源码可直接恢复旧 checkpoint”。

注册只包含必要构造字段，未把完整 policy 塞入 256 KiB 传输包；原完整 checkpoint 约 1.4 MB，仍由严格恢复函数读取、验证。没有修改 M1 输入身份、M3 策略/训练/shared brain、SkillRuntime 或 Electron。

## 实际验证

1. `PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_x2_behavior_desktop.py`：**18 passed in 1.02s**。包含原直接边界检查及新增的真实 `SkillExecutor` 前置拒绝复现、同任务新请求跨刷新边沿、拒绝一次消费、旧 active 与实际身体量保留、真实 loader 路径、gain 1/0.1 传递、严格恢复不放宽及加载源码快照。
2. 随后补充显式 `backward: null` 的拒绝检查，只重跑直接相关项：`... -k candidate_backward_tampering`：**1 passed, 17 deselected in 0.51s**。没有重复整个文件或全仓测试。
3. 对四个变更源码/测试文件做 `git diff --no-index --check /dev/null <file>`：均无空白错误输出；文件尚未跟踪，no-index 因存在内容差异返回 1。未运行 build、lint 或 typecheck。

测试使用明确标注的 CPU 协议 fixture。加载入口检查替换构图函数并记录参数，绝未创建图或 MPS 张量；它证明接缝调用正确，不证明真实候选恢复/推理成功。

## 保留限制

`heading_error_rad=0` 与无外部方向目标的限制保持原样，没有根据任务名构造目标或成功事实。拒绝投影占用既有单个反馈槽一帧，完整并行活动反馈作为独立证据保存，随后正常观测继续携带实际活动状态；未改变编码维度或伪造训练身份。真实候选联调、转向/挥手能力、取消资格、P4 干预和 P5 任务结果仍需后续各自实际证据。
