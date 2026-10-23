# M4 独立验收 — 工程准备 FAIL

日期：2026-10-23。范围：M4 接缝约定、实施报告、新增四个 Python 模块、Electron 差异及直接检查；只读审查业务代码，没有启动完整脑、MPS 计算、VM、ROS、仿真、训练或设备，没有提交。检查风险按本阶段 S3，但仅针对具体接缝疑点做最小 CPU 复现。

**结论：工程准备 FAIL，暂不准入真实候选联调。** 有一个确定启动阻断及两个会丢失真实输入信息的接缝问题。候选重建还需要对齐正在实施的 M3 新 backward 身份。P3/P5 仍未通过；本报告不评价未经运行的模型和行为成功率。

## 必须返工

### R1 / P1：worker 引用错误的加载模块，任何候选都会在构图前失败

`learning/x2_behavior_session.py:173` 使用 `from . import x2_frozen as frozen`，但 200–205 行调用其 `read_anchors`、`load_parameters`、`CHECKPOINT` 和 `REFERENCE`。这些成员实际位于 `learning/x2_inference.py`，`x2_frozen.py` 是 `FrozenInference` 包装实现，没有上述成员。

最小复现：用 `.venv-train/bin/python` 仅导入 `malecns_lab.learning.x2_frozen` 并查询这四个属性，全部为 `False`；调用 `x2_frozen.read_anchors(None)` 得到 `AttributeError: module 'malecns_lab.learning.x2_frozen' has no attribute 'read_anchors'`。未创建图或 MPS 张量。

返工应使用真实加载入口，保留旧来源/完整图校验，并核对新增入口包含的源码快照。不能通过删除加载校验或使用旧三轴策略绕过。

### R2 / P2：推理期间的新请求被 refresh 消耗，首次实际入脑缺少 request.changed

`experiments/x2_behavior_bridge.py:35–42` 用最近一次**观测查询**更新 `last_request`，从而生成 `request_changed`。`learning/x2_behavior_session.py:279–287` 在旧推理结束后为新请求取一次 refresh 观测，该观测只交给 gate；下一循环 270 行再次取样才真正交给策略，此时 `request_changed` 已为 `False`。

纯 CPU 的 `ClockExchange` / `RemoteExchange` 三次交换复现：

```text
actual old inference request_changed= True
refresh gate only request_changed= True
actual new inference request_changed= False
```

`senses/x2_behavior.py:30` 编码该边沿，而其身份明确排除 `request_id`。因此重复同一任务但使用新请求 ID 时，真实脑没有收到本应出现的新请求刺激；旧结果正确拒绝和状态持续不能弥补这一信息丢失。

返工应按**最后一次实际被脑消费的请求**生成边沿，或保留待消费的变化事实；可以继续进行 gate-only 刷新和失效检查，不得因此回滚已推进脑状态或让旧输出继续执行。

### R3 / P2：执行器前置条件拒绝只进事件日志，没有进入下一次脑反馈

`experiments/x2_behavior_bridge.py:252–263` 对 `core.submit_decision=False` 手工创建 `REJECTED ExecutionEvent`。但 `feedback()`（190–200 行）仅读取 `core.status`、`core.command`，没有对应的拒绝事实。正常提交前置条件失败不会改变这些字段。

最小 CPU 复现：显式 fixture 将 `SkillExecutor.standing_ready()` 设为 `False`，提交合法 ADVANCE/START，结果为 `submit=False, core.status=IDLE, core.command=None, core.events=[]`。外层手工事件会关闭 gate，但下一次 Observation 仍显示 IDLE（或上一技能的旧状态），脑不知道自己的本次动作已被执行层拒绝。

返工应把带真实 request/decision 归属的执行层拒绝作为后续脑输入的一部分，且不能抹掉仍在执行的技能/实际身体反馈；拒绝和独立保护应继续单列，不能替换成模型选择 HOLD 或任务成功。

### R4 / P2：候选重建没有传播新的 backward 配置

审查期间 M3 已将 `spike_backward_gain` 纳入 `X2BehaviorPolicy` 身份，`from_reference` 默认值为 1.0。M4 在 `experiments/x2_behavior_desktop.py:153–155` 只保留候选的 `policy.initialization`，`learning/x2_behavior_session.py:203–207` 也只传初始化和 head seed。对于登记 gain=0.1 的 stabilized 候选，重建身份与候选不一致，`restore_checkpoint` 的严格身份比较会拒绝。

这是当前接口检查所得，没有运行或声称已有合格 stabilized checkpoint。返工应传播候选真实登记的完整必要构造参数，保留严格身份检查；不能用默认值静默重建、放松比较或伪改 checkpoint 来源。

## 已核对的正确边界

- 五任务 renderer IPC 有枚举白名单，不接收 checkpoint 路径、MC 命令或 shell。普通 END 走任务输入，紧急停止单独控制资源生命周期。
- `NeuralSession` 只有一个策略实例；推理读取真实 DN 特征，未见 teacher 或任务到技能的运行时旁路。神经状态在决策后保留，旧请求原始 logits 与特征先归档后拒绝。
- macOS/Linux 单调时钟分别使用；nonce、请求/序号、原始来源年龄和 RTT 被校验，读旧文件不续期。VM 用原 nonce 签发时刻限制决策年龄，MC 技能另有有限时长。
- `wrap_execution` 保留原始 decision/request、finish decision/request、fact、实际请求和传感器反馈；不把 preset ACK 当完成。传输期间执行归属改变会拒绝并结束运行，未见静默替换技能或补成功；真实保护效果尚待修复后的故障实测。
- wave 的能力清单和服务调用均禁用，未发现把“有官方动作资源”宣称为可用技能的路径。
- 默认导入/preview 不启动脑、VM 或 ROS。运行入口使用单完整脑锁、有限预算、资源监督和新目录；`SkillRuntime.cleanup()` 对容器、VM、子进程、日志分别尝试清理。M4 紧急文件写入失败不会直接跳过该清理函数。
- 新遥测确实从该策略 `_step` 的实际 spikes 采样，manifest 改为新策略来源并绑定输入/候选身份；50 步批次、22 群组、32 选样和丢批信息保留。旧 schema 的读取路径保留。未观察到第二个脑或装饰活动。
- 没有候选配置时明确失败，无旧策略/合成图回退。当前只有 qualification 模式，`task_success=null`、`behavior_pass=null` 保留了能力边界。

## 检查与限制

采用实施报告及直接测试代码中的既有证据：Python 11 项、Electron 12 项、最终桌面构建和 diff 检查；本验收没有机械重复它们。新增仅为 R1 属性检查和 R2/R3 的 CPU 契约复现。R2 首次复现命令多写了一个不存在且未使用的 helper 导入，移除该导入后重试得出上述结果；没有修改实现或测试文件。

`heading_error_rad=0` 且无外部方向目标是已披露的能力缺口，本报告不把它算作已具备左右致意能力。后续需要真实、可登记的任务目标观测及相应模型分布；不得根据任务名伪造完成事实、注入教师答案或直接选择 MC 动作。wave 取消资格、可靠技能、合格候选、P4 配对干预、P5 25 段真实互动及同步视频仍需各自证据。

按项目流程把 R1–R4 限定交给新的实施者返工，再由新的独立验收者审查。返工只运行相关边界检查；未修复前不启动真实脑/仿真来撞这个已确定的启动错误，也不因构建通过把 M4/P5 标成完成。
