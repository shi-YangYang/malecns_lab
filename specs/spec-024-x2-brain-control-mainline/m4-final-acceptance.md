# M4 限定返工独立验收 — 工程准备 PASS

日期：2026-10-23。范围仅为 [初次验收](m4-acceptance.md) 的 R1–R4、[限定返工](m4-repair.md) 和 [接缝约定](m4-integration-contract.md)。本验收者没有参与返工，未改业务代码。按 S2 核对三个相关 Python 模块及直接测试，没有启动完整脑、MPS、VM、ROS、仿真、训练或硬件，没有 commit/push。

**结论：R1–R4 工程准备 PASS，可继续做真实固定候选的有界仿真资格试验。M4 产品能力与实际 P5 尚未完成；本结论不代表候选已合格、任务成功或真实遥测联调成功。**

## 逐项结果

| 项目 | 结论与证据 |
| --- | --- |
| R1 实际加载入口 | `learning/x2_behavior_session.py:205` 的 `load_candidate_policy()` 使用 `learning/x2_inference.py`，实际包含 `read_anchors`、`load_parameters`、`CHECKPOINT`、`REFERENCE`；历史来源和参数校验仍执行。随后调用原严格 `restore_checkpoint()`，没有删除完整图/输入/源码身份检查。`source_inventory()` 包含该真实加载模块；已有 CPU fixture 核对参数和源码哈希。 |
| R2 请求边沿消费 | `NeuralSession.last_consumed_request_id` 只在实际策略调用返回后推进；观察请求、远端响应及主机验收共同使用该值，真正推理前再次据此生成 `request_changed`。gate-only refresh 和执行轮询均不消费边沿；旧请求推理推进的神经状态仍保留，新任务使旧输出失效。既有同任务新请求跨多次 refresh 检查覆盖这条路径。 |
| R3 拒绝与执行事实 | `rejected_execution()` 归属于被拒绝的 decision/request；主机会话经 gate 校验后排队，host gate 拒绝单列来源。下一次实际策略调用投影并消费一个拒绝，单纯观察或刷新不消耗。投影中 status/behavior/request/execution 一致取自被拒绝决策，原始 decision 归属另存；yaw、vx、wz、standing、MC mode 等身体量来自当前观测。原当前反馈和 active execution 另行记录，gate/executor 活动状态没有被投影改写。 |
| R3 HOLD 收束 | worker 只有在 HOLD/FINISH、活动槽为空，且该决策对应实际 COMPLETED/CANCELLED、无 protection reason、`stop_confirmed=True` 时，才进入该请求的待命等待。REJECTED 不满足此条件，会继续给脑反馈；决策次数预算结束单列原因，`task_success=None`，没有把拒绝或保护计为模型成功。 |
| R4 候选构造与恢复 | 注册只传 checkpoint ID、初始化与存在时的 backward 身份；worker 以当前 `backward_identity()` 校验完整 backward 内容并传入实际 gain，显式 null、未知 gain 或被改写身份拒绝。缺字段仅按历史 gain=1 构造；原 `restore_checkpoint()` 仍要求完整 policy 等式，因此当前源码不能静默恢复旧源码身份。参数数组校验及恢复校验保持原样。 |
| R4 注册包预算 | 原 checkpoint JSON 为 1,396,571 字节，仍由 checkpoint 检查/恢复专用入口读取。worker 注册不复制完整 policy；以实际初始化、48 项当前源码哈希、运行字段及额外 2 KiB backward 预留构造的纯内存包为 7,599 字节，低于 `MAX_PACKET_BYTES=262144`。 |

## 验证范围

- 沿用实施者已经完成的 **18 passed in 1.02s**，以及显式 null 校验补改后仅重跑的 **1 passed, 17 deselected in 0.51s**；本验收阅读了直接测试及实现，没有重复运行这些检查或 Electron/build。
- 本次新增执行只有一条针对注册包大小的 stdlib CPU 计算：读取既有 checkpoint JSON、收集当前源码哈希并在内存构造注册字段，不写运行目录、不读取模型数组、不导入 Torch。输出 `fits=true`、`torch_imported=false`。
- 未发现本次 R1–R4 范围内仍需返工的问题。测试 fixture 证明接缝处理，不替代真实候选加载、数值推理、MC 行为或遥测证据。

## 后续资格试验边界

`heading_error_rad=0` 且没有外部方向目标、wave 禁用、当前尚无合格新候选，均仍是明确产品缺口。它们不阻止工程入口使用**已登记且严格身份匹配的固定候选**开展有限诊断；没有可恢复候选时应继续明确失败，不启动仿真碰运气，也不回退旧三轴模型或任务脚本。

方向任务不能据此宣称已经具备，后续需实际目标观测和对应模型证据。wave 保持禁用，直到独立技能与取消资格通过。P5 的 25 段真实互动、故障处理、同次真实神经遥测及所需同步记录仍须实际运行并独立计分；本报告只解除 R1–R4 工程接缝阻断，不改变 P2–P5 门槛。
