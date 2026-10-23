# M1/M2 限定返工报告

日期：2026-10-23。风险 S3（执行契约、互斥与资源关闭）。新返工实施者仅处理 M1 验收的生命周期接缝和 UNKNOWN 锁存，以及 M2 已冻结的 R1–R3；未修改旧 `control/x2.py`、数值内核、SDK、感觉编码或数据集。未启动 VM、ROS、仿真、训练、完整脑或硬件，未 commit/push。等待另一位新的独立验收者，不自行宣称 P1/P2 已通过。

## 实际修复

- `control/x2_behavior.py`：空闲 HOLD/FINISH 与 START 一样保留执行槽，实际 ACCEPTED→STARTED→COMPLETED 可以完整消费。`ExecutionEvent` 增加带默认值的 `fact`、`finish_decision_seq`、`finish_request_id`，原 Observation/Decision 与输入编码 API 不变。`skill_admitted` 表示本地接受；`preset_ack` 需实际 task_id，允许且仅允许一次 ACCEPTED 同状态事实；没有将 ACK 提升成开始或完成。`finish_requested` 只记录当前阶段，不假称 RUNNING。结束已有行为仍沿原执行、决策和请求身份，新结束决策单独关联并由门禁核验。UNKNOWN/FAILED 或保护事件保留活动归属并锁存门禁，不自动恢复。
- `control/x2_skills.py`：执行日志保留上述事实与结束决策关系；实际服务 ACK 晚于保护锁存时不能改回 ACCEPTED。ADVANCE 将实际位移投影到起始朝向，成功需前向投影大于 0.02 m，且大于侧向投影绝对值；反向、纯侧向或侧向占优不能计为前进。平面距离、前向/侧向投影、起始位姿、当前位置及判据均保留。此判据是本地技能验证条件，不是现场安全指标。
- 连续行为每次 common state 更新及发布周期检查预设播放器。出现非 IDLE、未知或过期的播放器状态时归零并锁存 UNKNOWN。底盘停稳只记 `base_stop_confirmed`，不会由零速度推断外部手臂动作已取消；整体 `stop_confirmed` 保持 false。原子序列拒绝、互斥及有限执行预算保留。执行器明确拒绝 UNKNOWN/FAILED 后的新命令，包括 `stop_timeout` 未产生其他 `motion.failures` 的路径。
- `experiments/x2_skill_host.py`：增加本入口专属清理，不修改旧桌面入口。生命周期日志/stdout 失败记录错误并触发退出，不能阻断清理。容器停止、容器状态确认、VM 停止、VM 状态确认、主进程与 probe 进程退出、日志关闭各自独立执行并保留错误。关闭命令异常后仍尝试状态确认；`resources_stopped` 只有本次已拥有容器、VM 和子进程都获确认且不存在外部资源时才为 true。未拥有资源不被操作，停止失败也不妨碍后续清理阶段。最终摘要写失败返回失败结果，不能伪称证据已封存。

## 真实检查

首先执行一个受影响子系统命令：

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_x2_behavior.py tests/test_x2_skills.py
41 passed in 0.15s
```

新增回归使用实际执行器事件对象接入门禁，覆盖空闲 HOLD、两种 ACCEPTED 事实、重复 ACK 拒绝、普通结束前尚未开始动作、原执行与新结束请求归属、保护锁存、反向/侧向前进误判、非零初始朝向的前进投影，以及 PREPLAYING/PLAYING/INTERRUPTING/ERROR/未知播放器状态。主机检查仅使用本地 mock，分别注入日志、容器、VM、主进程、probe 与最终摘要输出故障，并检查没有接管外部资源。

差异审查发现同一 UNKNOWN 锁存项还有明确分支：停止超时可进入 UNKNOWN，但 `motion.failures` 为空。为此增加执行器显式终态锁存及一个直接回归，只运行新增测试，没有重复上述 41 项：

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_x2_skills.py::test_stop_timeout_without_other_guard_failure_still_latches_executor
1 passed in 0.04s
```

新增文件以 `git diff --no-index --check /dev/null <文件>` 检视，未输出空白错误；其退出 1 表示新增文件与空文件有差异。最后的锁存条件与直接回归已按补丁检查。没有全仓测试、build、lint、typecheck，未重跑旧项目或科研结果。当前合计 42 个直接相关测试有通过记录，未声称最终版本整体重新执行过 42 项。

## 交接与限制

控制源码身份已变化，已通知 M3 返工者与 M4 实施者；后续 run 必须记录完成后的实际源码哈希。M4 封装 ExecutionEvent 时保留 `fact`、结束决策字段和原始反馈中的 task_id，使用各自接收时钟，不将 macOS 与 Linux monotonic 直接相减。

正式 GPU 技能实测、挥手紧急取消语义、跨进程脑控运行和实机阶段仍待对应证据。本轮无新增实际运动结果，不解除挥手的 probe 限制，也不将普通结束与独立紧急停止混为一谈。保护锁存没有自动清除入口；需要上层结束当前会话并按已有前置条件重新建立，不能把“未确认停止”视为可继续动作。
