# M1/M2 限定返工独立验收

日期：2026-10-23。审查者为新的独立验收者，未参与实施、前次验收或返工，未改业务源码。

结论分别为：**P1 工程 PASS；M2 首次真实 GPU 单技能探针工程准入 PASS。** 这不代表 P2 实际技能通过。当前尚无本阶段真实转向、挥手、取消或停稳运行证据；挥手仍只允许隔离仿真 `--wave-probe`，不能开放为已验证的用户能力。

本次范围限定为 [M1 验收](m1-acceptance.md)、[M2 验收](m2-acceptance.md)冻结项及 [返工报告](m1-m2-repair.md)的对应修改。风险 S3，静态复核涉及的契约、执行器、主机清理和直接回归测试；没有扩展为全项目重审。

## 冻结问题复核

| 项目 | 复核结果与代码证据 |
|---|---|
| 空闲 HOLD/FINISH 生命周期 | `control/x2_behavior.py:278` 在接受空闲合法决策时保留执行槽；`control/x2_skills.py:287` 发出的本地 ACCEPTED、实际站稳 STARTED、COMPLETED 均能经门禁消费。HOLD 接受本身不清空执行槽，也不代替站稳反馈。`test_idle_hold_actual_executor_lifecycle_passes_gate` 使用执行器实际事件接入门禁。 |
| 本地 admission 与官方 ACK | `ExecutionEvent.fact` 区分 `skill_admitted` 与 `preset_ack`；门禁仅允许 WAVE_RIGHT 在 ACCEPTED 状态收到一次有正 task_id 的官方 ACK，重复拒绝前不更新事件序号。执行器 `preset_ack()` 要求当前仍为 ACCEPTED 且尚无 task_id，保护后迟到 ACK 不能覆盖 UNKNOWN。两者均未把 ACK 提升为 STARTED/COMPLETED。 |
| 普通结束归属 | `finish_command` 单独保留结束决策和 request；后续事件沿用原 execution、原 decision 和原 request，并携带 `finish_decision_seq/finish_request_id`。门禁核验结束决策已被接受、属于同一原始执行且为 FINISH。`finish_requested` 保持当时状态，不伪造运动开始；挥手普通结束等待有限自然完成。 |
| UNKNOWN/FAILED 与未确认停止 | 门禁 `blocked_reason` 与 active 归属保留；执行器 `submit()` 明确拒绝 UNKNOWN/FAILED，独立于 `motion.failures` 是否为空。`stop_timeout` 的空 failures 分支也不会恢复准入。连续动作遭外部预设保护后，即使底盘 STOPPED，也只写 `base_stop_confirmed=true`，整体 `stop_confirmed=false`，不宣称手臂已停止。 |
| ADVANCE 方向证据 | `update_odom()` 按起始 yaw 记录前向、侧向有符号投影；成功要求前向超过 0.02 m 且大于侧向绝对值，保留起始位姿、当前位姿、距离和判据。反向、纯侧向不能记为完成；非零起始朝向回归直接覆盖坐标变换。 |
| 连续技能的外部播放器互斥 | `update_mc()` 及每次 `tick()` 均检查播放器；非 IDLE、未知或过期即请求零速并锁存 UNKNOWN。随后发布周期不再发非零连续目标，PREPLAYING、PLAYING、INTERRUPTING、ERROR 和未知值有直接回归。没有借零速推断上肢取消。 |
| 记录故障与资源清理 | `SkillRuntime.cleanup()` 不依赖状态日志成功。容器 stop、容器 inspect、VM stop、VM status、host、probe 与日志关闭分别捕获故障；某阶段失败后仍尝试后续阶段，停止命令失败后仍执行状态确认。只有已拥有资源和子进程均获关闭确认且没有外部资源时，才置 `resources_stopped=true`。生命周期/最终摘要输出失败会记录错误并返回失败，不伪称封存成功。 |
| 资源归属 | 清理仅操作 `owns_container/owns_vm` 对应资源及本次子进程；外部资源保留且不会标成全部关闭。没有改动旧桌面 cleanup、默认 Docker context、SDK/MC/Sim 或其他服务。 |

## 验证依据与停止

- 已阅读实施者原 M1 的 16 项、M2 的 13 项通过记录，以及返工的 41 项相关检查和随后独立新增的 1 项 `stop_timeout` 回归通过记录。后者修改了执行器拒绝条件，因此只补该具体分支，未冒称最终版本重新全跑了 42 项。
- 本次核对了上述实现与 `tests/test_x2_behavior.py`、`tests/test_x2_skills.py` 中的直接回归；没有发现必须另行复现的具体新疑点，故没有重复执行测试，也没有添加无关测试。
- 主机故障回归使用纯 mock，覆盖日志错误、容器/VM/host/probe 清理错误、最终摘要写失败，以及外部资源不可触碰；这些证据只证明工程故障路径，不等于真实资源已启动或已关闭。
- 本审查未启动 VM、ROS、仿真、完整脑、训练或设备；未 commit/push。冻结问题已满足，停止进一步验证。

## 准入边界

协调者可按已登记的版本、输出新目录、串行资源约束与有界预算开展首次真实 GPU 单技能探针。任一次真实失败须保留，不能据本报告自动重试、放宽方向/停稳阈值或把 UNKNOWN 改成成功。

M1 的生产完整图映射与 MPS 刺激仍由 M3 核验；本报告的 P1 是工程契约与编码/数据边界验收。M4 仍需把独立 probe 的事件补齐为完整 run/session/backend/skill_pack 协议，并分别处理 macOS 与 Linux 的接收时钟；这不是本次显式单技能探针准入的阻断，也不能声称跨主机脑控已接通。

**P2 实测、挥手紧急取消、M3 模型能力、M4/P5 桌面联动及 M6 实机均不在本报告通过范围。** 普通 END_INTERACTION 与独立紧急保护继续分开，既不跳过 wave 的取消缺口，也不把单技能探针升格为整条主线完成。
