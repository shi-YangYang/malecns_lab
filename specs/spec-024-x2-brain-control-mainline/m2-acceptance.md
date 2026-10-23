# M2 独立工程准入验收

日期：2026-10-23。结论：**首次真实 GPU 技能探针的工程入口准入 FAIL，等待限定返工；P2 官方技能实测仍未通过。** 本审查者未参与实施，未修改业务代码，没有启动 VM、ROS、仿真、完整脑或设备，没有 commit/push。

风险 S3：执行安全、来源归属、生命周期及停止证据。实施报告中的 13 项定向检查已有通过记录，本次只阅读测试，不机械重跑。针对三个具体疑点执行了两条纯假反馈/mock 最小复现命令，均无需实际设备；问题清单至此冻结。

## 阻断项

### R1：ADVANCE 可以把向后移动误判为完成

`control/x2_skills.py:155-160` 以起点到当前位置的无符号平面距离确认前进；`tick()` 的 358 行对 ADVANCE 无条件令方向判据通过。因此反方向甚至横向移动，只要距离超过 0.02 m 并停稳，都可写成 ADVANCE COMPLETED。正式技能准入需要实际方向和运动证据，不能只验证机器人移动过。

纯假反馈复现：提交 ADVANCE，连续 52 个 20 ms 样本沿初始前方的反方向移动，每样本 -0.004 m，线速度 -0.2 m/s；随后正常停稳 65 个样本。结果：

```text
backwards_advance COMPLETED 0.20800000000000016 []
```

限定返工：根据初始机器人朝向计算有符号前向位移，区分前进、反向和侧向偏离；原始位姿与方向判据落盘。反向/侧向运动不能计为前进成功。不得通过修改教师标签或放宽成功含义修复。

### R2：连续移动期间未处理外部预设动作

`standing_ready()` 在准入时检查 player IDLE，但 `update_mc()` 的 170-178 行只有当前技能是 WAVE_RIGHT 时才检查播放器；连续技能的 `tick()` 也不检查玩家状态。外部动作进入 PLAYING/INTERRUPTING/ERROR 后，连续速度仍会持续发布，与 Spec024 首版串行动作及外部并发动作未知/失败要求冲突。

纯假反馈复现：提交 TURN_LEFT 后，将 common state 的 player_state 改为 PLAYING，保留其余新鲜、直立和控制来源条件。下一 tick 结果：

```text
external_animation_during_turn (0.0, 0.0, 0.2) ACCEPTED []
```

限定返工：连续技能开始后的每个新鲜 common state/发布周期都守住动作互斥；出现未归属上肢动作或异常播放器状态时，立即停止非零连续目标，锁存独立保护原因。零速度不能表示手臂已取消；停止未确认时维持 UNKNOWN/失败和不可重启语义。

### R3：记录输出异常能阻断所有资源清理

新主机复用 `experiments/x2_desktop.py:505-506` 的 `cleanup()`，其首句 `status()` 会调用未保护的 `emit()`；新主机 `main()` 的 emit（`x2_skill_host.py:227-231`）会直接写入并 flush 生命周期文件。当写日志或 stdout 失败时，清理在任何容器/VM 关闭命令前抛出。`execute_skills()` 的 178-182 行仅记录 `cleanup_exception`，没有继续清理本次资源的后备路径。

纯 mock 复现：构造 SkillRuntime，标记 owns_vm/owns_container=true；emit 模拟 OSError，command 被替换为仅计数的 stub。调用 cleanup 的结果：

```text
OSError simulated lifecycle log failure
cleanup_commands 0 owns_vm True owns_container True resources_stopped False
```

限定返工：状态/证据输出失败不得阻止已拥有资源的有界关闭；各清理阶段相互独立，继续尝试停止本次容器、VM及子进程。保持不接管或关闭外部资源，不以无证据状态宣称 resources_stopped。验证只需日志故障与清理故障的本地 mock，不为此启动 VM，也不要求重写旧桌面入口。

## M1 接口依赖与尚未接通的边界

[M1 独立验收](m1-acceptance.md)已正式报告空闲 HOLD 的 ACCEPTED/STARTED 生命周期被门禁拒绝、挥手两次 ACCEPTED 与门禁转移冲突，以及 UNKNOWN 后的恢复语义差异。本报告引用同一问题，不重复交修；协调者应将 M1/M2 接缝交同一新实施者。

`submit_decision()` 当前只是把已通过门禁的决策投影为 SkillCommand；它本身不验证 run/session/policy，也不负责将事件封装成 M1 ExecutionEvent。实际 `_emit()` 的 dict 尚无 run/session/backend/skill_pack_id 等完整协议字段，独立 probe 直接发送显式 SkillCommand。该能力核验路径不是脑控运行链路，不能称 P1 或 M4 集成已经通过。

执行器有效期使用其所在进程的单调接收时钟。macOS 与 Linux VM 的 monotonic 值不能直接互用；后续桥接必须由主机校验观测/决策新鲜度，再给执行器独立的接收时刻与有界执行预算。ROS 原始消息头、接收 monotonic 与 wall time 已记录，但没有因此证明跨主机时钟对齐。此项是集成待完成边界，不阻止修复后进行显式单技能 probe。

## 已核实的静态边界

- 主机限制新输出目录，持有桌面锁和 `data/.continuous-training.lock`。继承 guard 检查至少 1 GiB 系统预留、父进程和停止信号；总预算 240 秒，正常工作提前 30 秒停止，probe 启动需剩余 41 秒。当前是实现预算而非已运行耗时证据。
- 启动前要求私有 Lighter 停止且无 socket、旧 CPU 容器 exited；使用显式私有 Docker socket，不切换默认 context，不停止 Noria。发现外部私有 workloads 时拒绝接管，未把外部资源伪装为已关闭。
- transport 快照列出 control 父包导入所需的 adapters/body/command/legacy，以及 x2/x2_protocol、experiment 父包和 runner。body 的 NumPy 依赖仍存在，preview 延迟导入控制模块；旧包初始化和历史源码未改。当前审查未进入容器实际导入，因此不以静态快照完整性冒充容器运行成功。
- 官方文件 hash 与本地配置版本在启动前落盘，容器按 `/workspace/<相对路径>` 再核对；部署配置将 packages 挂载到 `/workspace`。版本标签明确标为配置，不称现场核验；并未修改 SDK、MC 或 Sim。
- 已核对本地官方 SetMcPresetMotion、GetMcPresetMotionState、McPlayerState、McMotionStatus、McMotionType 和 JointState/JointStateArray：相关请求/响应字段、话题及右臂关节名有实际资产依据。官方 query 400/1 和 common player 状态被分开处理；ACK 不是动作完成。
- 连续目标限值 0.2、默认 1 秒/上限 2 秒，来源认领、反馈过期和角速度/角度漂移停止守卫存在。R1/R2 表明现有检查还不足以放行真实 probe，不能因其他边界正确忽略它们。
- 挥手默认不可给用户执行，仅 `--wave-probe` 开放隔离仿真核验；没有编造取消服务或用 motion=0、替换动作伪装停止。自然完成要求 ACK、query running、common PLAYING、实际关节变化、回位与稳定；保护取消保持 UNKNOWN，`wave_cancellation_verified` 恒为 false。当前没有真实动作、归属或紧急取消记录，不能据此放开用户挥手能力。

## 恢复条件

由新的实施者限定修复 R1-R3 与 M1 已冻结接缝，补最小直接回归证据；随后交新的独立验收者。不要扩大为全仓测试、旧脑数值重验或 SDK 改造。工程准入通过后，协调者才按已登记预算运行真实 GPU 单技能探针；任何一次真实失败保留，不自动重试或改阈值。P2 仍须实际动作、停止/取消与归属证据，不能由本工程审查替代。
