# M2 挥手结束与取消的官方接口边界

日期：2026-10-23。只读核对；仅新增本文，风险 S0，不运行测试、脑计算、Electron、VM、容器或硬件。适用本地 AimDK 1.1.0 / `faf4c94`、MC/Sim v1.0.0-20260522、`lx2501_3_t2d5`。没有修改厂商文件、业务代码或既有运行证据。

## 结论

**本次未找到官方明确支持的“终止本次挥手，同时保持动态平衡并回到待命”的调用契约。不能登记一个已获得官方语义支持的取消探针，也不能开放 M4 的挥手能力。**

`JOINT_DEFAULT` 是公开 API 可请求的模式，并非仅有内部枚举；但它是位置控制准备姿态，不等同于 `STAND_DEFAULT` 的动态平衡控制。通过它绕出动画播放器不满足本项目的站稳取消要求。

`STAND_DEFAULT → LOCOMOTION_DEFAULT → STAND_DEFAULT` 的模式名称与普通转换路径有官方依据，但“正在播放时允许转换、退出时清除原任务、重新进入不续播、手臂有界回位”均没有得到证明。**它最多是待官方确认的候选假设，本次不建议发送试探命令。** 文档还明确列出 `SetMcAction` 的 `MotionBusy=9` 拒绝原因；不能只看静态转换表就认定播放中的切换会执行。

## 已支持、假设与不可用

| 项目 | 证据与边界 | 当前处理 |
| --- | --- | --- |
| 自然完成挥手 | `SetMcPresetMotion` 的 1002/area2 与查询接口已公开；本项目还要求实际关节运动、播放器结束、回位与站稳。它不是提前取消。 | 保留原单次自然完成探针；此前因转向失败或 UI 准备超时仍未执行。 |
| 请求 `JOINT_DEFAULT` | 官方 Python 示例通过 `command.action_desc` 发送该字符串；同一示例区分 JD 的位置保持与 SD 的自动平衡。 | 已支持的模式 API，不作为保持动态平衡的取消方案。 |
| 请求 `LOCOMOTION_DEFAULT` 再回 SD | 本地转换表允许普通 SD↔LD；LD 使用 `rl_foundationwalk`，没有 SD 的 `animation_player` runner。 | 配置级候选假设。尚不能证明播放期间可切换或原任务已取消，暂不实施。 |
| 重发 `STAND_DEFAULT` | SD 自身仍配置动画播放器；没有公开的“同模式请求重置播放器”约定。 | 不能作为取消。 |
| `JOINT_FREEZE`、`STAND_BODY_CONTROL` 或包内未公开服务 | 枚举或注册名存在不等于当前官方允许外部使用；没有适用本取消目的的公开调用契约。 | 不从枚举/内部名字推导并发送指令。 |
| `interrupt=true` 替换另一动作 | 已知语义是替换播放；它需要另一个实际动作，未给出无动作取消和待命恢复契约。 | 不用任意动作或 motion=0 伪装停止。 |
| 零 BodyVelocity、停客户端、PD/DD、旧软急停枚举 | 都没有证明本次手臂动画取消且保持平衡；软急停枚举已弃用。 | 不作为挥手取消通过依据。 |

## 新核对到的具体事实

### 1. 公开模式调用并不承诺取消预设

当前官方[运动模式切换](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/modeswitch.html)列出 `MotionBusy=9`，并说明模式与状态反馈使用 `action_desc`；旧整数 `action` 自 v0.8.2 起不再使用。本地随包同页也有该说明（HTML 759–858 行）。`SetMcAction` 返回模式请求的 `CommonResponse`，没有欲取消的预设 `task_id` 参数。

这说明存在播放期间拒绝切换的接口结果，**不证明所有目标模式都会被拒绝，也不证明某个模式能绕过该限制**。具体允许目标与拒绝条件属于尚缺的版本契约。安全保护章节允许 JD 的描述也没有给出“取消动画并保持动态平衡”的保证。

官方模式文档及本地示例同时确认 JD 是公开入口：

- [`set_mc_action.py:26`](../../data/x2_sim/packages/aimdk/src/py_examples/py_examples/set_mc_action.py)：请求写入 `command.action_desc`；64–71 行公开 JD/SD/LD 选项。
- [`SetMcAction.srv`](../../data/x2_sim/packages/aimdk/src/aimdk_msgs/interface/robot/mc/action/srv/SetMcAction.srv)：字段是 header、source、command；不是面向预设任务的取消请求。

因此“JD 只是内部 FSM，完全不能通过 API 请求”不成立；“JD 名字带站立，所以能维持动态平衡”也不成立。

### 2. 配置证明 runner 不同，没有证明退出生命周期

本地实际仿真配置：

- [`sim/action_setting.yaml:23`](../../data/x2_sim/packages/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/action_setting.yaml)：JD 为 `idle + planner_lower + planner_upper`；32–47 行 SD 为 `rl_cpgtelecon + transition_retarget_injector + only_head/only_waist + animation_player`；48–52 行 LD 为 `rl_foundationwalk`。
- [`sim/action_state.yaml:1`](../../data/x2_sim/packages/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/action_state.yaml)：JD 归 PREPARE，SD 归 STABLE，LD 归 MOVE；58–62 行的默认稳定动作是 SD。
- [`sim/action_ruler.yaml:40`](../../data/x2_sim/packages/mc/bin/mc_param/src/robot/lx2501_3_t2d5/sim/action_ruler.yaml)：SD 的普通 next_actions 包括 JD/LD，55–58 行 LD 可以回 SD；静态图不表述播放器忙时的运行时守卫。
- [`classic/idle.yaml:4`](../../data/x2_sim/packages/mc/bin/mc_param/src/robot/lx2501_3_t2d5/classic/idle.yaml)：列出固定关节位置及 PD 增益；这不是与 SD 同一平衡策略。不能据此断言 JD 必然摔倒，但不能认证它维持动态平衡。

在本地 MC 包中未找到相应 `.cpp/.cc/.h/.hpp` 实现，动画播放器提供为 `libmc_runner_animation_player.so`；本次没有反汇编、二进制字符串猜接口或逆向。可读 YAML 不说明 runner 退出时是否清空任务、是否保留时间轴、是否在重新进入 SD 时续播、如何更新查询任务状态。

SD/LD 普通切换也不能等同于播放中取消：本项目已有的前进/转向记录涉及步态转换瞬态，不能拿那些未播放挥手的成功或失败代替本问题的证据。

### 3. 现有反馈无法单独证明“owned 任务已取消”

官方[预设动作查询](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/preset_motion.html)仅承诺 RUNNING/已完成两类结果；明确不能仅靠该服务区分打断或失败。查询请求与响应包含 task_id，但公共播放器状态没有 task_id。

本地 [`McPlayerState.msg`](../../data/x2_sim/packages/aimdk/src/aimdk_msgs/interface/robot/mc/motion/msg/McPlayerState.msg) 的 `INTERRUPTING=3` 定义为过渡到下一动作，并非“已取消且回到待命”。[`CommonTaskRequest.msg`](../../data/x2_sim/packages/aimdk/src/aimdk_msgs/interface/robot/common/msg/CommonTaskRequest.msg)及[`CommonTaskResponse.msg`](../../data/x2_sim/packages/aimdk/src/aimdk_msgs/interface/robot/common/msg/CommonTaskResponse.msg)只能关联任务查询；没有给出离开 SD 后任务状态的具体语义。

项目 [`x2_skills.py:335`](../../src/malecns_lab/control/x2_skills.py)目前如实将紧急取消挥手标为 UNKNOWN，普通 FINISH 等待自然结束；362–387 行自然完成要求实际运动、查询结束、播放器空闲、手臂回位和持续站稳。本次没有改变它。IDLE、服务 ACK、关节某一帧静止或 VM 退出，任一单项都不足以认证取消。

## 上海需向官方确认的精确问项

1. 针对现场机型、固件、MC/AimDK 版本，是否有公开且受支持的“取消当前指定预设 task_id，保持下肢动态平衡并将手臂回到待命”的接口？需要接口名称、消息定义、示例、拒绝码与适用姿态，不只要口头建议。
2. 若官方方案是 `SetMcAction`：1002/area2 正在 PLAYING 时允许哪个明确 `action_desc`？何时返回 `MotionBusy`？调用是否取消、暂停或仅排队？再次进入 SD 会不会恢复原动作？取消是否影响其他调用者的动作？
3. 原 `task_id` 在上述过程的 `GetMcPresetMotionState` 返回值是什么？能否区分自然完成、被本请求取消、被外部动作抢占、失败？公共状态与任务 ID 如何可靠关联，终态保留多久？
4. 取消时手臂是保持当前位置、受控回位还是直接交给另一控制器？回位目标、速度/加速度边界、最长结束时间以及此时下肢使用的平衡控制器分别是什么？是否有受支持的“取消完成且稳定站立”判据？
5. 客户端断开、取消请求超时或任务归属被外部抢占时，官方现场操作者应使用哪条受支持的接管流程？本地 Sim 与现场机器人在该流程上是否一致，哪些反馈或保护行为不同？

获得明确支持后，再登记**一个**匹配版本、固定时刻和固定预算的取消仿真核验；它至少要保留原 ACK/task_id、PLAYING+实际关节运动、取消请求/响应、原任务终态、恢复后的连续新鲜关节/姿态/里程计与模式反馈，并检查回到 SD 后没有续播。若官方说明该版本不提供保持平衡的预设取消能力，记录能力缺口，不自创动作、不修改成功标准。

## 检查与证据限制

只读检查了本地厂商接口、示例、配置、随包 HTML、项目执行器和 M2 实测报告；在线模式页直接读取两次超时，之后用官方站点检索返回完整当前页面，与随包内容一致；在线预设动作页读取成功。结论是接口支持边界，不是新仿真实测，也不是对闭源 MC 内部行为的证明。

本文 S0，仅检查新增文档 diff；未运行测试/build/lint/typecheck，未触碰训练源码或运行中的训练进程。保持 wave_cancellation_verified=false。
