# Acceptance — 固定策略双向验证的实际结果

2026-10-21。独立验收者未参与本轮模型、训练器或评估器实现，没有启动生产训练或评估；先前承担的工作是只读 validation-00 局部诊断。本次仅增量审阅已封存验证结果，不读取锁定测试、不执行模型、不拟合参数、不修改生产源码、协议或运行资产。

## Result

**评估执行完整：PASS。控制 AC3：FAIL。**

两个验证场景 × 五种模式的 10 个作业全部正常执行、封存并汇总；固定 joint 策略在两个方向都违反控制门槛。验证失败已足以否定本轮模型的接入就绪条件。这里的 AC3 FAIL 表示前置验证门槛失败、AC3 未达成，**不是声称锁定测试已执行后失败**。实际 `test_lock.json` 和 `evaluation/test` 均不存在；锁定测试应保持关闭。

执行 PASS 只证明这次对照按登记模型和场景完整运行，不能用来掩盖性能失败。AC2 训练执行通过的结论保留；AC4 模型实际部署恢复和 AC5 实际网页验收未由本报告证明，整个 Spec013 及机器人仿真接入准备仍未完成。

## 范围与原始证据

仓库根目录：`/Users/yang/project/malecns_lab`。

- 主运行：`data/yaw_joint_20261021_r1`；run ID `85dc26fd3a1961ad14405b7eff5d97d965c92f54f8e89c364a2407829d40b30b`。
- 冻结对照：`data/yaw_frozen_20261021_r1`；run ID `2cd5b4b550a0e3f39fe2dfaaac1808658227bf164e33afe7fb2926fff2a60d4b`。
- family ID：`90c0939d87c3b4c01193e96abe00f38e71102d81bb6ae74eb4a7ac5a2d672a67`。
- 固定 joint checkpoint80：`024fe62f2145ed94d8bb6313107c1c6f305bfc5a15a634a4c7f533e623a5a97e`。
- 固定 frozen checkpoint80：`6c1f116b3cc4bff4954545943c26010afc7ac348974f988005a71e25f47e1a79`。
- 汇总：`data/yaw_joint_20261021_r1/evaluation/validation/summary.json`。
- 十项作业：上述 `evaluation/validation/{zero,untrained,frozen,joint,visual_p}/{validation-00,validation-01}/` 中的 `registration.json`、`evidence.json`、`episode.npz`、`result.json`、`worker_result.json`、`supervisor_result.json`、`phase.json`、`resources.csv`、`worker.log`；六项神经作业另有 `mps_memory.csv`。

本次复用 validation-00 已完成的轨迹、分段指标及读出诊断，不重复其计算。新增核对包括十项注册/封存/终态/资源及汇总身份、validation-01 五项轨迹的独立指标复算、两场景总 gate 代数复算，以及正负场景在扰动前的重复输入响应一致性。完整图、39 份登记源码、80 窗训练链和正常恢复沿用 gate、恢复、训练执行 验收，不重新计算这些文件 SHA 或训练数学。

负向场景的新增 evidence ID：

| 模式 | evidence_id |
|---|---|
| zero | `1a8f9c7454f85d5a8486d071da460609865d6d006b313eb185ffedcf265071f1` |
| untrained | `b759b819ae1329e4eea5724bf15c37600acd426e02d971a4bf8ece99ce454b4c` |
| frozen | `a57640750369cc8d3812b3bdad3064964584bd822e84cdaf186270a67519d19f` |
| joint | `7b7298c0177c491738c85cba549b334511228d1e4c60a8e93b18763c05b1f7a1` |
| visual_p | `1e4df0cf4d63e9f4c4aa77fa6f08e8b7aa2b005df1ea22c5f2b3cfb7f3f068ef` |

## 固定选择、封存与执行完整性

十项 `registration.json.job_id` 与 `evidence.json.evidence_id` 的 canonical JSON seal 独立复算一致，十份 `episode.npz` 流式 SHA 与各自 evidence 一致。每项注册、worker、最终 result 的 job/run/evidence 身份吻合，`summary.rows` 恰好包含这十项原始 evidence，没有缺项、重复模式或另一参数选择混入。

所有注册 `kind=evaluate`、`windows=0`，`selection` 相同；joint/frozen 路径均固定在各自 `checkpoints/000080`。两个场景均为 seed201、初始 yaw `.15 rad`、2秒观察、`.40 s` 起始、`.50 s` 外扰脉冲，幅度分别 `+.275` 和 `−.275 rad/s`。每项最终 phase 均到 frame80，worker/supervisor/result 均为 completed，supervisor returncode 为0，无 Traceback 或失败后静默重试证据。

每个神经作业记录40,000个前向神经时间步，采用166,700节点、25,582,938边的完整参考图身份和 `mps_float32` 视觉后端；六项神经作业合计240,000前向步。zero 与 visual_p 不运行脑、不具有伪造的神经数组或 MPS 采样。十项各80帧包括40帧预热和40帧观察，合计800次环境积分；它们是固定策略评估帧，不是800次训练更新。

validation-01 新增原始数组检查：

- 帧索引连续 `0…79`，图像 `(80,72,96,3)` BGR uint8，时间为 `i×.05 s`；浮点字段有限。
- 前40帧命令全零；后40帧命令与候选动作逐值相等，没有把教师直接施加到环境；动作绝对值不超过 `.35 rad/s`。
- `actual_wz=(yaw_after−yaw_before)/.05` 与保存值相符；教师由前一已完成区间实际转速限幅产生，误差容限 `1e-14`。
- zero 所有命令为零；zero/visual_p 没有 `features_mv` 等脑活动字段。
- joint/frozen 的最终 checkpoint 元数据身份、optimizer step80、完成窗80与本次选择一致。用 checkpoint80 的读出作用于本次已保存特征/本次预热基线，负向场景 logit 最大误差分别 `3.40e-7`、`1.69e-7`；`.35*tanh(logit)` 与命令相符。
- 三种神经模式各自的正、负场景前48帧（外扰出现前），画面、视觉量、神经特征、群组活动、logit、命令和实际转速均逐元素相同。这是新场景重置后面对同一输入的重复响应证据，不等于已经验证未见纹理泛化。

源码固定策略路径可确认：`src/malecns_lab/learning/yaw/training.py:235–253` 仅在 `gate/train` 分支创建 optimizer，`:318–332` 的评估分支校验所选 checkpoint 后 `load_parameters`，直接调用 `evaluate_episode`；`src/malecns_lab/learning/yaw/rollout.py:103–145` 在 `torch.no_grad()` 中重置状态并前向，不调用 loss/backward/step。两运行 `progress.json` 仍为80窗、80次 optimizer 更新、epoch_complete=true。没有把评估误当继续训练或更新模型的代码路径。

## 负向场景的独立指标复算

复算使用保存的 `actual_wz_rad_s[40:]`：RMS、绝对峰值、`sum(abs(rate))×.05` 的累计绝对转角；最后10帧为末段 RMS。恢复时间独立扫描外扰目标结束后的区间，要求剩余全部样本满足 `abs(rate)≤.03` 且至少有 `.5 s` 后续支持，与 `drones/yaw_env.py:91–129` 的语义一致。没有调用生产 `score_episode` 或 `control_gate` 来充当独立计算。

| validation-01 模式 | RMS rad/s | 峰值 rad/s | 末 .5s RMS rad/s | 累计绝对转角 rad | 恢复时间 s | 最大命令 rad/s |
|---|---:|---:|---:|---:|---:|---:|
| zero | .109217831 | .249354351 | .005586911 | .137293678 | .45 | 0 |
| untrained | .103113397 | .236695432 | .002193990 | .125082990 | .35 | .019471513 |
| frozen | .142956323 | .224663413 | .140068894 | .264164868 | null | .277269453 |
| joint | .217642058 | .322654601 | .316812008 | .359054852 | null | .349698752 |
| visual_p | .043878725 | .095927693 | .000001097 | .046138460 | .05 | .198968533 |

所有复算指标与对应 evidence 在 `rtol=1e-10, atol=1e-12` 内一致，null 一致；这里“最大命令”是绝对值。null 仅表示这段有限观察没有完成稳定恢复。

与正向结果合并后，相对各自 zero 的 RMS 改善为：

| 模式 | 正向场景 | 负向场景 | 两方向均值 |
|---|---:|---:|---:|
| untrained | +2.9716% | +5.5892% | +4.2804% |
| frozen | +24.5425% | −30.8910% | −3.1742% |
| joint | −208.6224% | −99.2734% | −153.9479% |
| visual_p | +59.8357% | +59.8246% | +59.8301% |

负值表示更差。frozen 不能因正向单场景改善而被报告为训练成功；joint 本次也没有显示脑内部训练相对 frozen 的控制优势。visual_p 证明这两个合成场景存在有效视觉反馈解，不是 MaleCNS 模型成功证据。

## 门槛判定与负方向机制

按登记阈值独立重建的 gate 与 `summary.control_gate` 逐字段一致：

```text
passed = false
scene_checks = {validation-00: false, validation-01: false}
mean_rms_improvement = -1.5394786734176225
direction_improvement[+1] = -2.0862235927353656
direction_improvement[-1] = -0.9927337540998795
aggregate_passed = false
static_required = false
static_present = 0
```

两方向的 joint 均不满足“单场景恶化最多5%”“峰值不超过 zero×1.10”“末段 RMS≤.03”；平均改善20%和两方向均改善也未满足。validation 没有独立无扰动整场景，因此本报告不补写静态测试通过或失败；已观察到的扰动前漂移仍是实际问题。

负方向记录补充了一个关键事实：joint 不是仅对正外扰反应过强。与正向场景相同，扰动前已有正向漂移；负外扰发生时，joint 的10帧命令全部为正、全部与当时教师反号，实际转速仍为 `+.07434…+.08345 rad/s`。外扰一停止，22帧恢复期继续全部正推、全部与教师反号，末段稳定在约 `+.31…+.32 rad/s`；最后一帧实际 `+.313509103`，外扰贡献仅 `−.001172014 rad/s`。外扰暂时抵消部分正向控制，撤去后自身持续正转，说明两个方向都出现了相似的错误自持旋转。

frozen 负向扰动前同样自行负转；负向扰动期曾出现部分反向制动，但恢复期22帧中18帧、末段10帧中8帧仍与教师反号，末段实际转速始终为负。这些现象与 首场景神经投影诊断 一起支持研究静止时序波动、方向可分性、读出尺度及闭环稳定性；不证明某一个调整必然修复问题，也不把本次阴性结果泛化为整个连接组方法不可行。

## 资源与终态

独立从十份 `resources.csv` 和六份 `mps_memory.csv` 计算采样值，并与 worker/supervisor 报告相互核对：

| 项目 | 实际范围或最大值 |
|---|---:|
| 神经 worker 单项耗时 | 423.051–495.715秒 |
| 十项 supervisor 耗时合计 | 2633.864秒 |
| 进程树 RSS 采样峰值 | 1,760,411,648字节 |
| MPS driver 采样峰值 | 1,525,497,856字节 |
| MPS current 采样峰值 | 1,089,409,024字节 |
| 系统可用内存采样最低值 | 2,121,646,080字节 |
| 各项有效内存预算 | 2,203,107,328–3,626,958,848字节 |
| 固定系统预留 | 1,073,741,824字节 |

各项有效预算均为 `min(6 GiB, 启动前可用内存−1 GiB)`；RSS、MPS driver 各自低于其本项预算，系统可用内存样本均高于1 GiB，worker/supervisor 均在600秒内结束，MPS sampler 无错误。合计耗时是作业监督时长之和，不包含所有父进程间隔或人工分析时间。RSS 与 MPS 统一内存数值不能相加，采样不是对每个瞬时峰值或每个 GPU kernel 的独立观测。

## 处置与停止

本轮固定模型的完整验证结果为负，应保留所有 R1 结果、最终 checkpoint 和原始记录，不追加本轮训练预算、不放宽门槛、不打开锁定测试、不发布仿真接入就绪。需要新的训练侧诊断与新版本方案；根因证据和改进候选必须与下一版实际成效分开。

本任务只新增此验收报告。静态源码阅读、短 NumPy/标准库只读取证未导入 Torch，未运行生产模型、测试、build、lint、训练或拟合，没有读取锁定测试数据。按 S0 阅读新增文档 diff 后停止；不重复已有训练、完整图、源码摘要及第一场景诊断验收。
