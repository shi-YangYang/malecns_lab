# MaleCNS／X2 训练与控制故障诊断材料

调查日期：2026-10-24；时区：北京时间 CST（UTC+8）。本报告面向独立技术顾问，依据现存源码、运行档案、参数文件、Git 及会话记录形成。**没有在本次调查中修改业务代码、重新训练、运行神经推理或性能基准、终止进程、删除或覆盖科研资产。**本次仅新增此报告。

证据等级：**已确认**＝源码及原始产物直接支持；**技术推断**＝有相符证据但未完成因果隔离；**未知**＝缺少必要记录或实验。`completed` 仅表示注册的计算程序结束；工程测试通过、离线指标合格、实际仿真行为合格、真机合格是不同结论。

## 1. 核心判断与调查范围

**目前没有一个新版 MaleCNS／X2 五行为模型通过完整行为验收。继续按当前设计直接增加训练时长，没有足够技术依据。**这不是单一报错：已发生的数值梯度异常、输入和部署分布不匹配、少样本读出过拟合、内部参数有限更新反而使行为退化、控制映射未覆盖实际输出幅值、慢速神经推理等问题同时存在。修复启动错误并不能解决学习有效性。

最关键的已确认事实：

1. 原 90 分钟超时 run 中，约 **87.52 分钟（97.25%）**在首次生成完整脑特征缓存。复用缓存后 200 次读出更新只需 **251.670 秒**，但开发验证 **0/50 条完整序列正确**。
2. 夜间约 8 小时不是 8 小时持续训练。可识别的 **6 次学习试验尝试中，5 次计算完成、1 次读取错误失败**；另有 **8 次诊断／启动预检**，以及 **1 次零训练的执行契约派生**。本轮只有 **2 次内部 theta 的 Adam 更新**。
3. 原始联合反传曾出现 **829/11752 个 theta 梯度 NaN**。缩小 surrogate backward gain 后该探针有限，但梯度 L2 仍达 **5.137×10¹¹**。后来另一 theta-only 更新梯度有限、实际改变 4099 项参数，却让训练 CE **1.456290→3.913080**、正确帧 **8/8→2/8**，验证完整前进 **1/3→0/3**。
4. 连续交互读出训练请求 **16/16**，开发验证只有 **3/7**；配对读出训练分支 **18/18**，验证请求 **2/8**，END **0/2**，GREET 错误触发 ADVANCE。不是仅靠“多训几轮”已经有改善趋势的状态。
5. 最新桌面实际 SHOW_FORWARD 输入已经到达模型，模型 **8.546 秒后输出 TURN_RIGHT/START**，执行接口也收到了右转命令。因而该次“前进不对”的直接原因不是按钮没有传到后端。
6. 现存 **34 个 X2 checkpoint 数组文件**均与各自元数据 SHA256 一致；没有发现已登记成功模型被覆盖、损坏或遗失的证据。更准确的表述是“尚未得到合格模型”。

### 时间窗口与当前状态

执行目标记录的工作段为 **2026-10-23 21:29:10 至 2026-10-24 06:08:07，共 8 小时 38 分 57 秒**。用户提到的 90 分钟 readout 训练开始于 10/23 15:37，属于此前过程，单独列出。调查现场约 10/24 11:45；若机械取此前 8 小时，应为 03:45—11:45，与实际讨论的夜间工作段不同。

11:45 的只读进程检查未发现活动中的本项目训练 worker；旧执行目标处于 paused，本次没有恢复。存在的 `execute.claim`／`worker.claim` 文件只是历史执行凭据，不能单凭文件存在认定仍有训练进程。本次没有自行终止进程。

文件时间限制：前置四个 run 有 `registration.created_utc`；夜间多数只有 monotonic 时间，表中绝对开始／结束主要由 registration、summary 的创建／修改时间推定，标记为“≈”。目录出生时间与实际执行不总相同。standing r1 的目录建于 05:49，真正执行在 05:52。目录名中的时间也不全使用同一时区，不能直接据名字判定先后。

## 2. 技术环境与实际训练对象

### 环境摘要

| 项目 | 实际证据 |
|---|---|
| 主机 | Apple M5，10 个物理／逻辑 CPU 核，16 GiB 统一内存 |
| 系统 | macOS 26.4，build 25E246，arm64 |
| 训练解释器 | `.venv-train`，Python 3.14.6 |
| 训练框架 | PyTorch 2.14.1；NumPy 2.5.4；SciPy 1.18.1；psutil 7.2.2 |
| 神经计算 | 宿主 macOS 的 `mps`，float32；`PYTORCH_ENABLE_MPS_FALLBACK=0` |
| 注册线程 | CPU threads=4，interop threads=1 |
| 桌面 | Electron 45.0.0，TypeScript／Three.js |
| 仿真路线 | 官方 MC/Sim v1.0.0-20260522、AimDK 1.1.0；私有 Lighter VM、linux/amd64 容器，VirtualGL／Mesa／Venus／MoltenVK 图形路径 |
| 设备边界 | 无真机 X2 接入；本文没有真机能力结论 |

现场内存快照：available=5,492,047,872 B，约 5.115 GiB；swap used=1,664,483,328 B，约 1.550 GiB。**这是调查时状态，不能据此反推训练期间交换瓶颈或当前必须立即恢复训练。**训练期日志没有连续 swap/pagefault、CPU/GPU 利用率或热降频数据。

训练注册环境原文摘录（readout r1）：

```json
{
  "backend": "X2_MPSLIF_with_explicit_spike_backward_gain",
  "cpu_threads": 4,
  "interop_threads": 1,
  "device": "mps",
  "dtype": "float32",
  "fallback": "0",
  "machine": "arm64",
  "python": "3.14.6",
  "numpy": "2.5.4",
  "scipy": "1.18.1",
  "torch": "2.14.1",
  "spike_backward_gain": 0.1,
  "system": "Darwin"
}
```

### 模型与训练目标

当前训练不是从机器人传感器端到端学习行走动力学，也不是对完整连接组的每条突触独立优化：

- 完整参考图包含 **166,700 个神经元、25,582,938 条聚合后的非零有向连接、124,177,617 个原始突触计数**；不要把原始突触数与矩阵非零边数混用。图拓扑和连接符号固定。
- 可训练内部变量为 **11,752 维按类型共享的突触前增益参数 theta**，增益形式为 `1 + 0.5*tanh(theta)`；不是 166,700 个神经元参数各自独立学习，更不是所有边权学习。LIF 的膜时间常数、阈值等未在这些试验中训练。
- 输入是57个工程通道，包括任务 one-hot、能力集合、执行状态、站稳、速度／角度／新鲜度，映射到 **3,377 个 R1–R6 输入神经元**。每个输入神经元分配一个通道，编码占据率乘150Hz，确定性相位累加产生事件，事件电压增量68.75mV。已检查数据中`perception.valid=False`；当前五按钮试验不等于由原始相机图像理解人类意图。
- 输出特征是 **1,314个DN在两个连续250步窗口的平均膜电位，拼接为2,628维**；三个线性头分别输出5类行为、3类阶段、1个强度。行为顺序为 `HOLD, ADVANCE, TURN_LEFT, TURN_RIGHT, WAVE_RIGHT`，阶段为 `START, CONTINUE, FINISH`。总参数35,413，其中theta11,752、三个头23,661。
- 期望答案由工程 `teacher_step` 和合成反馈 fixture 产生。训练目标是模仿该规则在给定反馈下的决策，不是来自真人示范、真实任务奖励或真实机器人闭环轨迹的强化学习。
- 新连续交互 profile **没有 WAVE 能力**，因此 GREET 对应 HOLD/FINISH。即使这个标签拟合成功，也不代表模型学会挥手。所有强度目标为 0.5，不能证明强度头学会了幅值调节。
- 原 readout／joint 使用交叉熵及掩码参数 MSE、Adam；后续冻结脑连续读出改用标准化 DN 上的 ridge 拟合；theta-only 冻结六个头参数、仅对 theta 做一次 Adam。不同模式不能统称同一种“GPU 训练”：完整脑前向／反向和 torch 头走 MPS，而 NumPy ridge 求解及数据处理在 CPU。

选择依据能从研究约束和代码结构还原，但没有算法选型对比实验证明它最优：固定完整连接组是项目使命约束；类型共享gain保留拓扑与符号、减少参数量；读出训练使行为映射可学习并允许复用昂贵神经缓存；surrogate用于给硬脉冲提供可用的近似反向信号；后续ridge／单次theta试验用于隔离读出拟合与内部动态更新。**这些是设计动机或从实现作出的解释，不是“该方法已经适合X2”的实证结论。**

## 3. 完整时间线与运行计数

本报告路径统一以工作区 `/Users/yang/project/malecns_lab` 为根。表中保留原 run 名，方便检索；关键原始内容已嵌入正文，不要求外部顾问能访问本机。

### 3.1 前置运行：原始梯度异常与 90 分钟超时

| 开始 CST | `data/` 下运行名 | 模式／参数／数据 | 进度与耗时 | 结果 |
|---|---|---|---|---|
| 10/23 14:17:08 | `x2_behavior_probe_20261023_r1` | 原 surrogate；两帧 1000 步信用窗口，600 步预热；只第二帧 loss；joint，Adam lr=.001 | 37.366 s；0 更新 | theta 梯度 829 NaN，更新前拦截 |
| 14:58:52 | `x2_behavior_probe_20261023_r2` | 相同 forward／数据；spike backward gain=.1；同 Adam | 38.575 s；1 更新；theta 3427 项变化 | 数值探针完成，无行为验证；仍有 5.137e11 梯度范数 |
| 15:37:53 | `x2_behavior_readout_20261023_r1` | 冻结 theta；50 train＋50 validation episodes；每帧 500 步；Adam lr=.001，200 更新上限，5400 s 上限 | 5400.138 s；818 帧缓存；168 更新；167 次完整验证历史 | watchdog `wall_limit`，returncode −15；无最终 worker result，12 个 best-so-far checkpoint 保留 |
| 17:52:56 | `x2_behavior_readout_20261023_r2` | 认证复用 r1 缓存；相同初始化重新开始 fresh Adam；200 更新，600 s 上限 | 总 251.670 s，worker 225.386 s；200 更新／200 验证；0 神经调用 | 计算完成；按最低 validation loss 选 0149；完整序列 0/50，行为不合格 |

r2 不是从 r1 第 168 步继续优化；它复用昂贵的脑特征，重新进行了廉价头训练。

### 3.2 夜间学习、诊断与预检

共同证据：各目录的 `registration.json`、`summary.json`、`worker_result.json`、`comparison.json`、`resources.csv` 和封存 `sources/`。没有对应文件时在表中明确标注。

| ≈开始—结束 CST | `data/` 下运行名 | 模式／数据／参数 | 实际进度 | host wall | 结果 |
|---|---|---|---|---:|---|
| 10/23 21:49:53—21:49:54 | `x2_behavior_short_trial_20261023_r1` | 计划 joint；family007/107，五任务各前2帧；SG .01，theta lr1e−4、head lr1e−3，梯度组 cap1 | 0 更新，未训练 | 0.856 s | 错误寻找不存在的 initial-checkpoint |
| 21:50:38—21:54:41 | `x2_behavior_short_trial_20261023_r2` | 上述路径修正；DN 标准化；同平衡 joint 协议 | 1 Adam；theta 3108 项变化；验证前后各10帧 | 243.758 s | 计算完成；val loss 2.899148→3.847965，五任务两帧前缀 0/5→1/5，仅右转 |
| 22:50:58—23:00:28 | `x2_behavior_short_trial_yaw_20261023_r1` | 冻结脑；旧编码缓存／仅去 absolute yaw 对照；train42＋val57帧；两条件 ridge α1 | 99 新脑帧；2 次 ridge | 570.166 s | val 行为 F1 .377334→.771085；完整序列 0/5→1/5，仅 END；train 两条件均5/5 |
| 23:53:47—23:57:18 | `x2_behavior_short_trial_interaction_20261023_r1` | 修正相对目标、能力、连续状态；train2会话24帧，val1会话10帧；ridge α1，std floor .1mV | 34 帧，1 ridge，0 theta更新 | 210.574 s | train请求16/16；val3/7，前进启动0/2，GREET误ADVANCE |
| 10/24 00:22:31—00:23:28 | `x2_behavior_short_trial_interaction_causal_20261024_r1` | 两个前态，仅换 task GREET/FORWARD，加复现分支 | 8 帧，0 更新 | 57.446 s | 诊断完成；任务差异远小于前态差异、跨前态不稳定 |
| 00:58:04—01:03:35 | `x2_behavior_short_trial_interaction_paired_20261024_r1` | 原24训练＋18同前态配对；12验证帧／8请求；ridge α1e−4 | 54 帧，1 ridge；配对训练18/18 | 330.864 s | val2/8，前进2/3，END0/2，GREET误ADVANCE；保存 e224de |
| 01:35:46—01:36:37 | `x2_behavior_short_trial_task_rate_20261024_r1` | 固定两个前态／两任务，150 vs 300 Hz任务编码 | 8 帧，0 更新 | 51.724 s | 150Hz基线精确；300Hz筛选FAIL，未启动后续训练 |
| 02:08:00—02:08:48 | `x2_behavior_short_trial_task_path_20261024_r1` | 原150Hz；2前态×2任务×100ms；1ms桶、层级探针 | 8帧／4000步，0更新 | 48.787 s | 100ms DN任务差异仍跨历史不稳定；FAIL |
| 02:58:44—02:59:29 | `x2_behavior_short_trial_task_polarity_20261024_r1` | 任务端口改成 `(1-one_hot)/4`；不重跑旧baseline | 8 帧，0 更新 | 46.433 s | 筛选FAIL；未做原计划66帧训练 |
| 03:39:45—03:45:14 | `x2_behavior_short_trial_theta_20261024_r1` | paired来源；只训theta，六头冻结；SG .01，Adam lr1e−4、cap1；6分支8梯度帧，新连续12帧验证前后 | 48逻辑帧；29200实际step；1更新；4099 theta变化 | 328.843 s | train CE及正确率恶化；val请求2/8不变，完整前进1/3→0/3；FAIL |
| 04:35:04—04:35:57 | `x2_behavior_short_trial_theta_derivative_20261024_r1` | 固定封存前态与方向 d；theta0、±d/16，各2帧；仅诊断 | 6 帧，0 optimizer，theta恢复 | 53.879 s | 小扰动已改变大量放电事件；不能以方向导数代替实际收益 |
| 05:24:45 文件派生 | `x2_behavior_sim_execution_20261024_r1` | paired参数原样复制，仅改SIM执行契约及身份 | **0训练、0参数更新** | 不适用 | 78138c；后续桌面验收仍失败 |
| 05:52:47—05:52:48 | `x2_behavior_standing_causal_20261024_r1` | 计划实际首帧与仅standing=True反事实 | worker启动；0帧、0更新 | 1.165 s | `Source binding changed` |
| claim06:00:22；后验快照06:00:58 | `x2_behavior_standing_causal_20261024_r2` | 修正JSON比较；同2帧协议 | worker未启动，0帧 | 未记录精确时长 | `Insufficient safe brain budget` |
| claim06:04:33；后验快照06:05:48 | `x2_behavior_standing_causal_20261024_r3` | 同协议，只换新输出路径 | worker未启动，0帧 | 未记录精确时长 | 同一预算拒绝；后续目标阻塞 |

计数：14 次计算启动／预检尝试＝6 次学习尝试＋8 次诊断／预检；12 次 worker 真正启动、2 次启动前拒绝。学习尝试中，joint r1 没有执行学习；joint r2、yaw、interaction、paired、theta 五个程序计算完成。冻结脑共4次 ridge 拟合，内部theta共2次Adam更新。另1个执行派生不计训练。前置另4次，所以当前共19个 `x2_behavior*` run 目录。

12 个已启动监督 run 的 host wall 合计 **1944.494 秒＝32.408 分钟**，含启动失败与诊断；不是 GPU busy 时间。没有证据显示这些夜间训练 worker 被用户人工终止。原 readout r1 是监督器超时发送终止，不能写成人工取消。

### 3.3 同期仿真与桌面调试：不能算作训练

以下路径在 `data/x2_sim/agibot/data/var/`。起点为目录文件时间近似，状态来自 `host_summary.json` 或 `behavior_host_summary.json`。保留不同层的失败语义。

| ≈开始 CST | run | 总 wall s | 结果 |
|---|---|---:|---|
| 10/23 22:26:24 | `skills-20261023-222502-a31f805e` | 196.748 | Official UI preparation timed out；无技能动作 |
| 22:30:47 | `skills-20261023-223100-7ebc21a4` | 191.947 | 同上，无动作 |
| 22:41:06 | `skills-20261023-223900-156bc823` | 311.865 | 同上，无动作；已经加大准备预算仍失败 |
| 23:04:28 | `skills-20261023-230200-0d671f24` | 140.188 | WAVE_RIGHT 为 UNKNOWN，`wave_feedback_unavailable` |
| 23:32:03 | `skills-20261023-233100-1ef27ab8` | 246.452 | 固定挥手自然结束并归位，单技能 COMPLETED |
| 23:38:50 | `skills-20261023-233900-98dcab71` | 267.732 | 固定挥手取消试验 CANCELLED，`simulation_cancel_returned`；取消协议通过 |
| 10/24 02:10:54 | `desktop-20261023-181053-eeeb5ecd` | 101.872 | UI准备超时 |
| 02:14:29 | `desktop-20261023-181429-d03ce244` | 101.158 | UI准备超时 |
| 02:21:24 | `desktop-20261023-182124-49b225aa` | 348.812 | `Reserved stopping/cleanup window reached`；0脑decision；resources_stopped=true、stop_confirmed=false |
| 02:29:18 | `desktop-20261023-182918-efd8350a` | 337.974 | 两次脑decision，前进约.04079m并反馈；最终cleanup超时，整场失败；stop_confirmed=true |
| 03:52:52 | `desktop-20261023-195251-6bb4e957` | 204.482 | UI准备超时 |
| 04:14:34 | `desktop-20261023-201434-aca12633` | 257.674 | 第一请求前进.04112m；后续前进最终.00603m且侧偏更大，执行FAILED→bridge退出 |
| 04:38:55 | `skills-20261023-203855-8d5cf051` | 190.867 | UI准备超时，无动作 |
| 04:43:15 | `skills-20261023-204315-5d893449` | 222.758 | 无脑，连续两段.1m/s×2s；第一段合格，第二段最终前进.00615m失败 |
| 04:56:04 | `skills-20261023-205604-c569d60d` | 185.676 | 无脑，连续两段.2m/s×1s均COMPLETED，净前进.406766/.397224m |
| 05:29:14 | `desktop-20261023-212914-8aa7551a` | 109.575 | SHOW_FORWARD→TURN_RIGHT；右转最终净角不达门槛，bridge退出；整体FAIL |

这16场 host wall 合计 **3415.780秒＝56.930分钟**，包含 VM、UI准备、等待、推理、动作与清理；不是纯仿真计算时间。7场以UI准备超时结束。另有隔离SDK兼容构建记录 **223.999秒**。这些记录解释了一部分非训练时间，但不足以精确还原剩余每一分钟的编辑、审查、等待和离线分析；不能用相减结果冒充完整工时剖析。

## 4. 错误清单、原始日志与数值稳定性

### E1：NaN 梯度；被更新前保护拦截

来源：`data/x2_behavior_probe_20261023_r1/worker.log`、`gradient-audits/0001/audit.json`。调用栈中的行号为该次封存版本，不能替换成当前源码行号。

```text
File ".../learning/x2_behavior_training.py", line 529, in run_worker
    update = gradient_update(loss, boundary=boundary)
File ".../learning/x2_behavior_training.py", line 361, in _audited_gradient_update
    frozen.require(all(g["present"] and g["finite"] for g in grads.values()),
                   "Missing/nonfinite trainable gradient")
File ".../learning/x2_inference.py", line 43, in require
    raise ValueError(message)
ValueError: Missing/nonfinite trainable gradient
```

原始审计字段摘录；`brain.theta.gradient` 为排版时明确参数路径的标签：

```json
{
  "loss": 12.290372848510742,
  "loss_finite": true,
  "brain.theta.gradient": {
    "elements": 11752,
    "finite": false,
    "finite_l2": 9.804094287807786e35,
    "finite_max_abs": 6.778009709458196e35,
    "nonfinite_counts": {"nan": 829, "negative_inf": 0, "positive_inf": 0}
  },
  "first_boundary_v_gradient_l2": 4.277151859626434e16,
  "optimizer_step_attempted": false,
  "optimizer_steps_after": 0
}
```

确认 NaN 和极大梯度；**没有在此记录中发现 Inf，不应笼统虚构 NaN/Inf 都发生过**。参数未写坏。后续 gain=.1 同forward探针 theta gradient L2=513722846633.4384，max=476142895104，NaN=0，1次Adam完成。它仅证明这一次数值准入通过。

可能原因：1000步递归动力学的代理梯度传播产生巨大乘积／累积；hard threshold/reset/refractory 的导数近似可能改变优化几何。尚未逐神经步定位首个非有限张量，不能排除具体算子或实现细节参与；没有证据将其直接归因于 PyTorch MPS 驱动缺陷。

### E2：90分钟超时，非模型OOM

`readout_r1/result.json`监督信息：

```json
{
  "status": "failed",
  "reason": "wall_limit",
  "returncode": -15,
  "worker_status": "missing_or_malformed",
  "last_phase": {"phase": "readout_update", "update": 168,
                 "wall_seconds": 5399.386297209014},
  "wall_seconds": 5399.7657199589885
}
```

整个host为5400.138秒；第168次审计却明确：

```json
{"optimizer_step_attempted":true,"optimizer_step_completed":true,
 "optimizer_steps_after":168,"status":"completed","update":168}
```

因此准确进度是168次更新、167次完整验证历史，而非168次验证。终止边界在第168更新后的验证／持久化阶段，未记录精确最后一条GPU操作。

### E3：短试验缓存初始化路径错误

`short_trial_20261023_r1/worker.log`：

```text
File ".../learning/x2_behavior_trial.py", line 184, in run_worker
    partitions, matrices, center, scale, evidence, diagnostic = cache_diagnostic(reg["cache"], metadata)
File ".../learning/x2_behavior_trial.py", line 97, in cache_diagnostic
    source = inspect_checkpoint(cache / "initial-checkpoint")
File ".../learning/x2_behavior_training.py", line 243, in inspect_checkpoint
    metadata = frozen.read_json(directory / "checkpoint.json")
FileNotFoundError: [Errno 2] No such file or directory:
'/Users/yang/project/malecns_lab/data/x2_behavior_readout_20261023_r1/initial-checkpoint/checkpoint.json'
```

0.856秒、0更新失败。r2运行完成说明该启动问题得到修正；r2验证恶化说明这不是行为问题的根因。

### E4：序列化绑定检查误比较；不是checkpoint实际损坏

`standing_causal_20261024_r1/worker.log`：

```text
File ".../data/x2_behavior_standing_causal_20261024_r1/run.py", line 241, in worker
    frozen.require(protocol["initial_state"] == identity
        and protocol["expected_final_state"] == expected["final_state"]
        and protocol["observations"] == [o.to_dict() for o in observations],
        "Source binding changed")
ValueError: Source binding changed
```

原因已从保存JSON的list与内存tuple比较定位。r2改为全字段canonical比较，代码和CPU绑定检查记录通过；**r2/r3未进入神经前向，所以没有反事实实验结果**。不应写成“站稳因素已经修复／已经证实”。

### E5：两次启动前资源预算拒绝

`standing_causal...r2/r3/preflight-failure.json`共同字段：

```json
{"status":"preflight_rejected","reason":"Insufficient safe brain budget",
 "completed_frames":0,"worker_started":false,"simulation_started":false}
```

失败后快照：r2 available=2903326720B=2.704GiB，有效预算1.704GiB；r3 available=2783133696B=2.592GiB，有效预算1.592GiB。要求有效脑预算≥2GiB，同时保留1GiB系统余量。**这是保守预检拒绝，不是已经开始训练后OOM。**JSON中通用`can_launch=true`与试验额外2GiB门槛不是同一判据。

### E6：损失恶化、类别坍缩与泛化失败

readout r2选定0149，在402验证帧上：

```text
预测行为计数：HOLD=371, WAVE_RIGHT=23, TURN_RIGHT=8,
              ADVANCE=0, TURN_LEFT=0
预测阶段计数：CONTINUE=303, START=50, FINISH=49
behavior_macro_f1=0.159244
phase_macro_f1=0.451895
full_sequence_correct=0/50
best_validation_loss=3.036943 at update149
final_validation_loss=6.523457 at update200
```

这是明确的偏置／近似类别坍缩，不是五行为均衡学习。训练选中50集416帧的标签分布：HOLD180、WAVE137、ADVANCE35、LEFT32、RIGHT32，存在明显不平衡。训练loss从10.4167下降至.0852的早期区间，同时验证loss从10.2131升到61.0288；每次更新用不同episode，单更新训练loss不可当固定训练集曲线，但同一验证缓存的恶化具有可比性。

theta-only 原始比较摘录：

```json
{
  "train": {"before_ce":1.4562904641646288,"after_ce":3.9130795482215848,
            "before_correct_frames":8,"after_correct_frames":2,"frames":8},
  "screen": {"checks": {
    "train_ce_decreased":false,"validation_ce_decreased":true,
    "correct_requests_increased":false,"forward_not_regressed":false,
    "end_not_regressed":true,"greet_not_regressed":true},
    "supports_further_internal_learning":false,"deployment_qualified":false}
}
```

theta原梯度L2=11.7186946，clip至1；4099项变化，最大|Δtheta|=.0001000017。新连续验证CE5.830556→5.717393，但请求正确仍2/8；完整前进1/3→0/3，END0/2→1/2。不能用平均CE略降掩盖前进退化。

局部导数诊断（同一train-a两帧分支，**不是整个8帧训练均值**）：

| 条件 | 两帧平均CE |
|---|---:|
| theta0 | 1.456290199 |
| theta0+d/16 | 1.002381711 |
| theta0−d/16 | 3.326723817 |
| theta0+d | 9.245587852 |

方向点积：固定事件分支局部导数·d=−.000105630843；旧SG·d=−.013710865468；跨±d/16的中心差分斜率=−18.594736846。±d/16仍改变4099参数，最大6.251e−6；全图脉冲不同位分别166074／138615，最早5.9／11.7ms即分歧。**该有限差分跨越离散事件分支，不能当作传统光滑梯度检查通过，也不能据两帧挑选步长并宣称合格。**

未发现能够支持“所有层梯度消失”的系统证据；部分非零／零参数统计不等于完整梯度流诊断。没有记录到NaN被悄悄填零后继续训练；已见异常均留下failed记录或在更新前阻断。

## 5. 训练设计审查

### 数据与标签是否适合机器人任务

这些数据适合检验“工程状态→人工规则决策”的可拟合性，**尚不足以证明真实机器人连续互动能力**。主要缺口：

1. 合成反馈fixture不由模型自己的错误行为驱动；模型选错后，后续训练／验证输入仍沿教师预设轨迹继续。这是离线模仿与真实闭环分布之间的明确差异。
2. 旧数据每episode重置脑状态；所谓switch/repeat并未携带真实上一个请求的神经历史。线上却跨请求连续保留state。新连续数据修正了这一点，但只有2条训练会话／24帧，远不能覆盖连续状态空间。
3. 旧训练每帧包含WAVE能力，当前部署候选不含WAVE；旧非转向任务还有+.45 heading_error，线上相应值为0。绝对yaw对相对任务是不必要扰动。后续共享契约和去yaw修正了部分不一致，验证依旧失败。
4. 2,628维特征对应新连续24行、paired42行训练数据。ridge能够拟合这些行，不代表跨历史泛化；train16/16对val3/7、paired18/18对val2/8已直接体现差距。
5. 新连续训练6个SHOW_FORWARD启动全为standing=True；18配对分支也全强制True。最新线上首帧是SHOW_FORWARD＋IDLE/HOLD＋standing=False，这是缺失组合。旧大数据有False，因此不能泛化为“所有历史训练都没见过False”。
6. 新profile的GREET目标是HOLD/FINISH，普通CONTINUE覆盖不足，强度固定.5。当前训练目标本身不能支持“已学会挥手、连续调速、及时主动停步”的产品表述。
7. 按family／完整会话划分避免了相邻帧随机切分，但一些早期family只改绝对yaw；去yaw后前缀实际相同，不能将拟合相同输入当新场景泛化。后续validation已被反复查看并指导编码／数据修改，是**开发验证**；锁定测试未打开，尚无最终盲测。

### 是否只在学习人工规则

**监督答案确实来自人工规则；现有成果最多支持“尝试通过完整脑动力学表示去模仿规则”。**完整脑是否比直接线性／小MLP／有限状态基线增加泛化、记忆、抗扰动价值，目前没有同数据、同划分的有效对照证明。人工标签本身并非必然错误，但必须明确研究问题：如果目标只是按按钮触发五种已有官方技能，这种任务不天然要求完整果蝇脑；如果目标是研究连接组带来的因果行为价值，必须有对照和消融。

### 任务信息与历史状态的竞争

同前态task swap诊断中，任务造成的DN差异／前态差异RMS比：原始 **.002147**、标准化 **.012419**。两个前态分别将GREET／FORWARD都读为RIGHT/START与ADVANCE/START。跨前态任务差异余弦：原始 **−.03739779**，训练统计标准化后 **−.00605920**，不能混用。两个前态task delta RMS分别.11014056／.07140743mV，而state delta约43.23mV；同前态重复控制逐元素一致。

增频到300Hz、延长到100ms、改变任务极性，都没有在已测前态得到稳定DN任务表征。task-path显示前级L1/L2电导／膜电位差异较一致，但脉冲离散响应和末端DN差异不稳定。**已确认局部现象；“网络整体不可训练”或“所有历史下任务都不可分”仍属未证明。**

### 内部学习是否合理

surrogate gradient训练SNN是合理研究方法，不是因为forward二值就一定不能训练。但当前实现的代理导数、截断信用窗口、硬reset／不应期处理、类型共享gain约束与高维动态系统组合，需要单独验证其有限更新效用。现有两次本轮内部更新没有提供这种正证据。

读出与内部参数同时更新还增加归因难度：joint r2的保存数组对照支持头更新过冲是该次恶化的主要来源；随后冻结头只训theta仍恶化，说明“仅把头冻结”不能自动解决所有问题。也不能把这两种退化强行归结为同一个优化器bug。

关于代理梯度的一般性质，参见原始综述 [Neftci et al., 2019](https://arxiv.org/abs/1901.09948) 与理论研究 [Gygax & Zenke](https://arxiv.org/abs/2404.14964)：这类估计可用于SNN学习，但并不保证当前硬脉冲系统任意有限更新都会下降；文献不是本项目实现正确性的替代证据。

## 6. 性能调查：测到了什么，没测到什么

### 90分钟的阶段分解

来自readout r1的`mps_memory.csv`阶段标记：

| 标记 | 相对时间 s |
|---|---:|
| load_complete_graph | .054 |
| 首个train-features_episode | 3.728 |
| 首个validation-features_episode | 2646.471 |
| 首个readout_update | 5255.160 |
| host结束 | 5400.138 |

训练特征阶段2642.743秒；验证特征阶段2608.689秒；合计5251.432秒，占总时长97.25%。另一个口径，从parent开始至最后一帧缓存时间戳为5256.010秒，含初始化／收尾，约97.33%；两个数字边界不同，不是测量矛盾。

818帧×500＝409,000个任务神经步，另外100个episode各600步预热。缓存阶段还包括完整state保存、压缩、hash和IO；没有单独timer，不能把87.5分钟全部称为GPU kernel执行时间，也不能无测量声称磁盘是主瓶颈。

### 500步、更新和验证耗时

每步dt=.1ms，所以500步＝**50ms模拟脑时间**。

| 来源 | 测量口径 | 实际wall |
|---|---|---|
| readout训练缓存 | 同episode相邻完成时间，366间隔，含保存边界 | 中位5.606s，p95 5.904s，最大7.310s |
| readout验证缓存 | 同上，352间隔 | 中位5.589s，p95 6.139s，最大6.725s |
| yaw | policy调用＋MPS synchronize累计558.551914s／99帧 | 均值5.642s／500步 |
| continuous interaction | 195.959853s／34帧 | 5.764s／500步 |
| paired interaction | 314.377307s／54帧 | 5.822s／500步 |
| desktop201434 | 四次实际raw_decision计时 | 9.736283、8.159517、8.844601、14.986183s |
| desktop212914 | 一次实际raw_decision | 8.546158s |
| readout r1更新区间 | 含头forward/backward、全验证缓存读出、审计及可能保存 | 中位.826s、均值.860s，167区间 |
| readout r2更新区间 | 同上 | 中位.878s、均值.967s，199区间 |

读出每次更新后确实验证全部402个缓存帧，但**不会每次重新推演完整脑**。r2记录：

```json
{"MPSLIF.forward":0,"MPSLIF.reset":0,"MPSLIF.step":0,
 "X2BehaviorPolicy._step":0,"X2BehaviorPolicy.forward":0,
 "X2BehaviorPolicy.initial_state":0}
```

没有独立optimizer.step／validation timer，因此不能给“纯更新耗时”或“验证占比”的虚假精度。theta-only总328.843秒包含前态重放、训练分支、反向、前后验证、保存，不能说一次Adam本身耗时328秒。

theta-only确认29200实际step＝48×500前向＋1200预热＋4000 activation-checkpoint反向重算。该重算是节省内存的显式代价，符合[PyTorch activation checkpoint文档](https://docs.pytorch.org/docs/stable/checkpoint)，不是日志重试次数。

### 逐步校验与数据传输

task-path对4000步中的8步做了同步插桩，样本中位数：

| 局部阶段 | 中位 ms |
|---|---:|
| prior GPU wait | .159 |
| 输入调度 | .228 |
| 显式full-state validation | 4.638 |
| inline guards＋dynamics | 1.694 |
| sparse matvec | 4.639 |
| 返回 bookkeeping | .0115 |
| 总计，不含prior wait | 11.124 |

每神经步对v、g、18个pending全向量做finite检查，并把MPS条件转成Python分支；这是已确认的同步／扫描路径。**8个非随机插桩样本不能外推“全程40%都是校验”，也不能承诺优化倍数。**采样本身增加synchronize，且存在一个123.963ms离群步。

缓存读出每帧`feature.to('mps')`、结果回CPU；完整状态／输出归档同样回CPU。确认存在传输／同步路径，但无transfer bytes/time和kernel profile，尚无法给出其瓶颈占比。线上比离线慢也不能全部归因于仿真抢GPU，因无同步利用率记录。

### 内存实测

单位GiB；MPS driver与进程RSS在统一内存中会计范围重叠，**不可相加**。MPS日志每.1秒采样，峰值不是保证捕获的瞬时绝对最大值；[PyTorch MPS说明](https://docs.pytorch.org/docs/stable/mps.html)区分tensor与driver分配。

| run简称 | RSS峰 | MPS driver峰 | 系统available最低 |
|---|---:|---:|---:|
| probe r1／r2 | 1.526／1.516 | 1.487／1.493 | 4.612／4.128 |
| readout r1／r2 | 1.509／1.350 | 1.421／1.421 | 2.201／2.310 |
| joint short r2 | 1.391 | 1.495 | 2.342 |
| yaw | 1.530 | 1.421 | 3.365 |
| continuous | 1.702 | 1.421 | 4.759 |
| causal | 1.639 | 1.421 | 3.965 |
| paired | 1.428 | 1.421 | 2.798 |
| rate | 1.507 | 1.421 | 3.407 |
| path | 1.390 | 1.421 | 2.603 |
| polarity | 1.656 | 1.421 | 3.832 |
| theta | 1.388 | 1.517 | 3.123 |
| derivative | 1.479 | 1.493 | 3.188 |

本轮已完成run没有明确OOM终止记录。末尾两次资源预检拒绝已证实；缺少训练期swap／pagefault／热状态数据，不能把慢全部归因于16GiB内存或散热。日志只有tree RSS、system available、MPS allocation，不含CPU/GPU使用率。

## 7. 实际控制链与模型外问题

最新 `desktop-20261023-212914-8aa7551a/behavior_events.jsonl` 原始decision关键字段（省略2628维features数组，未改数值）：

```json
{
  "kind":"raw_decision",
  "inference_seconds":8.546158167009708,
  "decision":{
    "decision_seq":1,
    "behavior_logits":[-0.43610191345214844,0.5314908027648926,
                       -0.00027502328157424927,0.9048890471458435,0.0],
    "phase_logits":[1.4361019134521484,0.0,-0.43610191345214844],
    "parameter_raw":[0.0],"behavior":"TURN_RIGHT","phase":"START",
    "intensity":0.5,"neural_start_step":600,"neural_end_step":1100
  }
}
```

对应任务是唯一SHOW_FORWARD；gate接纳，99条实际命令`(vx,vy,wz)=(0,0,-.1)`，非零至归零1.996888秒。**没有非零前进命令，新固定前进档位未触发。**执行FAILED时净角−.015325621rad，未达原.02rad门槛；它曾到约−.08996后回摆，不能用峰值替代最终结果。`bridge_summary`较晚的最终清理姿态已继续变化，故这里引用FAILED事件时的实测值，而不是混用后来快照。

失败链原文：

```text
bridge_summary.reason = RuntimeError:Execution protection latched: FAILED
behavior_host_summary.reason = ValueError:Behavior bridge exited
behavior_worker_result.reason = InterruptedError:independent_emergency_stop
```

最后一项来自独立清理通道，不能据名称认定用户亲自点击了急停。host记录resources_stopped=true、stop_confirmed=true。最终FAILED未被后续神经前向消费，不能宣称完成了失败反馈再入脑的闭环。

新旧第一次输入对照：七个参数数组、输入映射、计算源码、23个数值初态数组一致；差异在237个输入端口，含standing、source_age和微小速度／角速度。新standing=False，旧True；新旧ADVANCE−RIGHT margin分别−.37340、+.74485。**站稳输入是强嫌疑，但未完成单因素反事实，不能认定唯一原因。**站稳定义是连续1秒满足新鲜度、模式、姿态、速度、播放器条件，不等于MC瞬时STAND标签；旧记录缺少内部稳定窗口历史。

模型外执行缺口也已确认：旧适配把强度.5映射为.1m/s。固定.1m/s×2秒连续两段的第二段仍退回，仅净前进.00615m；固定.2m/s×1秒的两段均通过，净前进.40677/.39722m。官方本地示例`aimdk/src/examples/src/mc/mc_locomotion_velocity.cpp:32,133`拒绝非零forward<.2；官方文档`.../locomotion.html:753`描述起步量级约.1m/s且因模型／固件而异。证据支持原执行映射缺少低强度资格，**不能由此断言闭源MC精确死区机制**。新SIM profile固定.2，只是执行适配，不是脑学会新强度。

SDK另有Set预设消息v1.1七字段／MC v1.0六字段契约差异。隔离旧消息helper后固定挥手自然结束和取消分别实测通过；后续会话工程接入有检查记录，但当前脑候选能力仍无WAVE，不能宣称脑控打招呼完成。

另外已修过桌面状态隐藏／旧轮询覆盖、新旧时钟域对齐等工程问题。它们影响用户能否看见真实状态，不能解释所有模型分类错误。Node/Python monotonic原点曾差约70310秒；新v2截图采用nonce括号时间区间，视频仍无Python锚。图谱真实批数据／显示帧率也不等于脑决策频率。

## 8. 现有模型、缓存与恢复能力

### 保留成果与资格

| 资产 | 实际状态 | 能否作正式X2模型 |
|---|---|---|
| probe r1 initial 66c471… | 未训练初态；失败前0更新 | 否 |
| probe r2 final 436227… | 单次数值探针更新 | 否，无行为验证 |
| readout r1 12个best-so-far | 超时前中间保存；含0149 | 否，未完成200步且行为失败 |
| readout r2 0149＝3f0c556… | 200步训练完成后按开发val最低loss选择；0/50完整序列 | 否；仍为默认qualification配置 |
| joint short r2 e3a583… | 一次联合更新，val恶化 | 否 |
| yaw两个readout NPZ | 闭式拟合完成，完整序列1/5仅END | 否，非正式checkpoint |
| continuous b1ebe739… | 可严格恢复；原验证3/7请求、整会话0/1 | 仅受控排障 |
| paired e224de70… | 可严格恢复；验证2/8，END0/2，GREET误动作 | 仅受控排障 |
| theta-before 30c446… | paired参数换trial身份，未新增学习 | 否 |
| theta-after 24dce9bd… | 一次theta更新，效果退化 | 否，保留失败证据 |
| execution-derived 78138c1d… | paired参数原样复制，仅换执行契约；桌面SHOW_FORWARD选RIGHT | 仅受控排障，无正式资格 |
| causal／rate／path／polarity／derivative／standing | 诊断或失败预检 | 不是新训练模型 |

两套readout均保留编号`0001,0016,0024,0030,0031,0036,0037,0050,0084,0139,0140,0149`。34个实际checkpoint数组均通过**本次只读哈希核对**；4个`provenance/checkpoint.json`仅来源元数据，不是缺失了数组的模型。

关键哈希：

```text
readout r1/r2 0149 arrays.npz:
fcc4c9ab0b718f975f56e83682301599c2fb3f75cd5abb9ed6aa829968959183
paired / theta-before / sim_execution arrays.npz:
c2e7f735a72ee38c26eba8189be4ee32495972cac41e35eb849f1bff9a500ab4
theta-after arrays.npz:
7729b40280929959f5147d663525aa21c77a52a4cc2ebf8404d1358cda5712dc
```

不能证明从未存在后来消失、未登记的文件；但现存登记资产未发现覆盖／损坏，也没有“已训练合格后丢失”的实证。

### 哪次验证最好

不同数据集／指标不能混排。原50序列验证0/50；yaw完整正常序列1/5；continuous原验证3/7请求；在后续paired开发验证中，**旧continuous baseline为4/8，新paired为2/8**。不能把旧baseline4/8冒充paired的成绩。theta新验证前后均2/8。所有这些都未通过完整会话及五按钮验收，锁定独立测试未打开。

仿真中曾出现单个前进请求完成并收到后续HOLD，但同一run或后续请求失败。这证明链路部分能力存在，不能替代连续任务可靠性。固定技能两次前进／挥手成功不含脑任务选择。

旧10/19三轴轨迹模仿资产仍在：CPU joint60窗口、frozen10窗口均未完epoch；MPS r1/r2 GPU0窗口，r3累计CPU60＋GPU100＝160窗口；gpu_recovery完成新目标160窗口。它们明确是`offline_teacher_trajectory_imitation_not_neural_closed_loop`，不是当前X2五行为。旧单增益probe的loss163.012→161.558也不能引用为新模型行为成功。

### 能否续训、哪些可以复用

不能笼统说“没保存optimizer”。checkpoint保存器支持Adam的step／exp_avg／exp_avg_sq；0149含18个optimizer数组（六个头参数各三项），theta-only含3个；ridge候选无Adam状态是正常的。

**但现有`restore_checkpoint`恢复的是模型参数，训练入口显式fresh optimizer，没有通过任意中断点恢复Adam、数据游标、RNG、完整神经状态的整套续训协议。**有数组不等于已经支持精确恢复。readout r2是缓存再拟合而非精确续训。

可安全复用的边界：

- 原始连接组、注释、未改写来源、数据集、封存sources／环境／协议可作为只读来源。
- frozen DN缓存仅在图、theta、编码、数值实现、初态、完整序列及相关身份一致时复用；r2已有成功复用证据。
- 改过yaw／频率／极性／能力／状态历史的旧DN，不能直接当新版输入下的特征；改theta后旧神经轨迹也失效。
- 失败候选、梯度、前态、logit可作离线审计对照；“可加载”不等于“值得继续优化／部署”。
- 保留封存hash，不通过改hash伪装同一checkpoint；兼容迁移应生成显式新身份并保留父来源。

## 9. 修改历史及证据等级

当前HEAD是 **4917126，2026-10-23 21:08:25+08:00，`refactor: 清理CPU部署并整理工程目录与历史文档`**。此前414d3f5为18:15保存多行为阶段；6ba3fb4为11:56 Electron；fbc5b2d为05:23纵向反馈；a2108a4为04:10冻结推理优化。**夜间之后没有逐修复Git commit**。调查开始的tracked diff为27文件、1783新增／283删除，另有多个untracked试验模块／测试；不能把未提交diff伪装成精确提交时间线。

| 变更 | 证据／可定位模块 | 已验证到哪一层 |
|---|---|---|
| 原SG→gain .1 | probe快照`brain/mps_trainable.py` 4408c5ec→a2d28cfc；policy 0fc9c201→07b06afa；training 9ed3fb35→3bc3d9d4 | 相同forward数值探针由NaN失败变成1次更新；无行为资格 |
| 增加缓存认证复用 | readout快照training 3bc3d9d4→b44cfafb；新增cache f230c34c | r2实际0神经调用，200步完成；行为仍0/50 |
| initial checkpoint路径修正 | short trial runner ee6d930f→ac39aa0f | r2启动成功，随后val恶化 |
| SG .01＋DN标准化＋平衡joint | `x2_behavior_trial.py`、`x2_behavior_trial_policy.py` | DN与旧缓存精确；头迁移误差约2.62e−7；更新有限；val失败 |
| 去absolute yaw／共享相对目标 | `senses/x2_relative_heading.py`、`control/x2_interaction.py`、session/data模块 | 99帧对照、F1改善但完整1/5；非全部修复 |
| 连续／配对数据和ridge | `x2_interaction_data.py`、trial/paired及policy模块 | train拟合，held-out顺序仍失败；没有部署资格 |
| task rate/path/polarity诊断 | 三个独立模块及绑定policy | 各8帧固定诊断完成且筛选FAIL；没有后续大训练 |
| theta-only／局部导数 | `x2_theta_trial.py`、`x2_theta_local_derivative.py` | 数值复现及失败指标有原始产物；未证实有效学习 |
| SDK旧消息隔离、取消归位判据 | preset helper/identity/compat、`control/x2_skills.py` | 构建和单技能自然／取消实测；不是脑控挥手 |
| 桌面状态、时钟区间 | renderer live-status、controller、recording、session | 历史局部软件检查／审查及v2采样；完整UI回放选择、实际急停按钮验收仍有缺口 |
| 固定前进执行profile | skill_profile、sim_execution_policy、host/session/bridge | 固定技能两段合格；派生参数未变；实际脑控选错，未触发新ADVANCE |
| standing JSON比较 | data中r1→r2一次性脚本 | CPU canonical绑定检查记录通过；完整脑试验被预算拒绝，因果结论缺失 |

其中“历史软件检查／审查通过”来自当时任务记录与会话回执；并非本次重新运行。没有完整统一测试报告可以对每个编辑逐项背书。源码快照能证明“到该run时该代码已经存在”，不能恢复所有中间编辑的精确时间或全部自动审查过程。

已确认重复问题：输入／状态分布变化后的泛化失败；同一UI准备超时7场；资源预算拒绝2次。路径错误和JSON比较错误是不同的边界实现缺陷。修复前者后出现后者不自动证明前者造成后者。当前也没有证据证明每一次行为退化都是新代码回归；一些是更严格的验证暴露了旧缺陷。

## 10. 问题分级与根因关系

严重度：P0＝阻止产品主线验收；P1＝阻止当前研究协议可靠推进；P2＝局部缺陷／证据质量问题。表中“仍存在”区分具体错误与未解决风险。

| ID／类别 | 严重度、确认度 | 证据 | 最可能根因／尝试 | 验证与残留、相互影响 |
|---|---|---|---|---|
| A1 数值反传异常 | P1，已发生 | 829NaN、finite max6.778e35 | 长时递归SG尺度；改gain .1/.01并审计／clip | 指定后续probe有限；整体更新有效性未解决，关联B4 |
| A2 wall超时 | P1，已确认 | 5400s、returncode−15 | 缓存生成已占97.25%预算 | 缓存复用运行完成；不代表行为改善，关联C1 |
| A3 路径／JSON绑定错误 | P2，已确认 | 两个完整stack、0更新／0帧 | run目录假设、序列化类型边界 | 路径下一run通过；JSON仅CPU验证，神经反事实未运行 |
| A4 内存预检拒绝 | P1，已确认 | r2/r3无worker，effective<2GiB | 可用统一内存不满足协议余量 | 没有降低门槛；历史阻塞真实，非OOM；调查时余量已变化 |
| B1 任务与训练目标覆盖不符 | P0，已确认 | GREET=HOLD，无WAVE，固定强度.5 | 能力profile／教师任务与产品目标未统一 | 部分共享契约修正；五行为训练目标仍不完整，关联D3 |
| B2 状态分布／数据覆盖不足 | P0，现象确认，具体因果部分推断 | 24/42训练行；train高val低；线上standing缺组合 | 连续历史、微小反馈扰动、任务信号弱 | 去yaw/continuous/paired仍FAIL；standing尚未隔离，关联D1 |
| B3 类别偏置／验证选择问题 | P0，已确认 | 0149预测ADVANCE/LEFT皆0，0/50 | 标签不平衡、顺序更新、读出过拟合／目标选择 | 最低loss仍行为不合格；validation多次开发使用；无盲测 |
| B4 有限更新收益不可靠 | P1，已确认 | theta train CE升、前进退化；小扰动换放电分支 | SG近似＋事件敏感性＋步长／参数化；贡献未完全分离 | 冻头theta单更新和局部诊断未支持扩训；不是简单梯度有限即可通过 |
| C1 完整脑前向成本 | P0用于实时控制／P1研究，已测 | 离线500步约5.6s、缓存87.5min | 完整稀疏图500次递推＋校验／归档 | 无纯kernel剖析；不能只优化Adam，关联D2 |
| C2 重复校验／传输／验证开销 | P2，代码确认、占比未知 | 每step全状态扫描；每update全缓存head验证 | 强检查粒度和小张量往返 | 8步局部校验≈matvec；未做性能对照，不承诺收益 |
| C3 不具备端到端精确续训 | P1，已确认 | 保存optimizer但restore仅参数；fresh Adam | 模型checkpoint≠运行快照 | 缓存可复用，任意中断恢复缺证据；加剧返工成本 |
| D1 点击已到但行为选择错 | P0，已确认 | SHOW_FORWARD→RIGHT真实logits／命令 | B2/B3，不能只归按钮 | 最新验收FAIL；固定前进适配未触发 |
| D2 慢反馈／主动停止未证实 | P0，已确认 | 8–15s/decision；固定技能先结束 | 模型延迟远高于动作时长 | 执行层停稳保护存在；不能称脑实时停止，END仍不合格 |
| D3 低强度执行与挥手契约 | P0，已确认局部问题 | .1档重复失败；SDK字段不同；无WAVE候选 | 模型输出支持域与官方技能不匹配 | .2固定技能／挥手单技能合格；脑五任务未通过 |
| D4 UI／归因／回放证据缺口 | P2，已确认 | UI准备7次超时；一场2请求只见1次工具点击；旧时钟不一致 | UI预算、状态传播、事件归因及异构时钟 | 局部工程修正；第二请求来源不明，不能归用户／自动点击 |

**高置信的并存事实**是少量人工fixture、训练拟合但验证失败、部署观测组合与训练覆盖不一致、实际首帧错选。它们之间的具体因果贡献仍需隔离。真实首帧同时变了standing、source_age及速度；standing是离散语义变化，不能将全部变化统称微小噪声，也不能宣称已找到唯一根因。完整脑计算成本同时使新表示实验昂贵、线上反馈缓慢。数值SG问题是另一条独立链，不能用数据修正自动覆盖；执行层幅值／版本契约又是第三条链。

不确定部分：任务信号弱究竟主要由输入端口选择、抑制传播、动态背景、DN特征选择、时间窗还是读出容量造成；站稳因素是否足以单独改变那次首帧；代理梯度偏差与真实有限事件跃迁各贡献多少。现有实验不能作唯一根因判决。

## 11. 独立技术判断与最低成本恢复顺序（仅建议，不执行）

### 应不应该继续当前架构

1. **训练方法作为研究探索可以成立；作为已经具备扩训价值的控制方案，证据不足。**固定完整连接组＋读出是可检验假设；theta的SG联合学习风险明显更高。
2. 为五个任务标签选择已有官方技能，**尚无技术证据证明需要166700神经元**。保留完整脑是本项目科研约束，但不意味着它是此分类任务的工程必要条件。小模型只应作为独立对照，不得冒充完整脑成果。
3. 当前过早进入完整脑联合／内部参数优化。冻结脑的连续交互读出尚未通过基本泛化与闭环验收；应先明确其失败来自标签／输入还是表示本身，再考虑内部优化。
4. 不是“全是工程bug”，也不是“果蝇脑天生训不动”。启动／序列化／预算／SDK问题属工程；已完成运行仍反复泛化失败及有限更新恶化，属数据、目标与优化方法问题。
5. 存在继续训练数小时仍无法验证模型价值的高风险。保持设计不变，最可能再次遇到历史状态敏感、训练指标好但真实错误、内部更新不下降、缓存因编码变化不能复用、最后由资源／时间限制停止。
6. 现有严格身份、状态、审计机制对科研复现有用；但每次新建试验和强绑定迁移、每步全状态扫描带来工程及计算成本。应区分必要的来源完整性与不必要的运行时重复工作，先测量再修改，不删除证据保护来换表面成功。

### 最优先的三个根本问题

| 顺序 | 要解决的根本问题／模块 | 几分钟内低成本验证设计 | 进入下一阶段的条件 |
|---|---|---|---|
| 1 | **冻结目标、标签、能力、状态契约。**`control/x2_interaction.py`、`learning/x2_interaction_data.py`、behavior teacher/dataset、`senses/x2_behavior.py`／relative_heading、session | 只用已存JSON／数组审计真线上观测与教师输出；列出任务×阶段×standing×能力×前一任务覆盖；用原始工程输入拟合简单线性／小MLP对照，完整会话划分，分钟级预算 | 五任务语义一致；GREET真实WAVE还是无能力回退已明确；END/CONTINUE/强度标签覆盖；简单对照能在未参与修改的会话验证正确，否则停止讨论完整脑扩训 |
| 2 | **证明冻结脑表示有可泛化增益，再谈内部训练。**behavior/interaction policy、readout、评估器；theta试验保持冻结 | 先复用兼容缓存做原始输入对照、DN-only、任务mask／历史分组比较；统计混淆、完整请求、误动作、停止延迟。必要时仅预登记2–4个完整脑帧的因果重放，≤180s，不扫参数 | 冻结脑读出在明确新会话上稳定通过；若简单输入基线好而DN失败，优先改表示研究设计；没有基线胜出证据不启动长联合训练 |
| 3 | **建立可计时、可恢复、有门槛的执行协议。**`learning/x2_behavior_training.py`、`learning/x2_behavior_cache.py`、`learning/resource_guard.py`、policy／checkpoint；控制profile独立 | 用现存小数组／短fixture检查保存→恢复Adam状态、数据游标、RNG、完整state与连续执行等价；加分段计时；现存少量MPS步做带/不带重复校验的定向profile，预算≤120s | 同身份恢复正确；可估算完整预算并留验证/保存余量；无无界重试；当前任务的实际控制延迟目标已定义 |

第二步提及的standing反事实应只用于诊断原保存帧，不能把人工改standing作为真实反馈送入控制来“修好按钮”。第三步的性能对照必须保留边界finite检查与出错即停，不能先删除所有校验。

### 何时才值得重新正式训练

- 固定一份数据／教师／profile／split版本，明确区分开发验证与新的锁定测试；不再用不同数据集“最好分数”互相替代。
- 简单对照先证明任务与标签可学；冻结脑若作为主线，必须证明同样新会话上完整请求和停止行为合格。对这类确定性小任务，可预先要求小型验收集所有必需组合通过，而不只平均F1提高。
- 完整脑内部更新先在独立、预登记的小训练分支证明实际forward loss下降；新验证完整请求增加，同时FWD、END和GREET误触均不退化。单纯theta有变化／gradient有限不足以放行。
- 有可预测时间预算、恢复点、可靠资源余量；不用“先跑90分钟看结果”代替计算预算。
- 固定官方执行profile已覆盖模型的实际行为／幅值；实际点击验收分开记录按钮输入、模型输出、执行命令、真实位姿／关节、反馈再入脑与当前会话神经遥测。
- 真正需要实时控制时先确定允许延迟；当前8–15秒决策只能考虑低频监督层，不能宣称60Hz脑控制。安全停止继续由独立执行层保证，普通END需单独验收。

### 立即停止条件

建议在协议中预登记，而非训练中临时降低要求：任何loss／梯度／参数／Adam state NaN或Inf；来源／缓存身份不符；固定训练对照更新后loss恶化；关键类退化或单类坍缩；无WAVE时误触动作；END可靠性退化；剩余预算不足以完成验证＋checkpoint＋清理；可用内存低于预留标准；规定开发验证间隔内完整请求无改善。出现上述条件保存失败状态后停止，不自动增加时长、频率、窗口或改阈值继续。

恢复设计应原子写入：参数、optimizer、scheduler（如有）、RNG／编码相位、完整神经state、数据episode/frame游标、累计step和剩余预算、代码／数据／模型身份；阶段完成写可验证manifest。先用小fixture验证“不中断”和“中断恢复”结果一致，再允许大型任务。特征缓存按已完成episode／frame增量封存，避免因最终总结缺失丢掉可用结果；身份变化必须显式新版本，不能原地覆写。

## 12. 交给独立顾问的关键问题与证据缺口

希望外部顾问重点判断：

1. 将抽象任务／机器人反馈编码到R1–R6、再以DN膜电位读出，是否是合理的任务注入／读出设计？如何设计最小对照证明完整连接组带来的价值，而不是只记住历史位置？
2. 当前hard reset／refractory停止梯度、1000步信用窗口、类型共享gain及缩放SG的组合，在这种事件敏感网络中是否有可行优化路径？应优先改估计器、参数化还是先放弃内部联合训练？
3. 24／42样本对2628维特征的ridge设计，怎样在保持完整脑要求的同时建立足够的状态覆盖和真正独立测试？应先冻结哪些输入契约？
4. 500步需要约5.6秒离线／8–15秒线上时，完整脑应放在何种控制时间尺度？是否应将科研表示验证与产品技能调度分离，怎样保证最终控制结论不被执行脚本替代？
5. 在继续投入前，哪些比较最能一锤定音：原始输入小模型、冻结脑读出、任务消融、连接组消融或有限参数扰动？如何设定停止标准避免反复更换小试验却不验证主张？

不能完整还原／尚缺的证据：全局不可变启动账本、逐编辑时间与每次检查的统一产物；部分run的UTC锚；被拒绝预检的精确开始／结束及当刻内存；训练期CPU/GPU利用率、kernel trace、transfer量、swap/pagefault和热状态；optimizer／validation独立timing；standing单因素神经结果；新候选锁定盲测及连续五按钮实际成功；04:14那场第二请求的IPC源归因；完整最终UI回放操作验收。没有这些证据的地方，本报告没有补写推测为事实。

## 附录A：供独立顾问检查的真实核心代码

以下为调查时工作区实际代码的相关片段，路径前缀均为`/Users/yang/project/malecns_lab/`。行号对应当前文件；历史错误栈仍使用其当时行号。为避免把实现细节隐藏在解释里，保留与数据、优化、状态、恢复和开销直接相关的代码；省略处明确说明。

### A1. 人工教师及其记忆依赖

`src/malecns_lab/learning/x2_behavior_data.py:29–65`：

```python
def teacher_step(observation, memory):
    o, f = observation, observation.feedback
    if memory.get("request_id") != o.request_id:
        memory.clear()
        memory.update(request_id=o.request_id, completed=[], rejected=False)
    if f.valid and f.request_id == o.request_id:
        if f.status == S.COMPLETED and f.behavior.value not in memory["completed"]:
            memory["completed"].append(f.behavior.value)
        if f.status in (S.REJECTED, S.FAILED, S.UNKNOWN):
            memory["rejected"] = True
    behavior, phase, intensity = B.HOLD, P.FINISH, .5
    active = f.valid and f.status in (S.ACCEPTED, S.STARTED, S.RUNNING)
    if o.task == T.END_INTERACTION or not f.valid or memory["rejected"]:
        pass
    elif active:
        if f.request_id == o.request_id:
            behavior, phase = f.behavior, P.CONTINUE
    elif f.standing:
        done = memory["completed"]
        if B.WAVE_RIGHT.value not in done:
            primary = {T.FACE_LEFT: B.TURN_LEFT, T.FACE_RIGHT: B.TURN_RIGHT,
                       T.SHOW_FORWARD: B.ADVANCE}.get(o.task)
            aligned = primary in (B.TURN_LEFT, B.TURN_RIGHT) and abs(f.heading_error_rad) <= .05
            behavior = primary if primary is not None and primary.value not in done and not aligned else B.WAVE_RIGHT
            if behavior in o.capabilities:
                phase = P.START
            else:
                behavior = B.HOLD
    return {"behavior": behavior.value, "phase": phase.value, "intensity": intensity}
```

该教师并非纯当前帧函数，依赖每请求completed/rejected历史。encoder排除teacher字段，在线不调用teacher，未发现标签直接泄漏。**简单模型对照也必须获得同等可用历史或递归状态，不能用无记忆单帧基线的失败证明完整脑必要。**当最新standing=False时，教师当前规则应HOLD/FINISH而非立刻前进；实际RIGHT仍不符合规则。

### A2. 读出结构与真实图规模

`src/malecns_lab/learning/x2_behavior_policy.py:81–84,115–125,239`：

```python
graph = load_training_graph(path)
require(graph.identity == source_identity["model"]["graph"]
        and graph.n == 166700 and graph.counts.nnz == 25582938
        and graph.identity["synapse_count"] == 124177617,
        "Full graph/type order differs from migration source")

self.behavior_head = nn.Linear(2628, 5).to("mps")
self.phase_head = nn.Linear(2628, 3).to("mps")
self.parameter_head = nn.Linear(2628, 1).to("mps")

features = torch.cat(sums) / 250
```

三个头仅用神经特征，不存在按按钮硬编码跳过脑的读出捷径。保存的feature身份是`DN mean voltage in consecutive 250-step halves`。这也意味着DN电位变化的可辨识性与历史稳定性是关键，而不仅是输入one-hot是否不同。

### A3. 代理梯度、硬事件、稀疏反传

`src/malecns_lab/brain/trainable.py:62–73`：

```python
class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, distance: Tensor, scale: float) -> Tensor:
        ctx.save_for_backward(distance)
        ctx.scale = scale
        return (distance > 0).to(distance.dtype)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        (distance,) = ctx.saved_tensors
        derivative = 1.0 / (ctx.scale * (1.0 + (distance / ctx.scale).abs()) ** 2)
        return gradient * derivative, None
```

`src/malecns_lab/brain/mps_trainable.py:45–52`中缩放函数的实际返回式：

```python
detached = spikes.detach()
return detached + gain * (spikes - detached)
```

同文件`:182–190`，去掉中间解释性注释后：

```python
spikes = surrogate_spike(v - cfg.v_threshold_mv, cfg.surrogate_scale_mv) * active
if spike_backward_gain != 1.0:
    spikes = scale_spike_backward(spikes, spike_backward_gain)
fired = spikes.detach().bool()
next_v, next_g = torch.where(fired, cfg.v_reset_mv, v), torch.where(fired, 0., g)
refractory = torch.where(fired & ~self.input_mask,
                        state.step + self.refractory_steps, state.refractory_until)
transmitted = fixed_mps_matvec(self.base_weights, self.base_weights_adjoint,
                              self.gains[self.group_index] * spikes)
```

同文件`:29–42`的自定义稀疏乘法：

```python
class _Matvec(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, adjoint, vector):
        ctx.save_for_backward(adjoint)
        return torch.sparse.mm(matrix, vector[:, None])[:, 0]

    @staticmethod
    def backward(ctx, gradient):
        (adjoint,) = ctx.saved_tensors
        return None, None, torch.sparse.mm(adjoint, gradient[:, None])[:, 0]
```

矩阵固定，反向对输入向量使用转置矩阵；没有发现“把W当成Wᵀ”的直接证据。gain缩放作用于每个spike代理导数，不是最终loss全局缩放；不同长度路径受影响不同。reset/refractory事件mask停止梯度，是算法假设。fresh Adam首步中统一梯度缩放会在`g/sqrt(g²)`中大体抵消（epsilon除外），所以L2 clip不是参数实际步幅或行为变化大小的保证；这是算法分析而非额外实测。

### A4. 训练损失及验证循环

`src/malecns_lab/learning/x2_behavior_training.py:441–452`相关主体：

```python
device = output.behavior_logits.device
b = tuple(str(x) for x in BEHAVIORS).index(target["behavior"])
p = tuple(str(x) for x in PHASES).index(target["phase"])
ce_behavior = F.cross_entropy(output.behavior_logits[None], torch.tensor([b], device=device))
ce_phase = F.cross_entropy(output.phase_logits[None], torch.tensor([p], device=device))
active = target["behavior"] in ("ADVANCE", "TURN_LEFT", "TURN_RIGHT") and target["phase"] in ("START", "CONTINUE")
param = (output.intensity[0] - float(target["intensity"]))**2 if active else output.intensity[0]*0
return ce_behavior + ce_phase + .1*param
```

没有真实motion reward、停止距离／延迟损失、类别权重或错误动作代价；独立behavior/phase CE不保证组合合法。mask关闭的parameter梯度为0可由目标设计解释，不能直接叫梯度消失。

同文件`:849–865`：

```python
for index in range(cfg["max_updates"]):
    phase("readout_update", update=index+1)
    episode = train[index % len(train)]
    optimizer.zero_grad(set_to_none=True)
    _, loss = cached_outputs([episode], features)
    update = gradient_update(loss)
    with torch.no_grad():
        predicted, val_loss = cached_outputs(validation, validation_features)
        value = float(val_loss.cpu())
    update["validation_loss"] = value
    history.append(update)
    if value < best:
        best = value
        selected = save_checkpoint(directory / "checkpoints" / f"{index+1:04d}", policy,
            {"registration_id": reg["registration_id"], "optimizer_steps": index+1,
             "selection": "validation_loss", "validation_loss": best}, optimizer=optimizer)
    write_json(directory / "updates.json", {"history": history})
```

同文件`:813–817`：

```python
for frame, feature in zip(episode["frames"], used):
    heads = policy.read_features(feature.to("mps"))
    predictions[episode["episode_id"]].append(_prediction(heads))
    losses.append(loss_for(BehaviorOutput(*heads, feature, None), frame["target"]))
return predictions, torch.stack(losses).mean()
```

逐episode轮换、按validation loss选最佳、逐帧CPU/MPS往返都在这里可见。无法从此循环推出每次验证重算全脑。

### A5. 状态校验和缓存保存成本

`src/malecns_lab/brain/mps_trainable.py:149–170`节选：

```python
for value in (state.v, state.g, *state.pending):
    if (value.shape != (self.graph.n,) or value.dtype != torch.float32 or value.device.type != "mps"
            or not torch.isfinite(value).all()):
        raise ValueError("State must be finite MPS float32 with full node shape")

def step(self, voltage_increment, state, *, spike_backward_gain=1.0):
    self._validate_state(state)
```

`src/malecns_lab/learning/x2_behavior_training.py:795–805`：

```python
with torch.no_grad():
    state = policy.initial_state(episode["frames"][0]["observation"]["session_id"])
    before_id = _save_state(folder, "state-0000", policy, state)
    features[episode["episode_id"]], predictions[episode["episode_id"]] = [], []
    for index, frame in enumerate(episode["frames"]):
        observation = Observation.from_dict(frame["observation"])
        output = policy(observation, state, checkpoint_blocks=False)
        predictions[episode["episode_id"]].append(_save_output(folder, index, observation, output, policy, before_id, parameters_id))
        features[episode["episode_id"]].append(output.features_mv.detach().cpu())
        state = output.state
        before_id = _save_state(folder, f"state-{index+1:04d}", policy, state)
```

`_save_state`在`:594–603`压缩完整state并计算hash。policy`:230–231`反向内存checkpoint实际调用：

```python
values = (checkpoint(block, *args, use_reentrant=False, preserve_rng_state=False)
          if checkpoint_blocks and torch.is_grad_enabled() else block(*args))
```

### A6. 保存了Adam，但恢复入口只还原参数

`src/malecns_lab/learning/x2_behavior_training.py:402–408`：

```python
values = {f"parameter.{name}": p.detach().cpu().numpy().copy() for name, p in policy.named_parameters()}
if optimizer is not None:
    for name, p in policy.named_parameters():
        for key, value in optimizer.state.get(p, {}).items():
            values[f"optimizer.{name}.{key}"] = value.detach().cpu().numpy().copy()
with (directory / "arrays.npz").open("xb") as stream:
    np.savez_compressed(stream, **values)
```

同文件`:417–435`，中间完整性检查省略，仅摘录相关原句：

```python
def restore_checkpoint(directory, policy):
    """Strict non-pickle restoration. Training starts a fresh optimizer."""
```

```python
with torch.no_grad():
    for name, p in policy.named_parameters():
        value = archive[f"parameter.{name}"]
        frozen.require(value.dtype == np.float32 and value.shape == tuple(p.shape), "Parameter shape/dtype differs")
        p.copy_(torch.from_numpy(value.copy()).to(p.device))
```

### A7. standing与启动预算为何不能从瞬时快照判断

`src/malecns_lab/control/x2_skills.py:468–487`相关条件：

```python
if (not m or not o or not 0 <= now-m.received <= .5 or not 0 <= now-o.received <= .25
        or m.source not in ("", SOURCE) or m.mode != "STAND_DEFAULT" or m.body_status != 1 or m.action_status != 100
        or o.tilt > .5 or math.hypot(*o.linear[:2]) >= .02 or abs(o.angular[2]) >= .03
        or self.player is None or self.player[1] != 0):
    self._idle_since = self._idle_pose = None
    return False
```

后续以新传感器序列和位姿漂移更新窗口，最后实际返回式是：

```python
return stamp-self._idle_since >= 1.
```

`data/x2_behavior_standing_causal_20261024_r2/run.py:186–187`：

```python
budget = resource_budget(psutil.virtual_memory().available, reserve_bytes=frozen.GIB)
frozen.require(budget["effective_rss_limit_bytes"] >= 2 * frozen.GIB,
               "Insufficient safe brain budget")
```

该脚本`:207–209`的修正仅针对JSON表示等价：

```python
def serialized_equal(left, right):
    """Exact JSON field/value equality across tuple/list and string-enum round trips."""
    return frozen.canonical_hash(left) == frozen.canonical_hash(right)
```

## 附录B：数据分布与现有离线读出诊断

下表由保存的dataset及selected_episodes统计，不是新训练：

| 数据 | 帧数 | 行为标签计数 | 阶段计数 |
|---|---:|---|---|
| 原readout选中train50集 | 416 | HOLD180、WAVE137、ADVANCE35、LEFT32、RIGHT32 | FINISH180、CONTINUE178、START58 |
| 原readout validation50集 | 402 | HOLD175、WAVE133、ADVANCE34、LEFT30、RIGHT30 | FINISH175、CONTINUE169、START58 |
| continuous train2会话 | 24 | HOLD14、ADVANCE6、LEFT2、RIGHT2 | FINISH14、START10、CONTINUE0 |
| continuous validation1会话 | 10 | HOLD6、ADVANCE2、LEFT1、RIGHT1 | FINISH6、START4、CONTINUE0 |
| paired validation1会话 | 12 | HOLD7、ADVANCE3、LEFT1、RIGHT1 | FINISH7、START5、CONTINUE0 |

所有这些行的强度标签都是.5，perception.valid都是False。paired另加18个同前态分支，与原24行合计42行拟合。

`short_trial_20261023_r2/linear-diagnostic.json`在416/402缓存帧上的固定ridge α1诊断：

| 特征／预测项 | train accuracy | validation accuracy | validation首帧accuracy |
|---|---:|---:|---:|
| raw behavior | .99519 | .27363 | .62 |
| normalized behavior | .99279 | .36070 | .76 |
| raw phase | .99038 | .40547 | .88 |
| normalized phase | .98558 | .43035 | .88 |
| raw task | .99038 | .22637 | .46 |
| normalized task | .98798 | .26617 | .50 |

因此有充分依据优先调查跨历史／跨情境表示与数据覆盖，而不是把“输入编码不同”直接当作“输出DN稳定可分”。训练样本少于特征维数，近99%训练拟合本身不是强证据。

**本报告没有作出“已经修好”结论，也没有启动下一轮实验。当前可交付的是可复核的失败诊断、仍然保留的科研资产和有停止门槛的恢复建议。**
