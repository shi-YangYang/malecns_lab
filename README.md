# CNSBridge

基于完整 MaleCNS 连接组的神经控制研究框架。

项目目标是在完整果蝇 MaleCNS 连接图约束下训练网络本身，学习无人机目标跟随。当前可用的 V2 冻结脑与读出工具保留为研究对照，网络内部训练按 [Spec 008](specs/spec-008-trainable-snn/spec.md) 实施，尚不代表已经学会控制。设备侧以 Tello 为实验平台，为差速移动底盘保留统一动作接口。

当前冻结脑对照链路：

```text
摄像头 → 视觉特征 → 冻结的连接组网络 → 可训练解码器 → 机身运动目标
                                                       ├─ 无人机适配器
                                                       └─ 差速底盘适配器
```

## 当前状态

- 已有 Tello 地面连接、视频、神经输出观察与分阶段短时实飞工具。既有实飞使用人工设计的读出与监督逻辑。
- 第二版使用已核实的完整官方神经元参考图：166,700 个节点、25,582,938 条内部有向连接、124,177,617 个突触计数，保留原始自连接和孤立节点，不采用旧阈值 3。数据边界和构建证据见 [V2 设计](docs/MODEL_V2.md)及 [P0 实施报告](specs/spec-007-model-v2/implementation-p0.md)。
- 新的[可训练SNN核心](docs/TRAINABLE_SNN.md)与[配对信号诊断工具](docs/OBSERVABILITY.md)已实现，使用独立训练环境；支持完整输入/输出集合及显式可选视觉背景假设。2026-10-19完整图一次内部增益更新、参数恢复与checkpoint重载通过；缓存伴随与共享SciPy前向后，[同协议实测](specs/spec-008-trainable-snn/forward-csr-run-20261019.md)总耗时由362.64秒经83.69秒降至32.39秒，采样峰值约1.753GiB，最新对照的全部已保存响应、梯度及参数精确相同。这是20ms短窗的数值优化，单次成本比较不等于实时性保证，神经策略学习与新模型实飞仍待证据。
- [完整图两种动力学的配对诊断](specs/spec-008-trainable-snn/observability-run-20261019-r1.md)已完成：固定视觉背景下，位置与大小刺激引起DN电位差异；超过容差的响应均在撤除后出现，各条件逐细胞总发放数相同。结果支持该假设下有限窗口的信号可观测性，不等于动作学习或生物还原。
- 已有共享运动表示、岭回归解码器、设备适配器，以及地面录像、人工标注、离线特征提取和按场景评估的流程。V2 的诊断与学习接口独立版本化，不能混用旧特征缓存。
- 已完成完整参考图的六类标准视觉刺激诊断：编码输入和输出神经活动随刺激变化；有静默及左右不对称，不能自动解释为正确动作。[实测记录与信号图](docs/MODEL_V2_USAGE.md#7-已完成的完整图诊断)同时给出覆盖、耗时和限制。
- 新解码器尚未完成真实数据上的有效性验证，也尚未接入实飞。差速底盘适配器只有离线数学转换，尚未连接实体底盘。
- 所有脑处理入口使用真实 MaleCNS 数据构建的连接组模型；缺少模型时直接报错，不切换到替代网络。这里运行的是连接组驱动的神经仿真，不是活体果蝇大脑。
- 尚未证明可靠避障、连接组优于直接视觉控制，或跨设备迁移。真实录像下的神经响应仍需实际采集和分析。

## 安装

需要 Python 3.11 或更新版本。在项目目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[tello,data,analysis,dev]'
```

项目名称、Python 包和命令行入口为 `CNSBridge` / `cnsbridge`。本地文件夹仍可叫 `FlyDrones`，实验与学习工具通过 `python -m` 调用。更新后需重新执行上面的安装命令。

## 先观察神经信号

第二版入口使用 `data/malecns_v2/full_reference.npz`。当前工作区已经从完整官方 v1.0 数据构建并核对该资产，直接使用即可。完整指已发布 minconf 0.5 数据下、作者定义的神经元全集之间全部非零连接；片段、胶质记录及发布边界另有账本，不冒充额外完整神经元。

还没有录像时，可以先用明确标为人工生成的标准视觉刺激驱动完整真实网络，观察信号：

```bash
python -m cnsbridge.diagnostics.replay_v2 \
  --stimulus static --observation-seconds 0.2 --repeat 2 \
  --budget-seconds 300 --out data/neural_v2/static_01
```

每段额外包含固定 2 秒预热。输出使用新的目录，已有目录会被拒绝；命令不连接设备、不训练动作，也不将标准刺激当成真实飞行录像。详细入口和产物见 [V2 使用说明](docs/MODEL_V2_USAGE.md)。

后续把 Tello 固定在地面、连接其 Wi-Fi 后录制：

```bash
python -m cnsbridge.experiments.tello.record --scene-id room_a --seconds 30
```

录制工具不起飞、不发送运动指令。采集完成后可断开 Tello，再重放；将 `SESSION` 替换为实际会话目录，同源素材使用同一 `family-id`：

```bash
python -m cnsbridge.diagnostics.replay_v2 \
  --recording data/ground_recordings/SESSION --family-id room_a_session_01 \
  --out data/neural_v2/recording_01
```

信号诊断无需填写动作标签。固定相机前的物体移动是视觉刺激，不能自动变成无人机动作目标。确认信号后，再独立标注期望动作；真实训练与评估需要采集数据，当前未据此证明动作效果。

V1 重复离线诊断和读出入口已移除；信号与读出对照统一使用 V2。`cnsbridge build-brain` 暂供旧设备实验构建过滤图，不属于完整参考实验，不用于重建或覆盖 V2 资产。地面采集、会话格式和时间说明见 [Tello 录像协议](docs/TELLO.md#地面录像与会话格式)。

## 项目结构

| 路径 | 用途 |
|---|---|
| `src/cnsbridge/brain/` | 真实连接组加载、构建与 LIF 仿真 |
| `src/cnsbridge/senses/` | 图像运动特征与神经输入编码 |
| `src/cnsbridge/diagnostics/` | 离线神经活动记录、刺激比较与静态图表 |
| `src/cnsbridge/control/` | 运动目标、设备数值适配、旧读出、校准与控制限制 |
| `src/cnsbridge/learning/` | 可训练解码器、数据准备、训练与按场景评估 |
| `src/cnsbridge/experiments/tello/` | 地面采集、观察、受限试飞及会话清理 |
| `src/cnsbridge/drones/` | 离线仿真辅助 |
| `configs/` | Tello 和 MaleCNS 配置 |
| `tests/` | 对应模块的检查 |
| `data/`、`logs/` | 本地数据、模型与实验记录，不作为源码提交 |

代码从 `cnsbridge.control`、`cnsbridge.learning` 和 `cnsbridge.experiments.tello` 导入。实验与学习入口直接运行包内模块，例如 `python -m cnsbridge.learning.train_v2 --help`。源码工作区的 `configs/`、`data/`、`logs/` 路径不随启动目录变化，设备实验仍使用当前 checkout 的 editable 安装。

保留的命令行工具为 `download`、`build-brain`、`inspect`、`calibrate`、`bench`。`calibrate` 服务于旧读出基线；V2 掩码读出使用 `python -m cnsbridge.learning.train_v2`，作为冻结脑对照。

## 文档

- [架构与接口](docs/ARCHITECTURE.md)：模块、动作坐标和扩展方式。
- [可训练SNN](docs/TRAINABLE_SNN.md)：内部参数、数值契约、独立环境、实际更新探针与可选背景变体。
- [有状态神经策略](docs/NEURAL_POLICY.md)：完整网络到全部DN三轴读出的可微接口、跨帧状态和场景重置；软件已验收，实际任务学习尚未完成。
- [配对信号诊断](docs/OBSERVABILITY.md)：新SNN的空白/位置/尺度刺激、完整状态分支、产物及独立校验；首轮及前向优化后的实际完整图对照已完成，科学边界见报告。
- [V2 完整脑诊断](docs/MODEL_V2_USAGE.md)：完整图、真实录像和标准刺激重放、信号图与实测结果。
- [V2 掩码学习](docs/LEARNING_V2.md)：独立动作标签、场景族分区、训练和严格配对评估。
- [V2 研究对照](docs/EXPERIMENT_V2.md)：完整图派生 R/W/P、协议登记、独立拟合和失败记录。
- [Tello 实验](docs/TELLO.md)：地面检查、录像会话协议、已有实飞工具及结果边界。
- [研究与数据](docs/RESEARCH.md)：建模假设、连接组构建、对照实验与引用。

## 开发流程

开发工作遵循 [AGENTS.md](AGENTS.md)。项目目标与技术约束见 [constitution](constitution/)，任务规格与验收记录见 [specs](specs/README.md)，当前进展见[任务交接](.ai/prompts/task-handoff.md)。

## 来源与许可

本项目基于 [SpikeCalls/FlyDrones](https://github.com/SpikeCalls/FlyDrones) 的代码发展，保留来源与历史实验，当前研究方向为完整连接组内部学习和可复现的控制验证。原代码采用 [MIT 许可](LICENSE)，原版权声明保留。MaleCNS 数据采用独立许可；来源和引用说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
