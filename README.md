# MaleCNS Lab

基于完整 MaleCNS 果蝇连接组的神经计算与具身控制实验平台。Python 包、分发包和命令行入口统一为 `malecns_lab`。

我们研究真实连接拓扑如何参与学习，并让神经活动、训练参数、动作输出和设备行为能够分别记录与验证。目标跟随是现有实验；当前推进神经活动三维展示，随后验证机器人仿真闭环和现场设备。

```text
MaleCNS 官方数据 → 连接组约束的神经模型 → 活动记录与分析
                         ↓
                  可学习的动作读出 → 机身运动目标 → 设备适配
```

## 当前能力与证据

| 部分 | 已实现 | 尚未验证或尚未实现 |
|---|---|---|
| 数据与神经计算 | 完整神经元参考图、LIF 仿真、固定拓扑下的类型共享增益训练 | 完整生物还原、真实连接拓扑的任务优势 |
| 活动观察 | 离线录像与标准刺激重放、神经活动记录、静态图表、配对刺激诊断；新增本地 WebGL 胞体图谱与群组活动回放 | 最新训练策略实时可视化、逐神经元完整动态、果蝇身体物理仿真 |
| 模型学习 | 冻结脑读出对照、内部参数与读出联合训练、完整 GPU 训练分支 | 固定最终策略的闭环成绩、可靠实时控制 |
| 设备 | Tello 连接、录像与旧人工读出的受限实飞工具；统一动作与底盘数值转换 | 新训练策略实飞、实体底盘和智元 G2／X2 接入 |

完整参考图包含 **166,700 个节点、25,582,938 条内部有向连接、124,177,617 个突触计数**。这里的“完整”指官方 v1.0 minconf 0.5 发布数据中、作者定义的神经元全集之间全部非零连接；保留原始自连接和孤立节点，不等于完整果蝇生物学。来源与边界见 [V2 数据设计](docs/MODEL_V2.md)和 [P0 构建报告](specs/spec-007-model-v2/implementation-p0.md)。

已完成标准视觉刺激与[持续输入的完整图诊断](specs/spec-008-trainable-snn/sustained-observability-run-20261019-r1.md)，证据支持刺激引起不同模拟神经响应。新的[全 GPU 训练分支已完成 160 窗](specs/spec-008-trainable-snn/gpu-recovery-run-20261019.md)：训练轨迹中的前进正向饱和由 230 降至 18 帧，负向目标方向正确由 0/121 增至 81/121，但仍有方向错误与超调。这些是训练过程指标，不是固定最终策略的独立成绩。闭环评估、拓扑对照和新模型实飞尚未完成。

脑模型缺少真实连接组资产时会报错，不提供合成网络回退。旧 Tello 实飞使用人工设计读出与监督逻辑，不能充当新模型学会控制或避障的证据。

## 安装与入口

需要 Python 3.11 或更新版本。在源码目录中：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[tello,data,analysis,dev]'
malecns_lab --help
python -m malecns_lab --version
```

基础离线依赖可使用 `python -m pip install -e .`。独立训练环境和依赖见 [可训练 SNN](docs/TRAINABLE_SNN.md)与 [GPU 训练](docs/MPS_TRAINING.md)。完整数据、模型和运行结果保存在本地 `data/`、`logs/`，不随源码仓库分发；新 checkout 不会自动拥有这些资产。

从旧 `cnsbridge` 环境更新或迁移项目根目录时请先阅读[迁移说明](docs/MIGRATION.md)：旧导入和 CLI 已移除，两个既有虚拟环境中的绝对路径需要修复，并更新 editable 安装。改名不改变持久化格式，但源码身份会改变，**不能直接在新代码下续跑旧 hash-bound 实验**。

先查看安全的离线入口，按文档选择已有资产与新输出目录：

```bash
python -m malecns_lab.diagnostics.replay_v2 --help
python -m malecns_lab.diagnostics.report_v2 --help
python -m malecns_lab.learning.train_gpu --help
```

实际完整脑重放、训练和设备动作分别按已登记协议执行。`malecns_lab build-brain` 保留给旧设备实验构建过滤图，不能用于覆盖 V2 完整参考资产。

## 本地三维脑图

已有完整参考、官方注释及标准刺激诊断记录时，可导出并打开本地页面：

```bash
.venv/bin/python -m malecns_lab.visualization.export
cd apps/atlas
npm ci
npm run dev
```

访问 `http://127.0.0.1:8765`，切换刺激、播放/暂停、拖动时间轴、旋转脑图并查看群组放电率。第一版是已有固定 LIF 诊断记录的回放，三维果蝇为形态示意；未接入动作控制。数据覆盖、显示含义和启动条件见 [神经活动图谱](docs/NEURAL_ATLAS.md)。

## 代码与文档

| 目录 | 职责 |
|---|---|
| `src/malecns_lab/brain/` | 连接组加载、构建与神经计算 |
| `src/malecns_lab/senses/` | 感觉特征和神经输入编码 |
| `src/malecns_lab/diagnostics/` | 神经活动记录、重放与分析 |
| `src/malecns_lab/visualization/` | 已有神经记录与官方空间坐标的只读导出 |
| `apps/atlas/` | 本地 Three.js/WebGL 脑图与刺激回放页面 |
| `src/malecns_lab/learning/` | 读出、内部学习、实验协议与评估 |
| `src/malecns_lab/control/` | 动作契约、数值适配、旧读出与限制 |
| `src/malecns_lab/drones/` | 现有目标跟随仿真环境 |
| `src/malecns_lab/experiments/tello/` | 已有 Tello 地面和受限实飞实验 |
| `configs/`、`tests/`、`examples/` | 配置、检查和离线工具示例 |

从[文档索引](docs/README.md)进入具体流程；[架构](docs/ARCHITECTURE.md)说明模块与动作坐标，[研究边界](docs/RESEARCH.md)说明建模假设与对照，[智元资料](docs/AGIBOT.md)记录 G2／X2 的开发入口与待确认条件。

开发遵循 [AGENTS.md](AGENTS.md)、[使命与技术约束](constitution/)和 [Spec](specs/README.md)。当前工作记录见[任务交接](.ai/prompts/task-handoff.md)。当前本地根目录为 `/Users/yang/project/malecns_lab`，用户已同步修改 Codex 项目设置。路径定位不依赖启动 cwd；设备实验要求源码 checkout 和 editable 安装。

## 来源与许可

本项目基于 [SpikeCalls/FlyDrones](https://github.com/SpikeCalls/FlyDrones) 的代码发展。保留 [MIT 许可](LICENSE)、上游版权及历史实验记录；改名不改变来源。MaleCNS 数据采用独立许可，详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
