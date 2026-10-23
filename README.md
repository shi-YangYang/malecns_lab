# MaleCNS Lab

基于完整 MaleCNS 果蝇连接组的神经计算与具身控制实验平台，Python 包和 CLI 为 `malecns_lab`。当前设备方向为智元 X2：交互和身体反馈进入脑，脑产生行为目标，官方 AimDK/MC 执行，Electron 展示仿真与同次脑活动。

完整参考图含 166,700 个神经元、25,582,938 条内部有向连接、124,177,617 个突触计数；指官方发布的完整神经元参考，不等于完整生物还原。缺连接组资产时报错，不用合成脑回退。

当前 GPU 仿真及旧纵向反馈已有实测；多行为任务按钮仍存在无动作问题，训练与实测暂停。状态以[当前交接](.ai/prompts/task-handoff.md)为准，已做过什么、失败在哪里统一见 [history.md](history.md)。

## 安装与运行

需要 Python >=3.11，使用源码 checkout 和 editable 安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[tello,data,analysis,dev]'
malecns_lab --help
```

基础离线环境可用 `python -m pip install -e .`；完整脑与 GPU 学习环境见[计算模型](docs/TRAINABLE_SNN.md)和[训练与存档](docs/training.md)。数据、模型、运行环境、日志和媒体在本地 `data/`、`logs/`，不随源码分发。

已有构建的桌面可用：

```bash
cd apps/desktop
npm start
```

首次构建、候选配置与回放见[桌面使用说明](apps/desktop/README.md)。打开应用不会自动启动完整脑或仿真；GPU 环境由[独立部署入口](deploy/x2/README.md)维护，不用时关闭。网页脑图和刺激响应使用[图谱指南](docs/NEURAL_ATLAS.md)。

## 文档入口

| 需要了解的内容 | 唯一入口 |
|---|---|
| 使命与长期方向 | [项目使命](constitution/mission.md) |
| 技术栈、模块职责与动作坐标 | [架构](docs/architecture/overview.md) |
| 数据来源、完整图与旧过滤图 | [数据与模型](docs/data-model.md) |
| LIF 数值、输入编码与神经策略 | [神经计算](docs/TRAINABLE_SNN.md) |
| 训练、GPU 路线、checkpoint 和恢复 | [训练与存档](docs/training.md) |
| 录像诊断、读出对照、拓扑与闭环评估 | [离线研究工具](docs/offline-research.md) |
| 科学主张与限制、论文来源 | [研究边界](docs/RESEARCH.md) |
| 旧 Tello 设备工具 | [Tello](docs/TELLO.md) |
| 历史阶段结果与关键证据 | [history.md](history.md) |

开发遵循 [AGENTS.md](AGENTS.md)：每个任务最多一份 Markdown，原地更新，只保留必要决策、最终结果、关键证据与未解决问题。目录职责以架构文档为准，不再保留旧 Spec 过程文件。

## 来源与许可

项目基于 [SpikeCalls/FlyDrones](https://github.com/SpikeCalls/FlyDrones) 发展，保留 [MIT 许可](LICENSE)和上游版权。MaleCNS 数据另有许可，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
