# 文档索引

从[项目首页](../README.md)开始。当前交付方向为 **官方 X2 仿真闭环 → 上海现场设备适配**；Mac Docker 已实测官方站立、前进和停止，独立验收已通过；速度映射与神经闭环待验证。仿真按需启动，当前已关闭。

## 当前入口

- [神经活动图谱](NEURAL_ATLAS.md)：查看记录、设置刺激并运行完整脑响应。
- [X2 官方接入](AGIBOT.md)：官方组件、Mac Docker 尝试范围和现场对齐事项。
- [架构与动作接口](ARCHITECTURE.md)：模块职责、坐标单位和旧环境迁移。
- [已有实验结果](EXPERIMENT_RESULTS.md)：完整图实测、训练结果及已停止的单轴稳向实验。

## 数据与模型

- [研究与来源](RESEARCH.md)、[完整参考图设计](MODEL_V2.md)、[历史过滤审计](MALECNS_AUDIT.md)。
- [录像与标准刺激诊断](MODEL_V2_USAGE.md)、[配对信号诊断](OBSERVABILITY.md)。
- [可训练 SNN](TRAINABLE_SNN.md)、[神经输入输出映射](NEURAL_INTERFACE.md)、[有状态策略](NEURAL_POLICY.md)。

## 保留的研究工具

- [GPU 训练](MPS_TRAINING.md)、[联合更新探针](JOINT_PROBE.md)、[连续训练](TRAINING_RUNNER.md)、[参数与状态存档](CHECKPOINTS.md)。
- [目标跟随环境](FOLLOW_TASK.md)、[训练数据](NEURAL_TRAINING_DATA.md)、[闭环评估](EVALUATION_SNN.md)。
- [冻结脑读出](LEARNING_V2.md)、[拓扑研究对照](EXPERIMENT_V2.md)、[重连生成](BOUNDED_REWIRING.md)、[研究策略准入](RESEARCH_POLICY.md)。
- [Tello 实验](TELLO.md)：已有地面采集与受限试飞；不代表新模型或 X2 控制成果。

研究工具不是 X2 接入的前置任务；不因文档保留而自动续训。历史记录中的路径、命令和阶段状态按当时源码解释，当前操作以本索引为准。

[阶段设计与实测报告](../specs/README.md) · [来源与许可](../THIRD_PARTY.md)
