# MaleCNS Lab 文档索引

当前 Python 包和 CLI 为 `malecns_lab`。新用户从[项目首页](../README.md)和[架构与动作契约](ARCHITECTURE.md)开始；旧环境先读[命名迁移](MIGRATION.md)。

## 数据与神经活动

- [研究与数据](RESEARCH.md)：数据来源、建模假设、结论边界与对照。
- [V2 完整参考模型](MODEL_V2.md)：完整发布数据的边界、参考图和第二版设计。
- [V2 重放与信号报告](MODEL_V2_USAGE.md)：真实录像、标准刺激、记录和静态图表。
- [配对信号诊断](OBSERVABILITY.md)：同一状态下比较刺激与空白响应。
- [早期数据与通路审计](MALECNS_AUDIT.md)：历史过滤图、旧输入输出选择和审计时的证据。保留当时名称及代码定位，不作为当前安装命令。

## 网络学习与验证

- [可训练 SNN](TRAINABLE_SNN.md)、[神经输入输出映射](NEURAL_INTERFACE.md)、[有状态策略](NEURAL_POLICY.md)：方程、可训练参数、感觉映射与读出。
- [目标跟随环境](FOLLOW_TASK.md)、[训练数据](NEURAL_TRAINING_DATA.md)：现有仿真实验与教师序列，未代表真实设备泛化。
- [GPU 训练](MPS_TRAINING.md)：当前全 GPU 分支及旧 MPS 分支的区别。
- [联合更新探针](JOINT_PROBE.md)、[连续训练](TRAINING_RUNNER.md)：已登记的资源探针与训练窗口协议。
- [策略参数](POLICY_CHECKPOINT.md)、[训练状态](TRAINING_CHECKPOINT.md)：部署参数与可恢复训练存档的区别。
- [闭环评估](EVALUATION_SNN.md)：来源准入和独立测试；不能将训练过程指标代替最终策略成绩。
- [重连生成](BOUNDED_REWIRING.md)、[研究策略准入](RESEARCH_POLICY.md)：拓扑对照资产与严格身份验证。
- [V2 冻结脑读出](LEARNING_V2.md)、[V2 研究对照](EXPERIMENT_V2.md)：保留的冻结脑对照流程。

## 设备与后续方向

- [Tello 实验](TELLO.md)：连接、录像、旧读出试飞及实体证据边界。
- [智元 G2／X2 资料](AGIBOT.md)：SDK 和仿真文档入口；当前尚无对应机器人适配器。
- WebGL 神经活动图谱、三维果蝇和跨设备实验是后续方向，当前没有前端应用或已验证的通用控制能力。

## 项目与历史证据

- [命名迁移](MIGRATION.md)：新入口、editable 安装、持久化格式和旧实验恢复限制。
- [路线与证据](../constitution/roadmap.md)、[Spec 索引](../specs/README.md)、[任务交接](../.ai/prompts/task-handoff.md)：阶段记录与验收。
- [第三方来源](../THIRD_PARTY.md)：代码、数据及研究方法出处。

旧 Spec、Decision 和已完成的实验报告记录当时事实，保留历史名称、路径、命令及结果。其命令不保证适用于改名后的 checkout；使用当前操作文档，精确复现旧实验则回到报告对应提交。
