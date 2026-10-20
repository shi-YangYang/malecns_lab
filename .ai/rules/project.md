# MaleCNS Lab 项目规则

本文件补充 [AGENTS.md](../../AGENTS.md) 的通用规范，不建立重复的治理目录。

## 工程与资产

- 业务包和 CLI 为 `malecns_lab`，源码位于 `src/malecns_lab/`；保留 `tests/`、`docs/`、`configs/`、`examples/` 的现有职责。
- 当前命名由用户在 [Decision 016](../decisions/016-malecns-lab-repository.md) 明确授权；旧 Spec/Decision 中的 cnsbridge 路径是历史记录，不再作为运行入口。
- 用户已在 [Spec 002](../../specs/spec-002-business-layout/spec.md) 授权业务目录调整，并在 [Spec 003](../../specs/spec-003-remove-compat/spec.md) 授权移除旧兼容入口：规范实现归入 `control/`、`learning/`、`experiments/tello/`，直接导入真实模块；实验与学习命令统一使用 `python -m`。
- 新业务代码直接依赖真实实现包，`learning` 可以使用 `control` 动作契约，`control` 不反向依赖 `learning`。已有 `brain/`、`senses/` 保留其模型职责；具体边界见 [Decision 002](../decisions/002-business-layout.md)。
- `diagnostics/` 负责离线模拟神经活动观察，刺激标注与期望动作标签独立；不实例化动作读出或设备控制，见 [Decision 004](../decisions/004-neural-activity.md)。
- `data/`、`logs/`、模型、录像和运行环境是本地资产，不因清理源码而删除，也不提交进 Git。需要删除或替换时依用户明确授权执行，既有授权不重复索要。
- 保留 [LICENSE](../../LICENSE) 和 [THIRD_PARTY.md](../../THIRD_PARTY.md) 的上游版权、数据来源与许可说明。
- `cnsbridge.*` 和 `flydrones.*` 序列化标识是已有数据契约，不能作为旧品牌残留批量替换；它们不提供旧 Python 包兼容。变更模式、来源身份、特征顺序或格式版本须在对应 Spec 中明确兼容策略。
- 包改名后的源码摘要与旧运行不同；不得修改旧 run、checkpoint 或来源快照，也不得放松 hash 校验以伪装续跑。原实验恢复使用对应 commit/环境，参数迁移需要新的显式协议。

## 动作与研究契约

- `BodyVelocity` 轴顺序为 `(vx, vy, vz, wx, wy, wz)`，采用 FLU：前、左、上；平移 m/s，转动 rad/s，正 `wz` 左转。这是期望速度，不是测量结果。
- 既有 `FlightCommand` 横移和偏航正值向右；符号转换属于适配器。不可把归一化 Tello RC 直接称为精确 m/s 速度。
- 冻结连接组后的 RidgeMotionDecoder 只训练读出。旧 `MotorDecoder` 与试飞监督含人工规则；旧实飞结果不能充当新读出有效性的证据。
- 仅使用真实连接组资产运行脑模型；禁止合成飞行网络、别名或缺失时回退，见 [Decision 005](../decisions/005-remove-minifly.md)。真实 MaleCNS 图的过滤、群组、版本和时间窗口必须可追踪。仿真时间、实际处理耗时、控制发包频率应区分。
- 第二版以完整官方数据为母本，先核实完整神经元参考图并保留全部已发布内部非零连接，不默认剪弱边、自连接或因注释缺失删节点。阈值 3 等简化只有在完整基线与受控实验支持后才可采用；没有实测不得声称实测，资源不足不得静默回退。原始片段与完整神经元分开对账，详见 [Decision 007](../decisions/007-full-dataset-baseline.md)。
- 训练/评估按完整场景隔离，不能随机拆相邻帧制造泄漏。期望动作来自独立标注/任务规则，旧 RC 和实测速度不自动等价于标签。
- 新读出的实体效果、可靠避障、连接组优势和跨设备迁移均需独立证据，不宣称已完成或全球首次。细节见[研究文档](../../docs/RESEARCH.md)。

## 验证与硬件

遵循 AGENTS 第 22 节及用户既有停止规则：先定 S0–S3，只执行与具体风险相称的检查，通过后停止；不为实施覆盖新增无价值测试，不扩大范围修复无关问题。

离线验证不自动授权连接设备、采集或实飞。任何实际硬件工作应在用户已明确授权的任务范围内，按 [Tello 文档](../../docs/TELLO.md)和对应 Spec 执行；不能从自动化测试启动运动，也不能因读取实验日志推断落地状态。
