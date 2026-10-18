# Acceptance — P2e 有状态三轴神经策略接口

## Result

**PASS**（软件接口范围）。本验收由未参与实施的新 Agent 完成；不代表完整图任务训练、闭环成绩或设备效果通过。提交与 Spec 状态关闭由协调者在读取本报告后完成。

## Spec Coverage

- **A — 路径、来源与全部 DN：通过。** 阅读生产工厂到 `load_training_graph`、`load_policy_neuron_map` 的调用链，确认复用完整参考验证，数值入口保留 `numerical_test_only`，无裁图或缺失回退。选择数组与身份绑定并要求升序 body ID；全部 DN 按前半窗、后半窗顺序参与读出。输入只经 `target_mask`、注册 R1–R6 事件和神经动力学，未见检测误差、距离、教师或真值到动作的直接通路。
- **B — 状态与时间：通过。** 显式配置计算偶数神经步，控制时钟独立；构造不推进动力学。预处理只在显式 `initial_state` 中以空白 mask、`no_grad` 执行。每次前向从完整 LIF 状态和已导出的驱动相位恢复局部驱动，脑/驱动/frame 三者步号一致；新状态保留队列、不应期、相位和累计事件。`detach` 只切断 LIF 梯度；失败路径未原位改变传入状态。
- **B — 特征与梯度：通过。** 读出使用每神经步 post-reset 电位的两半窗均值，固定物理归一化、单层线性、tanh 和明确的 SI 轴尺度。特征聚合没有脱离 autograd，Python 数值转换只在 controller 边界。参数名集合约束为 theta/weight/bias，固定背景检查非训练 rho=0。局部 Generator 初始化且 `fork_rng` 恢复线性层构造期间的全局 RNG 消耗。
- **B — 检查质量与环境边界：通过。** 新测试含独立 NumPy 时间聚合/线性代数参考（float32/64）、两帧与连续核心推进的逐项状态对比、完整 detach 保值、非零内部 theta 梯度及真实联合更新后的下一帧反向。其余检查覆盖角色、身份、相机、精度、时钟、异常、全部 DN 和注册输入，以及 controller 明确 reset、忽略 observation、零权重无视觉跳接。复核现有 rollout 的调用时间与外部丢失目标保护，适配器未复制教师或保护规则。
- **C — 文档及独立软件审查：通过。** 文档明确工程输入/背景假设、三种时间和当前未训练状态；旧 gain-only checkpoint 未扩展或冒充完整策略持久化。改动仅为本批新增模块、测试、说明和实施报告，未改动正在运行诊断绑定的既有源码；本报告不提前宣称协调者已 commit。

## Validation

- Risk Level: **S3**，但范围只涉及本批核心跨模块接口。
- Evidence Reviewed / Checks Performed:
  - 阅读项目规则、P2e 契约、实施报告、全部三个新增实现/测试/文档文件，以及其直接使用的脑状态验证、完整图/映射加载、驱动恢复、目标编码和 rollout 接口。
  - 阅读未跟踪新增文件，并核对开始验收时 Git 状态：本批四个新增文件，既有被绑定模块无修改。
  - 采用实施者已报告的定向执行证据：`PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_neural_policy.py`，**28 passed, 1 warning in 3.90s，exit 0**；独立核对测试断言是否覆盖契约，不机械重跑。
  - 本验收未执行模型、测试、全量检查或父进程证据校验，未运行完整图或硬件，未操作 Noria/Docker。未发现需要新增执行来澄清的具体风险。

## Issues

无阻断问题；没有业务代码修改或返工要求。

## Regression Risks

本次可证明数值接口及小电路反向路径，不能证明完整 MaleCNS 对当前任务已有足够的动作可辨识性、训练收敛、实时性或安全飞行能力。实际训练窗、资源预算、完整策略 checkpoint、冻结脑/急性恢复/直接视觉/拓扑对照仍属于后续子批；不得以本 PASS 关闭整体 AC3–AC6。

## Required Rework

无。协调者读取本报告后可按已授权阶段提交并推进后续协议。
