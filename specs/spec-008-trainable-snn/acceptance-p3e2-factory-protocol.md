# Acceptance — P3e2 研究策略工厂协议

## Result

**PASS**。2026-10-19，由未参与本轮设计撰写、边界审计或实施的新独立验收 Agent，对 [工厂 Spec](p3e2-research-factory.md)、[Plan](p3e2-factory-plan.md)和实际源码进行静态审查。当前 factory-only 范围具备可实施条件，可由协调者标记 Ready。

本结论是协议审查，不是软件实现、完整 R 构造、神经前向、训练或 AC6 的验收。全部软件 AC 仍需实施及新的独立验收；本批不要求提前实现参数、Adam、完整状态存档或研究 runner。

## Spec Coverage

| 范围 | 实际代码依据及协议判断 |
| --- | --- |
| 正式终局准入 | `generate_rewiring.accepted_candidate` 实际重新验证 run 来源/runtime、`scan_run` 的父批准链及唯一 terminal，并绑定候选文件。Spec 要求继续调用该门槛，再交叉绑定 candidate 的 transform、生成 run/checkpoint、母本、机械诊断与源身份；没有把 `load_research_control` 或文件存在误作正式终局充分条件。父层仅小元数据/头和流式摘要，worker 重验实际数组，分层清楚。 |
| 构造可行性 | `LaminaTonicLIF` 的真实构造确实调用只接受 G/数值夹具的基类，但构造本身不生成 LIFState 或执行神经 step。先用真实 G 初始化精确原类，再于首次状态/计算之前完整安装已验证 R，在接口层可行。公开工厂不接收半成品模型或运行状态，R 从未临时冒充 G 执行。 |
| 实际计算缓存 | 当前 `step` 正向调用 `_forward_csr`，自定义反向读取 `base_weights_adjoint`；仅改 `brain.graph` 无效。Spec 同时覆盖只读整数 CSC、Torch 正向/伴随 CSR、SciPy 共享视图、所有指针/索引/值及 float32 权重操作顺序，也要求释放旧视图引用。不同图实际 forward/全细胞 adjoint 与旧缓存反例均已列为软件验收条件，能够排除仅换标签。 |
| 完整元数据与共同计算 | `_graph` 会按真实数组生成 ordered-counts 和类型分组身份；`_build_policy_map` 可对已验证 R 构建全部原输入/DN，空间分配仅依赖同一节点/注释数值。Spec 保持完整 body 顺序、分组、全部 L1–L5 固定背景、3377 输入/1314 DN、空间/相位/读出初始化数值及严格共同契约。图相关 ID 允许真实变化，不通过删除任意字段掩盖数值差异。 |
| 身份和状态隔离 | `NeuralPolicy._initialize` 接受精确原脑类型并核对显式 role、图/map 及嵌套身份，没有写死只允许 G。`LIFState` 和 `PolicyState` 随 model/policy ID 隔离，感觉状态还绑定 drive/map。Spec 要求在初始化策略前重绑真实模型身份、加入工厂来源，随后重算 policy ID，并验证跨图状态拒绝；未要求修改原状态标签或原方法。 |
| 旧边界保留 | 旧 `load_training_graph`/生产 map 保持 full_reference；`policy_checkpoint._validate_identity` 拒绝 research_control，`training_optimizer._policy_parameters` 仍经 `_arrays` 触达该限制。Spec 如实保留这些拒绝，不以 `production=False` 或临时身份调用旧持久化。最新 F3/AC-F4 同时明确 `NeuralController` 只检查类型，不能声称它是角色隔离器；本批没有新增硬件接线。 |
| 来源与批次范围 | G 的 29 个和 R 生成的 13 个绑定源码保持，新增工厂适配来源进入新研究身份。Plan 仅新增必要准入/工厂/缓存校验及说明，复用原动力学和感觉/读出；持久化与调度另批。最终 R 后的受监督实际门槛和正式比较登记明确在首次 R 神经结果之前冻结，不由本协议授权提前运行。 |

## Validation

- Risk Level：**S0（本次仅设计文档与静态接口审查）**；将来实施的计算缓存/准入风险由 Spec 正确归为 S3，但不据此在本次运行测试。
- 阅读 AGENTS、使命/技术栈、项目规则、独立验收流程、Spec 008、当前交接及 Decision 005/007/008/009/010/011/012；阅读 P3e1 首两段实际 PASS、P3e2 边界审计及最新工厂 Spec/Plan。
- 静态核对 `trainable`、`tonic`、`trainable_graph`、`policy_groups`、`neural_policy`、`spatial_drive`、`generate_rewiring`、`rewiring_evidence`、`graph_controls`、旧参数/Adam 准入的直接调用关系。
- 检查工作区状态；本次只新增本验收文件，其他未提交规划/交接/运行报告由协调者维护，未将其归为本验收产出。
- 未导入 Torch，未运行测试/build/lint/typecheck，未加载完整矩阵、构造模型、执行图提案或神经计算。未重复已通过的生成器/模型检查；未操作 Noria、Docker、设备或资源设置；未 commit。

## Issues

未发现阻止当前缩小范围实施的协议冲突或关键遗漏。此前审计指出的 `NeuralController` 类型校验边界已经在本次审查所见最新 Spec 中准确表述，无需为本批扩大修改被冻结的原控制器源码。

## Regression Risks

- 原构造函数、`_graph`、`_build_policy_map`、`NeuralPolicy._initialize` 和内部缓存均是具体实现接口，不能假定只凭类型和自洽 JSON 就完成实际矩阵绑定。实施及软件验收必须按 AC-F2/F3 覆盖真实数组、共享存储、完整成员和旧缓存反例。
- 旧 G 缓存的 NumPy/SciPy 视图可能延长 Torch 存储寿命；同时存在的原始图、母本校验和 `_graph` 拷贝也影响峰值。本设计没有承诺其成本等同 G，首次完整构造需等待终局并另行登记、串行监督。
- 返回精确 `NeuralPolicy` 是离线研究构造能力，不是全应用运行时权限隔离。现有内存 `NeuralController` 接受该类型这一限制必须在说明中保留；本批不可据此接入设备。
- 不同图正确传播、非零梯度或初始输出差异均不能替代完整配对训练和任务收益证据；允许负结果，后续不得根据提前观察的 R 神经结果选择协议。

## Required Rework

无。本协议验收到此停止。协调者可以将当前工厂 Spec/Plan 标记 Ready 后安排单一实施者，完成软件验收并提交；实际 R 构造/前向、研究存档/训练及 G/R 比较仍按其后续门槛执行。
