# Acceptance — P2c 有界早期视觉背景变体

## Result

PASS

独立验收者：`tonic_acceptance`。未参与本轮实施，未修改业务代码。结论仅覆盖 [P2c](p2c-tonic.md) A–F 的数值实现、身份与持久化保护，不代表完整图传播、内部学习任务或实飞成功。

## Spec Coverage

| 标准 | 结论 | 独立核对的证据 |
|---|---|---|
| A 原无背景保持 | PASS | `TrainableLIF` 新 hook 默认返回 None；原积分表达式与事件顺序未改，默认分支不执行背景算术、没有 rho。既有独立矩阵指数参考覆盖相等/不等时间常数、延迟、抑制、不应期与输入豁免；新增测试核对严格静息及 theta 唯一参数。P1 运行协议未改。 |
| B 独立解析数值参考 | PASS | `test_tonic_dynamics.py` 用独立指数解计算首次严格跨阈值为第416步，并逐步核对跨阈值前电位；覆盖复位、不应期内 v 冻结、到达 g 累积与精确恢复边界。期望值不是由被测模型生成。背景在 active 的电位积分内加入，g 的原方程保持。 |
| C 类型、参数与边界 | PASS | 生产构造固定精确 L1–L5，要求每组存在并拒绝与外部输入重叠；测试工厂只允许有序非空子集。缺失注释归组沿用已验证图契约。rho 的 shape/CPU/dtype/finite 在背景计算前校验；nextafter 保护严格范围，非成员明确补0。fixed 为 buffer、learnable 为参数，两者初值前向相同；独立单步导数、参数更新及 reset 保留参数均有定向证据。 |
| D 图及身份 | PASS | 变体复用原 TrainingGraph 和完整积分，不重新筛选图或改写 counts；生产载入沿用完整参考校验路径。新增 schema、模式、类型顺序、成员数量/body ID 摘要、映射摘要、公式/范围/初值、tonic.py 摘要进入 model_id。数值工厂继续标记 numerical_test_only；测试覆盖原图计数/角色保留以及原核心、fixed、learnable 间状态混用拒绝。这里的计数保持证据是代码与小电路数值验证，没有新加载完整图。 |
| E 持久化保护 | PASS | save/restore 首先核对原动力学 schema、精确 theta 参数集合（包括额外别名注册）、形状/设备/精度/有限值，之后才创建目录或读取/恢复参数。read 在载入参数前拒绝非原 schema、非唯一 theta 声明与非法 theta 元数据结构；production load 先走 read。测试覆盖旧 theta 精确往返、两种 tonic 模式拒绝、额外参数/别名、缺失/额外 metadata 键及无部分写入/恢复。 |
| F 文档与边界 | PASS | TRAINABLE_SNN、Decision 010、signal-observability 和计划一致区分工程8mV候选、约41.6ms解析时间、严格零背景对照与后续配对刺激实验。明确旧 checkpoint 不支持变体及源码身份变化，不把数值通过表述为全图或任务通过。observability-plan 为未来 Draft，入口明确未实现、时间账本明确为候选。 |

## Validation

- Risk Level：S3，涉及共享积分入口及 checkpoint 持久化；验收范围限定本批直接影响。
- 已阅读 AGENTS、使命/技术栈、全部现行 `.ai/rules/`、验收流程、任务交接、母 Spec/Plan、Decision 010、P2c 和实施报告。
- 已独立审阅相关 tracked diff、未跟踪的新 `tonic.py` / 数值测试 / 规格文档，以及核心/图加载/checkpoint 完整实现与直接相关测试。
- 审阅实施者记录的唯一测试命令及结果：`tests/test_trainable_core.py tests/test_tonic_dynamics.py tests/test_training_probe.py`，使用实施报告所列筛选，**53 passed，8 deselected，1.42 s**；另有实施者 `git diff --check` 通过记录。
- 上述测试结果属于实施阶段证据。验收未重复执行已通过的测试或 diff 检查：静态审查未发现需要额外运行确认的具体缺口。
- 未运行完整测试集、完整图、资源监督进程、性能测量、任务训练或硬件；没有重新尝试用户仍在释放内存期间的 P1 探针。

## Issues

未发现阻断本批 A–F 验收的问题。

## Regression Risks / Limits

- 共享核心源码摘要改变，旧模型身份的 checkpoint 不会自动兼容；文档已说明。原 gain-only schema 内往返仍有测试证据。
- 变体尚无完整 checkpoint，现有 API 明确拒绝，后续策略持久化必须同时保存/验证 rho 及其动力学身份。
- L1–L5 背景偏移仍是工程假设。小电路解析一致不证明完整图视觉信号到达下降神经元，不证明8mV的生理正确性，也不证明控制收益。
- 固定背景的自发活动和刺激引起的活动差异须在后续预登记的配对实验中区分；Draft 诊断协议和资源预算不是已实施能力或运行记录。

## Required Rework

无。本批可由协调者更新状态并按授权分步提交；整体 Spec 008 的完整图更新、任务学习、贡献对照与部署证据仍待完成。
