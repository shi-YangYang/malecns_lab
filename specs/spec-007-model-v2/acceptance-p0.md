# Acceptance — P0 完整神经元参考图

## Result

**FAIL**。完整数据构图和账本证据符合本批范围，但完整参考图的群组角色校验存在可复现缺口，需局部返工。

本报告仅验收 Spec 007 P0 / F1 数据基础及选组，不代表 F1–F3 全部完成，也不验收并行视觉模块、P1 神经重放、P2 学习或 P3 对照。M1–M4 不据此勾选。

## Spec Coverage

| 标准 | 审查证据 | 判断 |
|---|---|---|
| 官方 neuron 身份有出处，逐 body ID 对账，不按旧数量凑集合 | Decision 007、复制的 `identity_reconciliation.json` / `identity_findings.txt`、`metadata.load_identity`；明确作者预印本 Methods 与 v1.0 表的版本边界 | PASS |
| 完整发布母本、11 文件用途及去重、已发布 minconf 0.5 边界明确 | `metadata.SOURCE_USES` / `verified_release`、完整下载清单；只把完整 weights 用作边源 | PASS |
| 全部内部正边、自环、孤立节点、缺注释节点保留 | `reference.build_reference`、`reference_audit`、实际 `build_report.json` / `graph_ledger.json`：166,700 节点、25,582,938 边、124,177,617 突触；101 条自环 / 474 突触、217 孤立节点 | PASS |
| 原始片段图与神经元图分开；边界节点和受影响边可回查 | 四类身份、逐源行 uint8 分类、批偏移、64 个唯一端点桶、未出现于原始边的注释 ID；原始 151,856,684 边 / 311,833,243 突触，未把未标注 segment 称为完整神经元 | PASS |
| nullable 注释、未知递质假设、精确整数计数保留 | `node_annotations`、int64 CSC、保存/加载字段与缺失掩码；未知符号 3,177、缺失递质 178 有单独记录 | PASS |
| 独立资产、来源摘要、拒绝覆盖及生产隔离 | 新输出目录排他创建、NPZ 排他保存、图/元数据/来源校验；生产加载与直接 Brain 构造拒绝 research_control 和 synthetic；裁图继承旧身份已被测试拒绝 | PASS |
| R16 / HAL 侧别和 subclass、33 输出 body ID 固定 | `malecns_v2.yaml`、`reference_group_specs`、`group_mapping.json` / `load_check.json`；R16 1,112/2,265，HAL 104/101，输出 33 个按 body ID 排序；HAL 驱动禁用的执行属于后续 P1 | PASS |
| 完整参考图运行选组与记录身份一致，错误配置明确失败 | V1 成员差异会失败；但 input/output 角色变更和重叠未检查，见下方 P0-R1 | FAIL |

## Validation

- Risk Level：**S3**。本批涉及核心连接组资产模式、来源身份及 Brain 入口；验收限于这些直接影响模块。
- 阅读 Spec、执行计划、设计相关数据/选组约束、Decision 007、实施报告、项目规则与验收流程；检查新增源码和 `connectome.py` / `brain.py` 差异。
- 审阅实际构建/加载报告及身份依据。实测 95.7004 秒、峰值 1,307,279,360 bytes 为构建报告结果，不是验收者重新测量。
- 采纳实施报告中相关测试 **9 passed** 和裁图直接构造保护的单测 **1 passed**，未机械重跑。Ruff 因环境缺模块未成功运行，未记为通过。
- 针对源码中发现的角色校验疑点，仅执行一次临时小 fixture 复现：复用 `tests/test_full_reference.py::source_fixture` 建立契约样例，将 `cfg['outputs']['DNg02_L']` 移到 `cfg['inputs']`，随后构造 `Brain`。结果为 `accepted=True`、`recorded_role='output'`、`runtime_role='input'`、`source_id_unchanged=True`。没有运行神经仿真；fixture 不构成真实脑实验。
- 未重建真实参考图、重新 hash/扫描大数据、重复已通过测试、检查无关视觉实现、连接设备或创建 commit。

## Issues

### P0-R1 — 完整参考群组校验遗漏输入/输出角色

位置：`src/cnsbridge/brain/brain.py:61–72`。

当前把输入和输出规格合并后，只核对群组名与 body ID。把 `DNg02_L` 从 `outputs` 移入 `inputs`，保留其选择规则，即可通过校验；随后 `meta.roles` 被写为 input，但 `provenance.group_mapping.rules.DNg02_L.role` 仍是 output，来源身份没有变化。该组包含 15 个输出细胞，因此这种配置错误会实质改变输入驱动与读出集合，而不是文案差异。同名组同时存在于 inputs 和 outputs 时，字典合并也会掩盖重叠。

这违反 F1 的选组身份及错误配置明确失败要求。现有完整图及默认 V2 配置的实测统计未受此问题影响。

## Regression Risks

返工应只加强完整参考图的配置角色分区校验，保留旧 V1 行为；不要为此重建或覆盖完整参考资产。不得把图资产来源身份误称为已经包含后续 P1 运行时间、视觉、动力学和完整运行配置的模型身份。

## Required Rework

1. 在应用/修改完整参考图的运行群组和角色前，拒绝 inputs 与 outputs 的名称重叠，并核对各组所属角色与记录的规则完全一致；保留现有成员核验。
2. 对“输出组移入 inputs”及“同名组同时出现在两侧”补充最小回归断言；错误应明确失败，默认 V2 配置继续被接受。只运行相关检查，无须重建真实图。
3. 由新的独立验收 Agent 审查返工；本验收者不修改业务代码。
