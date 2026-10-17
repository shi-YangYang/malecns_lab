# Acceptance — 第二版模型完整数据优先修订

## Result

PASS

本记录由未参与本次修订实施的独立验收 Agent 完成，仅验收新增 D7 及直接关联条款。原 D1–D6、R1 与前置数据审计不重复验收；原 [acceptance.md](acceptance.md) 保留历史。本次通过不表示完整参考图已构建、阈值已实测或 M1–M4 已通过。

## Spec Coverage

| 标准 / 关联条款 | 证据 | 判断 |
|---|---|---|
| D7 / F1：完整数据母本与全集边界 | [Spec F1](spec.md)、[设计：图构建和身份](../../docs/MODEL_V2.md)、[Decision 007](../../.ai/decisions/007-full-dataset-baseline.md)要求完整官方发布母本及来源清单、逐 body ID 核实全集；原始片段、待定与非神经元记录分别对账，不把旧数量、superclass 或 Traced-only 筛选直接当全集，不因注释缺失删神经元。官方 minconf 0.5 明确为发布边界。 | PASS |
| D7 / F1、M1：参考图保留与阶段门槛 | Spec、设计和 [Plan P0](plan.md)一致要求全集内部全部已发布非零连接、原始自连接、孤立节点及未知递质节点；旧 166,700 / 10,520,377 仅描述 V1。完整身份未核实不得回退旧图进入正式实验，M1–M4 保持未执行。 | PASS |
| D7 / F1、F8：简化须有实验依据 | 设计“简化必须由实测支持”、Spec F1/F8、Decision 007 明确阈值 3 尚无本项目实测支持；先取得完整基线，再对候选按登记容差及预算作受控比较，训练/验证选择、独立测试确认。资源不足如实记录，不能静默裁图，派生版本不覆盖完整参考及结果。 | PASS |
| D7 / F7：完整母图与对照自连接一致 | Spec F7、设计 G/R/W 表和“重连的精确定义”、Plan P3 均使用同一完整参考图。R/W 固定原自连接及其权重，只改变非自环边；R 拒绝新增自环。未把删自连接混入主拓扑或权重对照，既有不变量声明与该策略一致。 | PASS |
| D7：规则、决策与状态同步 | [Decision 006](../../.ai/decisions/006-model-v2.md)明确优先遵循 Decision 007；[项目规则](../../.ai/rules/project.md)、[路线](../../constitution/roadmap.md)、[交接](../../.ai/prompts/task-handoff.md)、[实施报告 Amendment](implementation.md)同步完整数据优先与未执行边界。33 个输出记录节点明确只是探针，不是裁图边界。 | PASS |

## Validation

- Risk Level：S0。此次为设计与治理文档修订，没有运行时实现范围。
- Evidence Reviewed / Checks Performed：阅读任务要求及项目入口、Spec F1/F7/F8/D7/M1、Plan、设计相关段落、Decision 006/007、实施报告最新 Amendment、项目规则、路线和交接；检查 Git 状态及上述已跟踪治理文档差异。新增设计文档直接阅读，不以未出现在普通 diff 中视为不存在。
- 本次只确认设计要求与文档一致性；完整数据实际身份、规模、保留情况、资源需求和简化效果仍需未来 P0 及研究实施验证，未伪写为本次执行结果。
- 未执行 tests、build、lint、typecheck、构图、神经仿真、训练、原始大表审计、资源测量或设备操作；S0 文档验收不需要这些执行。未重复既有验收或检查。

## Issues

未发现 D7 及直接关联修改的阻塞问题。Spec 中 D7 当前待勾选、Plan 当前 Acceptance 为交付前状态，由协调者读取本报告后更新。

## Regression Risks

本次没有业务运行行为变更。后续主要风险是把“完整发布母本”误称为过滤前全部生物观测，或把神经元参考图误称为所有原始 segment 图；修订已要求显式边界和逐项对账。完整图仍未构建和测量，不应将本次 PASS 外推为模型完整性或有效性结论。

## Required Rework

无需返工。D7 增量文档验收完成，按停止规则结束；不追加构图或实验。
