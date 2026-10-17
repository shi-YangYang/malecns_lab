# Acceptance — P0-R1 群组角色校验

## Result

**PASS**。本轮由未参与返工的新验收 Agent 独立审查，仅关闭 [原 P0 验收](acceptance-p0.md) 的 P0-R1 角色校验问题。原 FAIL 报告保留，其余已通过条目不重复验收；本结论不代表 P1–P3 或 M1–M4 已完成。

## Spec Coverage

| 返工要求 | 独立审查证据 | 结论 |
| --- | --- | --- |
| 拒绝 inputs/outputs 同名重叠 | `brain.py:64–66` 在应用配置群组前检查名称交集，明确抛出 `ValueError`；`test_full_reference.py:99–102` 覆盖同名 `DNg02_L` | PASS |
| 群组角色须与来源记录一致 | `brain.py:67–70` 先核对名称集合，再逐组比较配置角色与 `provenance.group_mapping.rules`；`test_full_reference.py:95–98` 覆盖输出移入输入 | PASS |
| 拒绝错误角色配置时不写入错误运行状态 | 两项角色检查均早于配置 `resolve_groups` 和 `meta.roles` 更新；`test_full_reference.py:93–104` 保存并核对拒绝后的群组成员、角色 | PASS |
| 默认 V2、已有成员核验及 V1 行为保留 | 默认 V2 成功构造断言仍在 `test_full_reference.py:90–92`，错误 V1 映射拒绝仍在 105–106 行；成员核验保留于 `brain.py:73–78`。新增角色限制仅作用于 `full_reference` 分支，旧资产仍按既有合并配置解析 | PASS |

## Validation

- Risk Level：**S1**，局限于 `Brain.__init__` 完整参考图的角色验证分支。
- 阅读项目规则、Spec/Plan/执行计划、P0 原验收与返工要求、P0 及 P0-R1 实施报告；核对 `brain.py` 差异和新增测试断言。另查看参考身份验证及 `GroupSpec` 解析的直接调用关系，以确认角色检查顺序。
- 采纳 [P0-R1 实施报告](implementation-p0-r1.md) 中唯一相关测试的实际结果：`tests/test_full_reference.py::test_complete_reference_retains_edges_identity_and_nullable_metadata` **1 passed in 0.43s**。验收未发现需要复现的具体疑点，因此没有机械重跑测试。
- 本验收未重建参考图、扫描或 hash 原始数据、推进神经仿真、检查并行 P1 实现、操作硬件或创建 commit。仅新增本报告。

## Issues

本轮返工范围内没有未解决问题。小型 fixture 的检查只证明配置契约，不作为真实脑响应或控制效果证据。

## Regression Risks

本次没有改变图资产、筛选规则、来源身份或原有成员核验。后续 P1 仍须独立验证完整模型的输入、动力学、因果时间与运行身份，不能由本局部 PASS 推导这些功能已经有效。

## Required Rework

无。P0-R1 已关闭，可解除该问题对 P1 完整参考图诊断的阻塞；P1 按自身验收范围继续。
