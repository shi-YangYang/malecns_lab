# Implementation Report — P0-R1 群组角色校验

## Summary

修复独立验收中复现的完整参考图角色漏洞：`Brain` 在应用配置群组和运行角色前，拒绝 inputs/outputs 同名重叠，并逐组核对 `provenance.group_mapping.rules` 中记录的角色。将 `DNg02_L` 从 outputs 移到 inputs 会明确抛出 `ValueError`，不再沿用不一致的参考身份继续构造。

## Files Changed

- `src/cnsbridge/brain/brain.py`：仅加强 `full_reference` 的配置校验；已有群组名称和 body ID 成员检查保留。V1 仍按原来的合并配置解析群组，不增加新角色限制。
- `tests/test_full_reference.py`：在现有完整参考契约测试中增加输出移入输入、两侧重名及拒绝后群组/角色不变的断言；已有默认 V2 成功构造与错误 V1 映射拒绝断言继续保留。
- 本报告。

## Important Decisions

这是已有 F1 / P0 要求的局部修复，不变更参考图身份、源数据筛选、持久化格式或构建源码。完整参考资产及先前构建统计继续保留，没有重建或重新 hash 原始数据。未修改并行 P1 的 LIF、视觉或重放模块。

## Validation

- Risk Level：**S1**，隔离于 `Brain.__init__` 的完整参考配置验证。
- 阅读相关差异，确认新检查发生在配置群组解析及 `meta.roles` 更新之前；原成员检查仍存在。
- 唯一测试命令：`.venv/bin/python -m pytest tests/test_full_reference.py::test_complete_reference_retains_edges_identity_and_nullable_metadata -q`
- 实际结果：**1 passed in 0.43s**。
- fixture 只验证配置与数据契约，没有推进神经仿真，不作为真实脑效果证据。未执行完整测试、lint、build、数据重扫、硬件操作或 commit；相关检查通过后停止验证。

## Known Limitations

本轮只修复 P0-R1，未替代 P1–P3 实现或研究结果。完整模型身份仍需后续 P1 记录编码、动力学与时间契约。

## Remaining Questions

无本轮未解决项。交由新的独立验收 Agent 审查，本报告不填写最终验收结论。
