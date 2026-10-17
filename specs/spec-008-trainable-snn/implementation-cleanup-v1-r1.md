# Implementation Report — V1 清理文档返工 R1

## Summary

按 [返工范围](rework-cleanup-v1.md)修复原独立验收发现的两处文档遗漏。保留审计事实与原 [FAIL](acceptance-cleanup-v1.md)；本次不修改业务实现或原实施报告。

## Files Changed

- `docs/MALECNS_AUDIT.md`：把已删除 collector 的活动链接改为 `747a8aa:src/cnsbridge/diagnostics/collector.py:72` 历史源码身份，明确 V1 入口已移除，并指向现行 `MODEL_V2_USAGE.md`。
- `docs/MODEL_V2.md`：明确旧资产保留不代表旧离线流程仍可执行；将旧 prepare/decoder 规划替换成实际 V2 模块，区分已验收的 P0–P3 工具、尚未验证的研究结果及后续 P4 范围。
- 本返工报告。

## Important Decisions

历史资产和审计结论继续保留；只纠正当前入口与实现状态，未把工具验收描述为学习成功、结构优势或实飞成功。

## Validation

- Risk Level: S0（两处文档表述与引用修正）。
- 阅读对应差异，核对表中 V2 文件实际存在；定向检查两份文档中的旧 collector/prepare/decoder 引用，剩余项均为明确标注的历史源码身份。
- `git diff --check -- docs/MALECNS_AUDIT.md docs/MODEL_V2.md` 无错误。
- 未重复已通过的 20 项测试，未运行训练、仿真、全图诊断或硬件操作。

## Known Limitations

本报告只覆盖两处 S0 返工；不改变原清理范围、其他并行实现或其验收状态。

## Remaining Questions

无实施阻塞。交协调者安排新的独立验收；未 commit。
