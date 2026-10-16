# Task Handoff — spec-003-remove-compat

## Role / Project Mode

Done；EXISTING。实施与独立验收完成，AC1–AC4 全部 PASS，基线 `5bf6642`。无未处理报告或返工项。本记录随提交保存，不是待执行任务。

## Goal / Context

删除旧 motor 包、根 safety/calibrate 转发模块和 tools 七个脚本，统一真实包入口。上一任务保留兼容的要求已由当前请求替代。

上述目标已完成，后续直接使用文档中的新入口，不恢复旧转发层。交付提交标题为 `refactor: 移除旧兼容层并统一模块入口`，哈希以 Git 历史为准。

## Spec

- [Spec](../../specs/spec-003-remove-compat/spec.md)
- [Plan](../../specs/spec-003-remove-compat/plan.md)
- [Decision 003](../decisions/003-remove-compat.md)
- [实施报告](../../specs/spec-003-remove-compat/implementation.md)
- [验收报告](../../specs/spec-003-remove-compat/acceptance.md)

## Scope / Relevant Files

16 个兼容源码文件及其缓存；tests/test_business_layout.py；README、当前 docs、constitution/tech-stack、.ai/rules/project；check.py 帮助示例。主 Agent 管理本交接、Spec/Plan、Decision 和历史替代说明。

## Out of Scope / Constraints

不改算法、真实控制逻辑、持久化格式、依赖或实验资产。旧 MotorDecoder 基线与 cnsbridge calibrate 命令保留；flydrones.* 数据标识保留。禁止硬件操作、下载和训练。不改写历史验收结果。

## Acceptance Criteria

AC1–AC4：旧层移除、当前命令更新、新入口离线可用、真实实现与资产不变。

## Commands

S3 有意移除公开入口，验证只覆盖变更边界。一次 `.venv/bin/python -m pytest -q tests/test_business_layout.py` 已通过：12 passed，exit 0。独立验收审查差异、引用与文档命令，一次 git diff --check 通过。无待执行验证；不重复已通过检查。

## Expected Output

已产出独立 PASS 报告，协调者已处理全部结果并更新完成状态。任务闭环；收到新需求后再建立相应 Spec。
