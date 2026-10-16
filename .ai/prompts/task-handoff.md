# Task Handoff — spec-002-business-layout

## Role

Done。业务重构与独立验收已完成，AC1–AC6 全部 PASS，无未处理报告或返工项。本文件随重构提交保存，不是待执行的实施或验收指令。

## Goal

业务代码已按现有职责迁入 control、learning、experiments/tello；原工具路径、Python 导入和实验行为保留。恢复时阅读结果，不重复实施；后续需求另建对应 Spec。

## Context / Project Mode

EXISTING，基线 `8546f6c`。用户已明确“业务目录也要重构”；目录调整授权已落实。交付提交标题为 `refactor: 按控制学习与实验职责重组业务目录`，具体哈希以 Git 历史为准。

## Spec

- [Spec](../../specs/spec-002-business-layout/spec.md)
- [Plan](../../specs/spec-002-business-layout/plan.md)
- [Decision 002](../decisions/002-business-layout.md)
- [实施报告](../../specs/spec-002-business-layout/implementation.md)
- [验收报告](../../specs/spec-002-business-layout/acceptance.md)

## Relevant Files / Scope

src/cnsbridge 的动作/学习/控制保护/校准实现，七个 tools、对应测试与直接文档。具体映射以 Spec 为准。一个实施 Agent 串行处理耦合迁移；主 Agent 管理本交接和 Spec 状态。

## Out of Scope

算法改进、飞行参数调整、模型训练、实机连接、实验资产、依赖更新和远程发布。

## Constraints

遵循 AGENTS、constitution 和项目规则。四个神经/视觉哈希源保持字节不变；持久化格式和 SI/FLU 契约不变；旧脚本不承载重复实现。工作目录变化不能改变原项目配置/数据定位。

## Acceptance Criteria

AC1–AC6：真实职责拆分、依赖与导入兼容、原断言回归、任意 cwd 下无硬件入口、数据与行为不变、文档及独立验收。

## Commands / Validation

S3。`.venv/bin/python -m pytest -q` 已取得明确结果：144 passed、251 subtests passed；首次输出捕获不完整及按规则重跑的原因见实施报告。旧 schema 基准、四个哈希源字节和 trial/hover 函数 AST 已对照基线核验；独立审查及一次 git diff --check 通过。无待执行验证，不重复检查或连接硬件、下载、训练。

## Expected Output

已产出独立 PASS 报告，协调者已处理全部结果并更新完成状态。任务闭环，无剩余问题；收到新需求后再推进新的工作。
