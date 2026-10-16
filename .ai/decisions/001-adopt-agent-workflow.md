# Decision 001 — 在现有 CNSBridge 中接入 Agent 工作流

日期：2026-10-16。状态：已决策，实施与验收状态见当前 Spec。

## Context

用户要求依据其提供的通用 AGENTS.md 改造项目，随后 commit。项目已有 `src/cnsbridge/`、实验工具、测试与中文文档；不是空白工程。基线为 `bf73319`。

## Decision

- 明确 `project_mode: EXISTING`、`business_code_dir: src/cnsbridge`、`multi_agent: AUTO`。
- 根目录保存完整通用规范，前置项目配置和阅读入口；补充 `constitution/`、`specs/`、`.ai/`。
- 沿用现有业务目录、包/CLI、依赖和数据契约；项目知识引用已有文档，不复制另一份实验手册。
- 本次限于文档与流程，定为 S0；由一个实施 Agent 完成，再由新的独立验收 Agent 审查。
- 用户前一轮已授权的三项文件删除单独提交；本次接入另行提交。仅本地中文 Conventional Commits，不改写历史、不推送。

## Reason

符合原规范的已有项目接入规则，且能在不改变实验行为的前提下保存项目边界、验收依据和中断恢复信息。当前请求与已有资料足够确定实施范围，没有必要重复询问研究或硬件选择。

## Alternatives

- 按 NEW 重新组织业务目录：与现有工程及用户当前需求不符。
- 只复制 AGENTS.md：缺少规范要求的项目事实、Spec、交接与验收记录。
- 同时重构飞行或训练逻辑：超出本次授权范围。

## Consequences

后续任务需维护对应 Spec 与重要决策，实施和验收各司其职。治理接入不代表新读出通过实飞、可靠避障或底盘迁移验证；这些研究结论仍需要单独证据。

相关记录：[Spec](../../specs/spec-001-agent-workflow/spec.md)、[Plan](../../specs/spec-001-agent-workflow/plan.md)、[项目规则](../rules/project.md)。
