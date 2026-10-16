# Plan — 接入 CNSBridge Agent 开发流程

状态：Done。实施完成，独立验收 PASS，无返工；见 [验收报告](acceptance.md)。

## 涉及模块

仅 Agent 治理文档与 README 导航；业务模块不变。基线提交为 `bf73319`。

## 修改顺序

1. 协调 Agent 读取用户规范、README、工程配置、架构与研究文档，并建立本 Spec。
2. 一名实施 Agent 按 Spec 完成 AGENTS、constitution、模板、`.ai/` 文档和 README 导航，返回 `implementation.md`。
3. 新的独立验收 Agent 逐项阅读 Spec、实施报告和相关差异，在 `acceptance.md` 记录 PASS/FAIL；不修改实现。
4. 有具体问题才按规范进入返工；PASS 后协调 Agent 更新状态并提交。

## 数据流与接口变化

无运行时接口或数据流变更。治理信息从项目事实进入 constitution，再由 Spec → 实施报告 → 独立验收报告记录任务；`.ai/prompts/task-handoff.md` 指向当前任务与阶段。

## 预计涉及文件

- `AGENTS.md`
- `constitution/{mission,roadmap,tech-stack}.md`
- `specs/README.md`、`specs/_template/{spec,plan,acceptance}.md`
- `specs/spec-001-agent-workflow/{spec,plan,implementation,acceptance}.md`
- `.ai/decisions/001-adopt-agent-workflow.md`
- `.ai/workflows/{implementation,acceptance}.md`
- `.ai/prompts/task-handoff.md`
- `.ai/rules/project.md`
- `README.md`

前一轮待提交的三项删除另行提交，本轮不重做清理或删除其他文件。

## 验证计划

Risk Level: S0。实施者阅读新增文档与差异、核对路径及项目事实；独立验收者按 AC1–AC6 进行文档审查，可执行一次针对差异的 `git diff --check`。不运行 pytest、build、lint、typecheck、飞行工具或数据流程；上一轮业务测试结果不可写成本轮执行结果。通过后不追加检查。

## 风险与迁移策略

主要风险是文档将规划能力误写为已验证、引入过时命令，或无必要迁移业务目录。以现有 README、pyproject、docs 和代码为事实来源，链接具体实验流程，不复刻第二套使用手册。仅新增治理文件和 README 入口，无数据迁移。

## 分工与交付

协调 Agent 负责 Spec、状态、关键决策核对和本地提交；实施 Agent 仅负责列出的文档；独立验收 Agent 仅负责审查和验收报告。实现与验收使用不同 Agent。所有报告处理完成后交付，不创建远程仓库或推送。
