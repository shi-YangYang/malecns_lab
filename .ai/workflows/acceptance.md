# 独立验收流程

1. 使用未参与本轮实施的新 Agent。阅读 [AGENTS.md](../../AGENTS.md)、[使命](../../constitution/mission.md)、[技术栈](../../constitution/tech-stack.md)、[项目规则](../rules/project.md)、[交接](../prompts/task-handoff.md)、对应 Spec/Plan/实施报告和相关 Decision。
2. 按每条验收标准核对实际文件与差异，包括尚未被 Git diff 展示的新增文件。区分本轮变更、既有用户修改及前序任务清理；不将它们混算。
3. 独立判断与风险相称，不机械重跑实施者已通过的检查。S0 通常只需文档、事实、路径和差异审查；是否执行检查由当前 Plan 与具体问题决定。
4. 在当前 Spec 的 `acceptance.md` 按[验收模板](../../specs/_template/acceptance.md)记录 PASS 或 FAIL、标准覆盖、真实验证、问题、回归风险和返工要求。不用模板占位符冒充结果，不代替实施者修改实现。
5. FAIL 返回协调者组织新的实施 Agent 和新的验收 Agent。PASS 返回协调者更新状态与交付；满足停止条件后不追加无依据检查。

若运行环境不支持独立 Agent，必须明确写为单 Agent 二次审查，不能伪称独立验收。
