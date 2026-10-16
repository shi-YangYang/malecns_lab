# 实施流程

1. 阅读 [AGENTS.md](../../AGENTS.md)、[使命](../../constitution/mission.md)、[技术栈](../../constitution/tech-stack.md)、[项目规则](../rules/project.md)和[当前交接](../prompts/task-handoff.md)，再读取对应 Spec、Plan、Decision。确认角色为 Implementation，Spec 已 Ready。
2. 检查工作区，识别用户已有修改；按 Plan 范围阅读相关代码和文档。不能安全合并的冲突交给协调者，不 reset 或恢复无关文件。
3. 仅实施已确认范围。高度耦合文件由同一实施 Agent 串行处理；重要新问题或范围变化先返回协调者更新 Spec。
4. 按 S0–S3 执行计划中的最小充分验证。S0 阅读差异与新增文件即可；运行逻辑检查不得启动实体设备。通过后不无理由追加或重复。
5. 在当前 Spec 目录生成 `implementation.md`，使用 AGENTS 第 14 节 Implementation Report 格式，列出实际修改、决策、验证、限制和问题。未执行检查必须如实注明。
6. 将报告交协调者并停止实施；不自己填写最终验收 PASS。协调者安排新的独立验收 Agent，提交由当前任务明确指定的责任人执行。

返工时另读完整验收报告、返工目标和上一实施报告，只修复与返工目标直接相关的问题。
