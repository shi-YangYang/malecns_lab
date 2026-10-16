# Specs

Spec 统一位于 `specs/spec-XXX-short-name/`，编号递增，名称用小写英文与连字符。不要另建 `.ai/specs/` 或 `docs/specs/`。

## 生命周期

`Initialization → Planning → Implementation → Acceptance → Done`；验收失败进入 `Rework → Acceptance`。

协调 Agent 在 Planning 阶段完成足够的需求理解，将无阻塞关键问题的 Spec 标记 `Ready` 后才分配实施。无需为形式重复询问用户已明确的信息。每个状态应反映真实进展；`Ready` 表示可实施，不表示已验收。

| 文件 | 责任与内容 |
|---|---|
| `spec.md` | 协调者维护目标、边界、兼容性与逐项验收标准 |
| `plan.md` | 实施顺序、预计文件、风险、迁移与相称的验证计划 |
| `implementation.md` | 实施者记录实际修改、实际验证与限制 |
| `acceptance.md` | 新的独立验收者逐项判断并记录 PASS 或 FAIL |
| `rework.md`（需要时） | 协调者整理失败原因、返工范围及完成条件 |

从 [_template/spec.md](_template/spec.md)、[plan.md](_template/plan.md)、[acceptance.md](_template/acceptance.md) 创建任务文档。模板占位符不是事实；未验收不得预填 PASS。正式验收报告的结论只能是 PASS 或 FAIL，并给出逐项证据。

验收 FAIL 时，由协调者创建新的实施 Agent 返工，再创建新的独立验收 Agent；实施者不负责最终验收。环境不支持多 Agent 时，按 [AGENTS.md](../AGENTS.md) 做阶段隔离，并如实记录单 Agent 审查。

实施与验收都按 S0–S3 控制验证范围，不机械重复已通过检查。PASS 后由协调者核对报告、更新状态；只有用户要求或工作流明确要求时创建提交，使用 `<type>: <中文说明>`，不默认推送。

## 当前记录

- [spec-001-agent-workflow](spec-001-agent-workflow/spec.md)：已有项目接入 Agent 开发流程。当前阶段以 [Plan](spec-001-agent-workflow/plan.md) 和[任务交接](../.ai/prompts/task-handoff.md)为准。

- [spec-002-business-layout](spec-002-business-layout/spec.md)：按职责重构业务目录，当前状态见对应 Plan 与任务交接。
- [spec-003-remove-compat](spec-003-remove-compat/spec.md)：删除旧兼容层，统一新模块入口与文档命令。
- [spec-004-neural-activity](spec-004-neural-activity/spec.md)：离线神经活动记录、刺激比较与可视化。
- [spec-005-remove-minifly](spec-005-remove-minifly/spec.md)：移除合成飞行网络，所有脑入口统一真实连接组。
