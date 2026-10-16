# Task Handoff — spec-004-neural-activity

## Role / Project Mode

Done；EXISTING。实施、文档、三图目检和独立验收完成，AC1–AC5 全部 PASS，基线 5854ba1。全部子任务结果已处理，无未处理报告或返工项。本记录随提交保存，不是待执行任务。

## Goal / Context

用户选择先观察录好的视频产生的模拟神经活动。已新增离线诊断工具，不需要动作标签；真实 MaleCNS 录像的刺激响应仍待实际采集后分析。

## Spec

- [Spec](../../specs/spec-004-neural-activity/spec.md)
- [Plan](../../specs/spec-004-neural-activity/plan.md)
- [Decision 004](../decisions/004-neural-activity.md)
- [实施报告](../../specs/spec-004-neural-activity/implementation.md)
- [独立验收](../../specs/spec-004-neural-activity/acceptance.md)

## Scope / Relevant Files

新增 diagnostics 包、定向测试与 analysis 可选依赖，更新使用指南及相关治理说明。入口为 python -m cnsbridge.diagnostics.activity，输出新目录中的 activity.npz、CSV/JSON、三张 PNG 和 index.html。具体命令见 [神经活动诊断](../../docs/NEURAL_ACTIVITY.md)。

## Out of Scope / Constraints

核心 Brain/LIF/Retina/InputEncoder 四文件未变；不改变旧数据格式或控制/学习链路。未运行设备、真实录像、全量 MaleCNS 或训练，未提交测试实验资产。合成演示保存在忽略目录 data/neural_activity_demo/synthetic_32459262ba/。

## Acceptance Criteria

AC1–AC5 全部 PASS：离线入口/错误处理，计数与单位，刺激与时基统计，可读图表，文档与兼容边界。无返工项。

## Commands

S3 新持久化格式，验证限制在新诊断子系统。一次 .venv/bin/python -m pytest tests/test_neural_activity.py -q -s：18 passed，退出码 0；协调者一次目检三图，独立验收 git diff --check 通过并确认核心四文件未变。无待执行检查，不重复通过测试。

## Expected Output

所有结果已处理，用户已授权 commit。交付标题 feat: 新增离线神经活动记录与刺激分析；哈希以 Git 历史为准。不 push，任务闭环；收到新需求后再开始后续工作。
