# Acceptance — P2-01 CI 三态解释返工

## Result

**PASS**。P2-01 已关闭；与 [上一轮验收](acceptance-p2.md) 中其余通过项合并，P2 工具验收通过。本次不替代 P3 集成验收或真实动作数据的研究验证。

## Spec Coverage

- `paired_scene_comparison()` 使用严格比较：仅 `ci[0] > practical_delta` 支持实用优势；仅 `ci[1] < practical_delta` 表示不支持达到登记实用量；其余返回 `insufficient_evidence`。跨越、任一端点接触或两端同时等于 δ 均不会归入前两类。
- 原有簇不足、缺失场景评分及覆盖不足的提前返回路径保留，仍返回 `ci95=None` 与 `insufficient_evidence`。
- 返回结构、配对身份/覆盖校验、差值方向、随机种子、簇抽样与百分位计算保持原有行为；修正限定在最终解释状态，不涉及预测、评分或拟合。
- 新增一个相关测试覆盖全部高于、全部低于、跨越 δ，以及下界、上界、两端等于 δ 六种情况。二进制可精确表示的 0.25 阈值使边界检查不依赖隐含容差。
- `docs/LEARNING_V2.md` 的状态语义与实现及设计一致，并明确低于实用量不表示等效、覆盖不足不生成区间。

## Validation

- Risk Level：**S1**，单个评估函数的局部结论状态修复。
- 阅读项目规则、验收流程、Spec/Plan、设计判据、前一轮验收、返工要求和实施报告；独立核对评估函数、定向测试及使用文档。
- 采用 [实施报告](implementation-p2-r1.md) 中的已通过证据：`.venv/bin/python -m pytest tests/test_learning_v2.py::test_paired_comparison_distinguishes_ci_threshold_and_equal_boundaries -q`，**1 passed in 0.03s**。未发现需要复现的新疑点，因此未重复运行。
- 本次未执行测试、build、lint、真实数据训练、神经仿真或设备操作；未修改业务代码，未提交 Git。前一轮其余 P2 通过项未重复验收。

## Issues

无本次返工直接相关的未解决问题。

## Regression Risks

该分类只解释已算出的区间；正式结论仍取决于预登记、独立场景、覆盖和完整对照。合同数据与工具通过不构成真实动作性能或连接组优势证据。

## Required Rework

无。满足局部返工停止条件，不追加验证。
