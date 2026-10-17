# Acceptance — P3-01 总预算闭环返工

## Result

**PASS**。本次仅关闭 [P3 集成原验收](acceptance-p3-integration.md) 的 P3-01。原报告其余通过项继续有效，未重复验收图工具、P1、P2 或完整脑诊断；本结论不代表 M4 真实动作研究已完成。

## Spec Coverage

- `execute_experiment` 使用从执行起点计算的登记总 deadline，在普通 fit、急性 P 计算及其保存、逐项聚合评分、配对计算与最终保存前后检查预算。最后一次配对或最后一次写入耗尽预算也会进入失败终态，不再返回 `complete`。
- `TimeoutError` 从内部任务分支直接抛到外层终态处理，停止后续拟合、评分和配对。正在执行的阶段允许返回；失败记录是超时后允许进行的必要收尾。
- `run_failed.json` 明确保存 `status=incomplete`、`reason=total_budget_exhausted`、耗尽阶段、耗时、错误和完整 replay/fit 任务清单，`comparisons` 为空。已完成任务保持可追踪，当前运行任务变为 `failed`，剩余 `pending` 任务变为 `not_executed`。
- 若最终 `run_summary.json` 写入才超时，处理分支改写本次已生成的总报告为 `incomplete` 并清空比较，同时记录失败；此前局部模型、预测和报告没有被删除。
- 返工局限于调度预算及失败收尾，不改图构建、P1/P2 数据契约、拟合/评分定义、九候选搜索或登记 bootstrap 次数。现有普通独立拟合与急性 P 的专用固定 G 读出路径保持分离，没有为满足预算省略条件或减少计算次数。
- [使用文档](../../docs/EXPERIMENT_V2.md) 对阶段边界检查、不可中断阶段、完整任务状态、空比较、最终报告纠正及局部产物保留的描述与实现一致。

## Validation

- **Risk Level：S1**。只审查一个调度函数及其必要局部辅助的预算闭环，按照 [返工要求](rework-p3.md) 限定范围。
- 阅读项目约束、验收流程、Spec/执行计划、相关 Decision、原 FAIL、返工要求与 [实施报告](implementation-p3-r1.md)，逐项检查当前调度源码、六个参数化案例及对应文档。
- 采纳实施者已完成的唯一相关检查：`tests/test_experiment_v2.py::test_total_budget_closes_after_computation_and_persistence`，**6 passed，0.44 s**。本验收没有无理由重跑。
- 六个案例分别在普通 fit 保存、聚合评分、急性计算、急性保存、最后一次配对以及最终总报告保存后，将受控时钟由登记 100 s 内推进至 1,000 s；断言覆盖错误阶段、空比较、全部六项 replay/fit 任务、剩余任务不执行、独立 fit 调用数、局部产物保留及最终报告不残留 `complete`。
- 本次只写本验收报告；未修改业务代码、未运行测试/build/lint、未执行真实脑/真实录像/硬件、未提交。

## Issues

无直接相关的未解决问题。P3-01 已关闭。

## Regression Risks

预算采用阶段边界检查，不是强制抢占单次 NumPy、急性计算、评分、bootstrap 或文件写入；实际墙钟可能超过登记预算，结果必须标为未完成。该限制已明确记录，不以硬实时能力描述。局部产物只能作为未完成执行的审计材料，不能代替完整实验比较。

## Required Rework

无。满足本次局部验收停止条件，不追加前序检查或实验。
