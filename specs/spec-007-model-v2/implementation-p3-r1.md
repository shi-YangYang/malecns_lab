# Implementation Report — P3-01 总预算闭环返工

## Summary

仅修复 [P3 集成验收](acceptance-p3-integration.md) 的 P3-01：`execute_experiment` 在每个计算/保存阶段边界检查登记的总墙钟 deadline；最后一次比较或最终保存超时也不能得到完整实验结果。原验收其他通过项不变，交新的独立验收者复核本缺口。

## Files Changed

- `src/cnsbridge/learning/experiment_v2.py`：局部阶段预算检查、超时退出、任务终态及最终报告纠正。
- `tests/test_experiment_v2.py`：一个参数化预算回归检查，共六个阶段边界案例。
- `docs/EXPERIMENT_V2.md`：说明阶段检查、失败产物及不可中断计算的处理。
- 本报告。

## Important Decisions

- deadline 仍从执行开始计算，登记预算、拟合算法、九候选、评分定义、bootstrap 次数、图与独立拟合规则均未改变。
- 急性 P 计算前后、普通 fit 返回、两种保存之后、逐项聚合评分、每次配对 bootstrap 以及最终报告写入前后均检查 deadline；数据准备、共同条件检查及重放也在阶段边界检查。
- 当前不可中断计算允许返回。超时通过 `TimeoutError` 立即退出调度，不进入后续拟合、评分或比较；只进行必要失败记录。
- `run_failed.json` 保留全部 replay/fit 任务和 `stage`、`reason=total_budget_exhausted`、耗时及错误。已完成任务保留，尚未开始任务为 `not_executed`，当前未完成任务为 `failed`；已有文件不删除。
- 若最终 `run_summary.json` 写入本身越过 deadline，仅改写本次刚生成的总报告为 `incomplete`、清空 `comparisons`，同时保存失败报告，避免磁盘残留一个看似成功的结果。

## Validation

- **Risk Level：S1**，单个编排函数及局部辅助的预算状态修复，没有修改 P1/P2、图计算或科学评估定义。
- 仅执行一次最相关检查：

  ```bash
  .venv/bin/python -m pytest tests/test_experiment_v2.py::test_total_budget_closes_after_computation_and_persistence -q
  ```

- **结果：6 passed，0.44 s。** 可控时钟从登记 100 秒内跳至 1,000 秒；覆盖普通 fit 保存、聚合评分、急性 P 计算、急性 P 保存、最后一次配对比较、最终总报告保存。
- 检查确认全部登记任务保留，未开始任务明确不执行；后续独立 fit 不再发生；比较为空；已写局部拟合/急性文件保留；最终保存超时没有残留 `complete` 总报告。
- 已阅读改动及文档。没有重跑前序测试、整个测试套件、lint/build、真实脑、真实录像或硬件；没有提交。

## Known Limitations

阶段检查不强行打断正在执行的单次 NumPy/评分/bootstrap/文件写入，因此实际墙钟可能超过预算；超时后状态和结果被明确判为未完成，而非假称硬实时中断。失败记录及必要的最终报告纠正不受已耗尽的计算预算阻止。

## Remaining Questions

无新增用户决策。等待新的独立验收者审查 P3-01；本报告不是最终验收结论。
