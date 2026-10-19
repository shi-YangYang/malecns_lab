# Implementation Report — GPU 感觉相位上界返工

## Summary

针对 [独立验收 FAIL 的 R1](acceptance-gpu-foundations.md)，仅修复 float32 舍入使合法满增量返回 `phase=1` 的状态边界。保留原相加、发放与减去一个事件的顺序，将减法后的舍入上界限制为 float32 `nextbelow(1)`，即 `0.9999999403953552`。

`phase=nextbelow(1)`、`dt_ms=1`、`max_rate_hz=1000`、占有率 1 时，预期每步恰发放一次事件，剩余相位保持 `nextbelow(1)`；不会额外发放事件或缩小合法配置范围。原有限性、范围和设备校验保持，未使用 `nan_to_num`。

## Files Changed

- `src/cnsbridge/senses/gpu_drive.py`：对事件减法结果应用明确的相位上界，并将该数值规则登记到 `event_rule`。来源哈希及规则共同产生新的 drive ID，旧身份不能静默恢复。
- `tests/test_gpu_foundations.py`：新增一个直接覆盖 R1 的定向测试，检查零、四分之一、二分之一、四分之三及 `nextbelow(1)` 相位，在满增量下连续 3 步、第一步后快照恢复、每步单事件、完整增量、输入状态与快照不变。
- 本实施报告。没有改动优化器、旧来源文件、旧资产或其他 Agent 的文件；没有提交。

## Important Decisions

- 只处理已发现的残余相位舍入上界，保持原运算顺序与正常区间结果，避免为了此边界改动默认感觉数值序列。
- 这是一条新的显式 float32 状态规则，仍不宣称与旧 float64 感觉事件逐位等价。
- 数值检查必须等协调者释放当前完整图诊断占用，并使用已有 `workspace_lock()` 串行执行。

## Validation

- Risk Level：**S1**，单个函数的明确数值边界修复。
- 静态实现及差异检查已完成。协调者明确通知完整图诊断进程退出、释放数值时隙后，测试 fixture 持有 `workspace_lock()` 执行单个检查。
- 实际唯一命令：`PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_gpu_foundations.py::test_gpu_drive_full_increment_phase_boundary_continues_and_restores -q -s`。
- session **89133**，已收取 exit **0**：**1 passed in 1.05s**。连续 3 步及第 1 步后的恢复结果一致，每个输入累计恰好 3 个事件，最大相位 `0.9999999403953552`，传入状态与快照不变。
- 本 fixture 的新 drive ID：`1cd2c9f02113b5678bed2ae3eeef3c8fe1b59a03c91622054c1a90ce8eb9916e`。这不是完整连接组训练身份。
- 不重复原两个已通过的基础测试，不运行优化器、完整图、全项目测试或其他验证。

## Known Limitations

本返工只修复感觉状态闭包的已知边界，不能证明完整图训练、学习有效性或实体控制结果。最终验收由新的独立 Agent 执行。

## Remaining Questions

无待用户决策事项；定向检查通过后停止实施，等待新的独立验收。
