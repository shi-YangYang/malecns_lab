# Implementation Report — P3d 终态准入返工 R1

## Summary

按 Ready [返工范围](rework-p3d-r1.md)修复独立 [FAIL](acceptance-p3d.md) 的两项证据边界。原实施报告及 FAIL 保留；本报告不代替新的独立验收，不宣称完成生产评估或 AC3–AC6。

训练最终参数准入现在要求所有段都有已验证的父终态，并将每段父结果及完整实际终态文件 SHA 绑定进评估来源。评估另外记录 `evidence_complete`，将环境终止与正式场景证据完成分开；资源/IO 失败的终局不会开放下个场景、test 或完整比较。

## Files Changed

- `src/cnsbridge/learning/evaluation_contract.py`：新增正式训练终态准入；核对父/worker/run 身份、原始 supervisor、实际进度及父前缀、完整文件清单/SHA、完成检查与允许的历史资源/IO失败；最终段要求 completed。来源和条件绑定全部 `finalized_segments`。
- `src/cnsbridge/learning/evaluation_evidence.py`：正式完成字段统一驱动 episode report、冻结顺序、val/test 完成、等权均值与配对差值；失败终局保留 terminal、原始段证据和 diagnostic_metrics，正式 success 为 false。补父结果身份/完成原因核对。
- `src/cnsbridge/learning/evaluate_snn.py`：只有当前父进程通过完成检查后才将本段终局标为正式完成；失败段不升级。
- `tests/test_evaluation_snn.py`：补准入和失败终局交叉反例；更新直接受影响的完成判据 fixture。
- `docs/EVALUATION_SNN.md`：说明正式训练准入、terminal/evidence_complete、诊断评分与失败终局不能继续前向的边界。
- 本实施报告。未改模型、训练、环境、评分核心、数据、预算、原验收报告或协调者的 Plan/Handoff。

## Important Decisions

- `training_evidence.scan_run` 仍用于验证数值和唯一链；新增评估准入层进一步验证父终态，不修改当前训练绑定源码，也不把恢复扫描器的 160 窗返回值当正式完成。
- 中间资源/IO失败的已验证前缀允许后续显式续段最终完成；保留原 failed 与全部原始来源。缺父 result、无效证据、数值/协议失败、身份或前缀不符、最终失败段拒绝准入。
- `terminal` 保留物理环境事实；`evidence_complete` 要求其终局段有父验证的 completed 结果。正常封存的物理失败仍是正式研究终局，但行为 success 为 false。
- 对资源/IO失败终局不增加恢复放行流程，不再次前向，不自动重试。非终局失败前缀仍按原协议显式继续。局部原始评分仅在 diagnostic_metrics 中表明诊断语义，不能借其 success 解锁正式比较。

## Validation

- Risk Level: **S3**，范围限新评估子系统的持久证据、准入和共享完成状态；未运行整个项目或旧训练套件。
- 协调者先收取生产 session **33707** 的 exit 0、确认模型进程列表为空并释放测试窗口；返工实施者随后执行：

  ```bash
  PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_evaluation_snn.py -q -k 'final_training_admission or environment_terminal or terminal_resource_failure or nonterminal_io_prefix or resource_failed_prefix or all_validation_order or physical_failure_is_terminal or empty_failure or parent_imports_no_torch'
  ```

  session **45460** 已收取 **exit 0**：**25 passed、20 deselected，4.58 秒**。只有既有 Sparse CSR beta 警告。通过后未重复执行；已通知协调者可恢复生产训练，无存活测试/模型作业。
- 20 个新增用例：合法 160 窗来源的 40 个父终态、历史资源及 IO 失败后 completed 续训；缺末父、坏父证据、缺 hash、父/worker身份错、数值失败、原始监督不符、最终失败、前缀错；duration/物理终局 × completed/资源/IO 六种交叉；失败终局阻断顺序/val/test/均值/配对；非终局 IO 前缀显式续段完成且原失败原字节不变。
- 5 个受影响原检查：父入口不导入 Torch；资源前缀及数值失败；全部 val 门槛/封存；物理失败的 80 帧分母；空失败/假完成。其余 20 项旧通过检查未重跑。
- 准入用例明确隔离已由旧组件验证的 scanner 和最终数组读取，使用磁盘上的 40 段父/worker/监督/清单，真实执行新增终态准入及已有完成检查；**不是 160 窗真实训练，也不是生产后备模型**。评估用例使用明确小数值电路/零基线的真实环境和封存链。
- 使用标准库 SHA-256 与当前 `data/neural_training_joint_20261019_r1/run.json` 的登记清单核对：**29 个绑定训练源码，mismatches=[]**。未导入 Torch或运行完整图。
- 已阅读本轮修改及工作区范围。既有 `.ai/prompts/task-handoff.md`、`p3c-continuation-20261019.md` 的修改属于协调者，本实施者未编辑。原 FAIL/实施报告保持原样。

## Known Limitations

两种真实最终 160 窗训练尚未同时完成，本批未登记或执行完整图评估。软件反例通过不代表目标跟随、内部学习收益、拓扑贡献、实时性或实飞能力。失败终局若将来需要专门复核恢复，仍须另立明确协议，当前工具只拒绝自动升级。

## Remaining Questions

本返工范围无已知待确认项；请新的独立验收者复核 R1/R2。未 commit，由协调者在独立验收后统一提交。
