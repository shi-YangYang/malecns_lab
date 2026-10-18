# Implementation Report — 系统额外预留 1 GiB

## Summary

按用户明确要求及 [资源调整子规格](resource-reserve-1g.md)，新运行默认额外预留 1 GiB。P1 登记、实际监督参数与独立证据校验一致；配对诊断默认及最低预留为 1 GiB，仍允许协议显式提高预留。保留 6 GiB RSS 封顶、600 秒墙钟及 0.1 秒采样，不修改神经模型、图、优化目标或历史运行。

## Files Changed

- `src/cnsbridge/learning/resource_guard.py`：共享 `DEFAULT_SYSTEM_RESERVE_BYTES`，预算计算与监督默认值共同使用。
- `src/cnsbridge/learning/probe.py`：从登记预算传入预留、墙钟及采样间隔，预检失败说明反映实际登记预留。
- `src/cnsbridge/learning/probe_evidence.py`：新 P1 登记严格绑定共享的 1 GiB 规则。
- `src/cnsbridge/diagnostics/observability_protocol.py`：模板及协议最低预留共用资源常量。
- `tests/test_training_probe.py`、`tests/test_probe_evidence.py`、`tests/test_observability.py`：调整已有边界/协议检查，补充登记到监督的参数传递及运行中默认预留边界覆盖。
- `docs/TRAINABLE_SNN.md`、`docs/OBSERVABILITY.md`：当前操作说明更新为 1 GiB；历史 4 GiB 首次登记段落及原始记录未改。

## Important Decisions

遵循 [Decision 012](../../.ai/decisions/012-one-gib-system-reserve.md)。生产 P1 不接受任意更低或更高的预留；配对诊断协议允许更高预留。通用资源监督仍保留显式参数接口，现有轻量测试可明确传入独立测试预算；生产入口传入已登记协议。Noria、Docker 及其数据未操作。

## Validation

- Risk Level: S2（同一资源策略涉及登记、监督、证据验证与诊断协议）。
- Checks Performed: `git diff --check` 与改动 diff 阅读；一次定向 pytest：

```sh
PYTHONPATH=src .venv-train/bin/python -m pytest \
  tests/test_training_probe.py tests/test_probe_evidence.py tests/test_observability.py \
  -k 'resource_budget or preflight or supervision_uses_registered or supervisor_default_reserve or registration_binds or windows_and_budget or fixed_masks_and_step_ledger or clean_exit_or_resource_failure or supervision_checks_original or timeout_terminates or rss_stop or supervisor_completion or status_only_report or evidence_module_does_not_import or parent_and_validator_import' -q
```

- Result: **41 passed, 66 deselected in 1.26s**；diff 检查通过。覆盖 1 GiB 临界值、available 减预留、6 GiB 封顶、P1 严格登记与监督传参、诊断 1 GiB/更高预留及拒绝更低预留、RSS/系统余量/超时失败终态和轻量导入。
- 没有运行完整测试套件、模型数值测试、checkpoint 测试、配对采集或完整图计算。

## Known Limitations

本子批只改变资源契约；尚无 1 GiB 预留下实际完整图学习结果。采样监督不是操作系统硬隔离，预算调整不保证训练能完成。后续由协调者独立验收后提交，并在新输出目录运行 P1。

## Remaining Questions

无待决实施问题。实际计算资源峰值与数值结果待新登记运行测量。
