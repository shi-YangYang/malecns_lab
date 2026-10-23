# P1 真实脑遥测实施

2026-10-23。风险 S3（冻结数值旁路、新版本数据归档与监督执行接线）。本报告仅描述 Python 遥测；不代替完整图性能/等价和桌面联调验收。

## 已实施

- `learning/x2_telemetry.py` 从已验证 `graph.identity.reference_provenance.group_mapping` 取得22组的全部真实成员；按组名排序，保留完整分母。默认从参考组的33个输出神经元按body ID升序取前32个，manifest明确选样范围，不声称全脑脉冲记录。
- `x2_frozen._step` 在既有数值运算后只读实际 `spikes`，不重建脉冲、不改变阈值/状态/动作计算。私有GPU计数向量与50×32采样缓冲独立于模型；每50步在现有sticky检查通过后聚合群组，与逐步采样拼接，恰好一次 `.cpu()`。0.1ms步长下每批是5ms神经时间，不是宿主时间。
- 独立线程仅持有CPU副本；队列最多8批，满时立即丢弃当前批并计数。单批256KiB、manifest1MiB、整个归档16MiB。原子写入 `manifest.json`、`batches/000001.json`、完整 `latest.json` 和 `summary.json`。发生采样/存储异常时停止采集并报告，不使数值循环等待磁盘。关闭最多等待写线程2秒。
- summary包含 `complete`、`batch_count`、`produced_batches`、`dropped_batches`、`errors`、`total_bytes`。非正常结束、丢批、写失败、未flush尾批都不能标记完整；完整性与数值worker成功分开。
- `x2_neural --telemetry` 和 `x2_inference --backend optimized --telemetry` 显式启用。注册记录此开关和当前遥测源码hash；旧run、checkpoint、历史源码均未改。预览入口仍只导入标准库。
- 每次forward前设置真实观测序号，脉冲记录的 `decision_sequence=null`；其时刻尚未产生动作，之后通过已有decision文件关联。precondition记录单独phase且观测序号为空。

主要修改文件：`learning/x2_frozen.py`、`learning/x2_telemetry.py`、`learning/x2_session.py`、`learning/x2_inference.py`、`experiments/x2_neural.py`、`experiments/x2_inference.py`、`tests/test_x2_telemetry.py`。

## 已执行检查

一次定向检查：

```sh
.venv-train/bin/python -m pytest -q tests/test_x2_telemetry.py tests/test_x2_frozen.py tests/test_x2_inference.py::test_default_entry_and_import_are_inert tests/test_x2_inference.py::test_optimized_cli_is_explicit_and_preview_stays_inert tests/test_x2_neural.py::test_failed_supervisor_publishes_abort_without_host_ack
```

结果40 passed。覆盖CPU与MPS小张量的真实脉冲/组率/步号/一次CPU拷贝，旁路前后局部step精确一致，队列背压、磁盘失败、归档预算、异常尾批、预览惰性导入和原有frozen监测接线。它们是小型注入与传输检查，不是完整脑等价证据。

随后完善summary即时错误发布及精确总字节数，只重跑受影响 `tests/test_x2_telemetry.py`：10 passed（0.61s）。无全仓测试、训练、完整脑或仿真启动。

## 有界完整图配对入口（尚未运行）

由协调者在独立审阅后串行执行；两个目录必须为新的目录。使用已完成的原始baseline，不重跑它：

```sh
.venv-train/bin/python -m malecns_lab.experiments.x2_inference --execute --backend optimized --baseline data/x2_neural_inference_20261023_r1 --output data/x2_telemetry_off_20261023_r1
.venv-train/bin/python -m malecns_lab.experiments.x2_inference --execute --backend optimized --baseline data/x2_neural_inference_20261023_r1 --telemetry --output data/x2_telemetry_on_20261023_r1
```

每次沿用完整脑锁、监督进程、1GiB系统预留和240秒上限；入口预检要求GPU VM/CPU仿真停止。每次600步预热+3×500步输入，保存4个全部递归边界及3个完整输出，与原始baseline逐元素精确比较；on正常完整采集应有42批。

配对验收还必须比较两次相同当前wrapper/runtime/input注册身份；两者各自全部状态/输出与同一baseline精确相同才构成on/off精确一致。计算耗时为 `timing.json` 的precondition+三次forward之和，on/off增幅≤15%；`result.json.peak_observed_rss_bytes` 增量≤128MiB。on的 `telemetry/summary.json.complete` 必须true。现有旧优化性能门槛（相对原baseline至少20%改善）仍保留，不能替代此次15%遥测开销门槛。

目前未执行上述完整图配对，没有声称实际性能门槛通过。32个神经元以外没有逐步事件；缺坐标、显示端跳批/延迟由桌面单独呈现。桌面显示限频不改变这里的每批归档。
