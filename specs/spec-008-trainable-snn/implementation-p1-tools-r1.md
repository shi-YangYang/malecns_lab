# Implementation Report — P1a 证据验证返工 R1

## Summary

针对原独立验收的“退出码 0 加任意 `status=passed` 即误报成功”缺口，分离进程完成与 P1 数值成功。通用监督仅返回 `completed`；P1 父进程必须通过登记绑定、完整产物与逐项数值一致性校验后才能写 `passed`。本次只完成软件返工，没有运行完整图。

## Files Changed

- `learning/resource_guard.py`：正常进程退出改为 `completed`，不再从 worker 的状态字符串推断研究结果。
- `learning/probe_evidence.py`：独立轻量校验模块。检查 v2 登记/结果模式、登记摘要、固定协议、资源配置、当前实现摘要、完整参考图角色与模型身份、阶段时间、产物 SHA256、数组精度/维度/有限性和 checkpoint 关联；复算一步 Adam、内部增益、梯度/参数/响应统计、损失及恢复判据。仅使用标准库与 NumPy，不导入 Torch、加载完整图或重新仿真。
- `learning/probe.py`：v2 登记产生稳定 `registration_id` 与 `protocol_id`；worker 与数值产物绑定同一身份，保存初始 theta 和全部产物摘要；父进程最终确认实际证据。损失报告统一从保存的 float32 响应用 float64 归约，优化目标与反向计算不变。
- `learning/checkpoint.py`：保存可选 `run_identity`，纳入已有 checkpoint 元数据摘要。既有加载验证、参数结构和默认调用方式保留。
- `tests/test_probe_evidence.py`：完整软件证据夹具及有具体缺口依据的拒绝回归。
- `tests/test_training_probe.py`：原状态字符串“complete”案例改为监督层完成语义，明确永不据此返回 `passed`。
- `docs/TRAINABLE_SNN.md`：补充两级完成语义、v2 证据验证与首轮原始文件保留说明。

## Important Decisions

- 通过需要真实存在的全部数值产物、匹配的身份/摘要和由数组复算的一致结论；缺失或矛盾证据落为 `invalid_probe_evidence` 并记录具体异常。一个伪造的布尔值不能替代恢复向量或更新数组。
- 非零内部梯度、实际 theta/gain 变化、下游响应变化、归零恢复、checkpoint 响应匹配与原始图摘要不变必须同时成立；损失下降独立记录，仍不作为隐藏筛选条件。
- 完整测试夹具仅为手写的小型产物束，带明确 `software_fixture_only` 标记；其完整参考图形状元数据不能被生产图加载器加载，不是任何真实脑、训练或全图运行证据。生产验证仍拒绝 `numerical_test_only` 图角色。
- 证据校验能发现文件不完整、跨运行混用和内部矛盾，不能仅凭本地产物证明人为伪造的数据真实发生过。真实图来源继续由登记 worker 的生产完整图加载器建立。
- 不修改 P0 核心、固定刺激/优化目标/数值参数、600 秒/6 GiB/4 GiB 预算或首轮 `data/trainable_probe_20261017_p1/`。原 FAIL 与失败资产保留。

## Validation

- Risk Level：S2，关联的监督、探针及证据读写模块交互。
- `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_probe_evidence.py tests/test_training_probe.py -k 'evidence or registration or supervisor_completion' -q`：**30 passed，12 deselected，1.04 s**。一次执行即通过。
- 覆盖完整软件证据加干净进程退出；错误 schema、失败恢复声明、原图变化、缺 checkpoint/响应、登记/产物/checkpoint 身份错配、错误统计、NaN/错误精度/维度、陈旧摘要、真实恢复向量不匹配、错误 Adam 更新、缺阶段、测试图角色及内部一致但无实际更新的拒绝；登记协议/实现/预算改变；裸状态字符串与监督失败不可升级；轻量模块不导入 Torch/完整图加载器。
- `git diff --check` 通过；检查当前文档差异。未重复 P0、12 项本次未改动的原 checkpoint/资源检查、旧完整脑检查或全项目测试。
- 未启动真实完整图探针、硬件、付费任务；没有 commit。

## Known Limitations

本次 30 项通过仅为软件证据。P1 实际完整图更新、恢复、资源耗时与下游任务学习仍未完成。完整图运行须由 root 在用户确认内存释放且新独立验收通过后，在新目录登记执行。

## Remaining Questions

请 root 安排新的独立验收者审查本轮返工；P1a 可以单独提交，但不能勾选 P1 实际更新或总 Spec AC2。
