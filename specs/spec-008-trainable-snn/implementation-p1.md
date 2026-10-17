# Implementation Report — P1 探针工具与首次登记

## Summary

已实现身份约束 checkpoint、父进程时间/内存监督和固定完整图一次内部更新协议；16 项定向检查通过。2026-10-17 实际登记首轮后，启动前可用内存为 3,674,931,200 bytes，不满足系统预留 4 GiB，工具按契约写入 `resource_preflight_failed`，没有启动模型子进程。**P1 数值可行性尚未完成，不能勾选完整图更新/恢复/重载验收项。**

## Files Changed

- `src/cnsbridge/learning/checkpoint.py`：新目录保存 JSON 与非 pickle NPZ，核对参数字节、模型/参考图/类型分组/输入集合身份、精度、形状、有限值，提供生产重载与明确夹具恢复。
- `src/cnsbridge/learning/resource_guard.py`：资源预算、父进程短周期采样、进程树终止、原子 JSON 状态、超时/RSS/系统预留/异常/不完整报告等失败终态。
- `src/cnsbridge/learning/probe.py`：登记先于启动，固定 R1-R6→L1 / 200 步协议、内部 Adam 一步、相同输入前后/恢复/重载、原始图摘要复查、阶段与数值产物。
- `tests/test_training_probe.py`：16 项持久化与监督边界检查；小电路保持 `numerical_test_only`，不作为完整图证据。
- `docs/TRAINABLE_SNN.md`：P1 命令、资源和证据契约，以及首轮未启动事实。
- 本实施报告。

未修改 P0 核心、完整图资产、既有设备环境或其他并行批次文件；未提交。

## Important Decisions

- P0 核心未作推测优化。没有实际全图计算，因此尚无稀疏伴随或状态检查热点实测。
- 父进程不导入 Torch/图模块；子进程启动至退出均在预算内，包含导入、载图、模型/优化器构造、前向、反向、保存与恢复。4 GiB 系统余量同时参与启动前和运行中检查。
- 上限为采样监督，不承诺不可突破的硬隔离；原始图只读，计算流程不裁剪任何边或神经元。
- checkpoint 仅保存内部 theta；不保存未声明的输入/输出学习参数。生产重载只接受 `full_reference`，小图不能从测试通道进入生产。
- 一次内部学习结果需非零实际 gain 更新、下游响应改变、恢复和 checkpoint 比较、原始图不变共同支持；损失是否下降单独公开，不作为隐藏筛选条件。

## Validation

- Risk Level：S3，训练持久化/资源监督/完整图集成；仅执行本批定向检查与一次协议登记。
- `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_training_probe.py -q`：**16 passed，1.30 s**；1 条既有 PyTorch CSR beta 提示。覆盖精确参数恢复、坏字节/模式/精度/输入/分组/形状/非有限参数拒绝且不修改模型、生产拒绝夹具、不可覆盖旧目录、资源预检不启动子进程、正常/非零退出/坏报告/checkpoint 空壳、超时和 RSS 终止。
- `PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.probe --graph data/malecns_v2/full_reference.npz --out data/trainable_probe_20261017_p1`：**exit 1 / resource_preflight_failed**。真实登记 UTC `2026-10-17T15:20:09.308261+00:00`，available=3,674,931,200 bytes，reserve=4,294,967,296 bytes，effective RSS budget=0。保留 `registration.json` 和 `result.json`，worker=`not_launched`。未执行完整图前向、反向、参数更新、恢复或重载；不声称机器算力已被实测否定。
- 没有重复已通过 P0/旧全脑检查，没有全项目测试、build、lint、硬件或付费训练。

## Known Limitations

当前内存条件阻止首轮计算；16 项工具检查不能证明完整图路径在实际规模可用。尚无实际全图加载成本、反向速度、峰值 RSS、内部更新和恢复证据，也没有目标跟随学习或飞行结果。工具可先独立审查；完整 P1 必须等资源改变后重新登记实测。

## Remaining Questions

root 已通知当前内存不足并处理外部条件；本实施不擅自退出用户应用或放宽预留。是否以及何时有可用内存变化由 root 协调，不能自动重试。独立验收应区分工具实现审查与尚未完成的 P1 实际学习验收。
