# 设计与实测记录

[完整脑到X2的纵向反馈MVP](spec-021-x2-neural-loop/implementation.md)已完成两轮实测并[独立PASS](spec-021-x2-neural-loop/acceptance.md)：实际前进0.419m和0.429m，逐轮水平停稳，模型参数未变，仿真已关闭。[冻结推理优化](spec-020-x2-frozen-optimization/acceptance.md)已独立PASS：完整状态与输出精确一致，计算耗时下降62.3%。MVP使用实际里程计驱动虚拟标记；尚未验证真实相机、学习目标跟随或真机。日常操作从[文档索引](../docs/README.md)进入。

本目录保留阶段设计、独有数值协议和实际实验报告，删除重复实施、返工和验收流水。历史报告描述当时状态，不是当前待办清单；过程文件中已经提交的版本可从 Git 历史查看，未提交且被精简的过程说明不保证另有副本。原始数据、参数、源码快照和日志没有删除。

## 保留的阶段设计

| 阶段 | 范围与当前定位 |
|---|---|
| [001](spec-001-agent-workflow/spec.md)、[002](spec-002-business-layout/spec.md)、[003](spec-003-remove-compat/spec.md) | 已完成的工程流程、目录与入口整理 |
| [004](spec-004-neural-activity/spec.md)、[005](spec-005-remove-minifly/spec.md)、[006](spec-006-malecns-audit/spec.md) | 神经活动诊断、删除合成网络及历史数据审计 |
| [007](spec-007-model-v2/spec.md) | 完整参考图、V2 诊断与冻结脑研究工具 |
| [008](spec-008-trainable-snn/spec.md) | 完整图可训练 SNN、三轴跟随、GPU 与研究对照；保留实现及实测，不自动续训 |
| [009](spec-009-malecns-lab/spec.md) | 已完成的项目与 Python 包改名 |
| [010](spec-010-neural-atlas/spec.md)、[011](spec-011-interactive-atlas/spec.md) | 保留的 WebGL 图谱与交互刺激响应 |
| [012](spec-012-yaw-closed-loop/spec.md) | 历史单轴固定读出实验；无一致控制优势，网页入口已移除 |
| [013](spec-013-yaw-learning/spec.md) | 历史单轴学习：两分支各 80 窗完成、固定模型验证失败；后续支线已停止 |

## 关键实测

- [完整脑到X2两轮纵向反馈](spec-021-x2-neural-loop/implementation.md)。

- [完整脑冻结推理优化](spec-020-x2-frozen-optimization/implementation.md)。

- [完整脑冻结推理基线](spec-019-x2-neural-inference/implementation.md)。

- [X2适配器前进与反馈停稳](spec-018-x2-motion-adapter/implementation.md)。

- [X2 速度与停车离线核对](spec-017-x2-motion-contract/implementation.md)。

- [官方 X2 站立、前进与停止](spec-016-x2-motion-mvp/implementation.md)。

- [完整参考图构建](spec-007-model-v2/implementation-p0.md)。
- [CPU/MPS 数值差异](spec-008-trainable-snn/mps-fullgraph-run-20261019-r1.md)。
- [持续输入响应](spec-008-trainable-snn/sustained-observability-run-20261019-r1.md)。
- [全 GPU 160 窗训练](spec-008-trainable-snn/gpu-recovery-run-20261019.md)。
- [交互刺激实测](spec-011-interactive-atlas/actual-run-20261021.md)。
- [单轴基线](spec-012-yaw-closed-loop/actual-run-20261021.md)、[单轴学习](spec-013-yaw-learning/actual-run-20261021.md)、[固定模型验证失败](spec-013-yaw-learning/acceptance-validation.md)。

源码直接引用的 MPS 数值先例报告保留原路径与原文。新的改动按 [AGENTS.md](../AGENTS.md) 和 [_template](_template/) 使用与风险相称的记录与验证。
