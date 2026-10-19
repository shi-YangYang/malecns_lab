# Decision 013 — 完整连接组 GPU 验证后迁移

## Context

2026-10-19 用户要求先测完整连接组在当前 Mac GPU 的内存、速度和 CPU 数值差异，验证没有问题后切到 GPU 训练。此前训练框架一直是 PyTorch，现行前向稀疏乘法为 SciPy CPU，反向为 Torch CPU。2×2 MPS COO 通过仅是兼容性线索。

## Decision

按 [MPS 完整图协议](../../specs/spec-008-trainable-snn/mps-fullgraph-probe.md)执行完整算子、完整训练窗口和保留状态迁移三个阶段。用户授权覆盖门槛满足后的实际迁移，无需再次索取同一授权；门槛失败则保留负结果和 CPU 路径。

## Reason

GPU 支持一种稀疏格式不等于完整 SNN 数值可靠、内存足够或训练更快。必须同步计时，分别报告 RSS、MPS 分配/驱动和系统可用内存，并检验递推、梯度及 Adam 更新。

## Alternatives

已有 SciPy 反向审计暂留 Draft，不作为本次实施方向；直接修改旧注册后端或重置训练进度不采用。

## Consequences

完整拓扑、float32、输入映射和学习语义保持；G29/R13 及历史性能脚本保持字节不变。若可迁移，建立真实新源码/后端身份和关联旧 checkpoint 的新 run，保留参数、Adam、全部递归状态和游标。1GiB 系统额外预留与 Noria/Docker 保留要求不变。不启动无人机。
