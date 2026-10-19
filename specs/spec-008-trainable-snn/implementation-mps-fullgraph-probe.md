# Implementation Report — 完整连接组 MPS 探针 A

2026-10-19；依据 [Ready A 协议](mps-fullgraph-probe.md)。本报告是实施记录，不是独立验收。

## Summary

新增受监督、独占、不可覆盖的完整图 CPU/MPS 稀疏算子探针。尚未执行完整图实验；不修改神经模型、训练后端、旧 run 或任何既有源码。

## Files Changed

- `examples/fullgraph_mps_benchmark.py`：轻量父进程、固定协议、以新入口为根的保守 AST 源码闭包、快照/运行环境/图文件身份、资源登记、防覆盖与最终清单。
- `examples/mps_benchmark_worker.py`：父子关系及单次 claim、完整 full_reference 图/原始计数检查、原 TrainableLIF 构造、MPS 双向缓存、固定实验和证据汇总。
- `examples/mps_benchmark_kernel.py`：CSR→COO 转换与分块回读精确校验、固定 72 次调度、同步计时、逐次保存、差异指标、固定矩阵全维输入 autograd。
- `examples/mps_benchmark_memory.py`：独立 0.1 秒及阶段采样、current/driver 两种 sampled peak、recommended memory 和严格向下舍入的 allocator cap。
- `tests/test_mps_benchmark.py`：4 项与新增探针直接相关的检查。

## Important Decisions

- 只复用原 helper；不替换旧模块函数、协议全局变量或源码。新源闭包包含新 worker 的惰性导入与所有实际本地依赖，worker 再核对实际导入。
- CPU 前向直接使用原模型共享 SciPy CSR，CPU 伴随使用原 Torch CSR；MPS 两方向使用完整驻留 COO。COO 临时 CPU 副本逐方向释放，原始图和原模型 CSR 保留；禁止 NxN 稠密化。
- 固定输入沿用 `make_inputs(N,1901)`；每输入依次 forward、adjoint，每方向 1 次预热、5 次正式 CPU/MPS 调用。所有 72 次输出分别落盘，失败保留已完成证据；转换、传输、回读、写盘与计时分开。
- 自定义 autograd 只保存固定 WT，不申请逐边梯度；输入和上游梯度使用已登记 dense 向量，明确保存两者及全部输出/输入梯度。它不能证明 SNN BPTT 或 Adam 正确。
- 数值比较以 float64 统计，保持预登记 `rtol=1e-5, atol=1e-5`；零参考而差异非零的相对误差标为未定义，不加任意 epsilon。包含零支持与非零符号差数。
- 父进程仍按进程树 RSS 与系统可用量监督，MPS allocator cap 另设为不高于有效 RSS 预算/推荐上限。统一内存指标不相加；采样失败使 worker 失败，不宣称硬限额。
- `protocol_complete` 只表示固定协议完成；数值准入、性能候选分别输出。任何未满足条件均不会自动切换训练。

## Validation

- Risk Level: S2，隔离离线实验工具，多模块的登记/监督/数值/采样交互。
- 所有涉及数值的检查持有 `data/.continuous-training.lock`；root 已留出独占槽。没有完整图模型加载，没有 Noria/Docker 操作。
- 第一次：`PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_mps_benchmark.py -q`：3 passed，1 failed。失败是最终 cap 断言：浮点乘法得到预算 + 0.0000005 byte；该次 72 次 tiny MPS 数值、调度、显式反向已通过。
- 将 fraction 用 `math.nextafter(..., 0.0)` 向下舍入后，仅复跑失败项 `... -m pytest tests/test_mps_benchmark.py::test_tiny_signed_nonsymmetric_mps_schedule_autograd_and_memory -q`：1 passed，1.13s。
- 累计 4 个不同检查通过：轻量导入及新源闭包/CLI、失败证据/独占/防覆盖/fallback 拒绝、真实资产文件字节 SHA 登记及快照/协议篡改拒绝、200×200 明确非脑代数的 signed/non-symmetric MPS 双向与 autograd/采样检查。没有重复已通过的 3 项。
- tiny 检查保留 PyTorch 稀疏 invariant 默认提示与 CSR beta 提示，不代表错误。生产模型构造仍使用原有 `check_invariants=True`。
- `git diff` 无既有文件内容变化；新文件逐项阅读。未执行全项目测试/build/lint，未 commit。

## Known Limitations

完整图 MPS 的内存、速度、误差结果仍未知。两方向中位数总和仅是 kernel 成本，不能称为训练加速；自定义 autograd 验证不能替代后续完整训练窗口准入。受监督进程可能在构造或某个算子阶段触及资源/算子限制，届时保留负结果，不裁图或扩大容差。

## Remaining Questions

无实施层面待决定问题。独立软件验收后，由 root 在新目录运行一次：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  examples/fullgraph_mps_benchmark.py \
  --graph data/malecns_v2/full_reference.npz \
  --out data/mps_fullgraph_20261019_r1
```

本实施者未运行该命令。B/C 仍由 A 实测及独立复核的门槛决定。
