# Implementation Report — 完整图前向内核基准

## Summary

按 [Ready 子规格](forward-kernel-benchmark.md) 新增单个离线实验脚本。生产模型及稀疏内核保持原状；本次只完成脚本，不包含完整图性能结果。

## Files Changed

- `examples/fullgraph_matvec_benchmark.py`：固定协议、轻量预登记、既有资源监督、完整图加载、共享 SciPy CSR 构造、交错计时与原始证据。
- 本实施报告。

## Important Decisions

- 父进程只使用标准库、psutil 和既有 `resource_guard`；不导入 NumPy、SciPy、Torch 或模型。先在全新目录保存源码快照、源码 SHA、完整图文件 SHA、协议身份和 1 GiB 预留预算，再启动 worker。
- worker 在重型导入前核对登记、预算、当前源码、源码快照与图文件；独占 `worker.claim` 防止重复 worker 覆盖旧证据。额外 1 GiB 预留、最多 6 GiB RSS、600 秒、0.1 秒采样复用现行监督，没有管理 Noria / Docker 的逻辑。
- `load_training_graph` 和现行 `TrainableLIF` 构造全量权重；CPU float32、Torch 4 / inter-op 1 线程、0.275 mV，模型输入组记录为 R1–R6。图摘要严格使用 TrainingGraph 的 body IDs、CSC indptr、indices、data 顺序及 `<i8` 编码。
- SciPy `csr_array` 一次构造，记录三个底层数组的共享内存、dtype、shape、stride、地址、顺序和值、前后 SHA；不复制为另一套图，不修改数组。
- 种子 1901、全零 / 恰好 floor(N/100) 个 1 / 同一生成器随后产生的稠密 uniform 输入；每输入每方法一次预热及五轮交错计时。SciPy 计时含 NumPy / Torch 桥接，输出复制与落盘在计时外。
- 保存全部输入、选择索引、两方法各五轮输出、预热输出、三十条时序记录及每输入中位数、数值比较。各已完成输入增量保存，资源失败不自动重跑。
- `result.json` 保持监督终态；`worker_result.json` 单列数值比较和候选标志。正常采集结束可为 `protocol_complete`，即使某项数值比较或速度不满足候选条件；候选必须三输入均通过固定容差且各 SciPy 中位数严格更低，并满足所有完整性检查。此标志仍需独立证据审查，不是生产替换许可。

## Validation

- Risk Level: S2（离线实验与身份、资源记录交互；未改生产核心）。
- Checks Performed: 阅读新增脚本；唯一执行检查为 `PYTHONPATH=src .venv-train/bin/python examples/fullgraph_matvec_benchmark.py --help`。
- Result: 帮助命令退出码 0；未加载图、未做模型计算、未启动 benchmark、未运行完整测试套件。新增文件不出现在普通 tracked diff 中，已直接检查内容；协调者正在编辑的 Spec / handoff 未由本实施者覆盖。

协调者在软件独立验收与提交后可执行一次：

```sh
PYTHONPATH=src .venv-train/bin/python examples/fullgraph_matvec_benchmark.py \
  --graph data/malecns_v2/full_reference.npz \
  --out data/forward_kernel_20261019_r1
```

## Known Limitations

- 没有实际速度、共享存储或全图数值结果；`--help` 仅验证入口能在轻量父进程中启动。
- 此实验只比较固定矩阵乘向量，不能证明完整 LIF 脉冲轨迹、梯度、训练速度或设备实时性。
- 资源监督为现行采样机制，不是操作系统硬限额。调用者应保持不并行运行其他完整图实验；脚本不停止其他项目进程。
- 非有限值保留在原始 NPZ；JSON 差异统计使用 null 且数值条件为 false，避免丢失负结果。
- 源码快照是当前基准及关键模型加载链的身份记录，未复制 Python 环境或整个仓库；包版本和 Torch build 配置由 worker 另存。

## Remaining Questions

无新增产品或协议决策。等待新的独立软件验收；本实施者不提交，不出最终验收 PASS。
