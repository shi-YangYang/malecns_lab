# 反向成本只读审计

状态：Draft，2026-10-19。本文件是静态审计及候选后续门槛，不是实施报告或性能结论。当前拟实施范围以[反向内核基准](backward-kernel-benchmark.md)为准；本审计未运行 Torch、数值测试、完整图或训练，未改变 G29/R13 绑定源码。

## 已核实的代码事实

| 位置 | 事实及解释边界 |
|---|---|
| `src/cnsbridge/brain/trainable.py:83–99` | `_FixedSparseMatvec.forward` 为 `torch.from_numpy(matrix @ vector.detach().numpy())`；backward 为 `torch.sparse.mm(adjoint, gradient[:, None])[:, 0]`。只有密集输入有梯度，固定矩阵不学习。 |
| `trainable.py:140–163` | 正向 Torch CSR 与伴随 Torch CSR 已在构造期建立；伴随直接来自原 CSC 的缓冲顺序，当前反向没有逐步 transpose/转格式。SciPy 只读共享视图目前只覆盖正向。 |
| `trainable.py:225–256` | 每神经步保持完整状态、代理脉冲及延迟队列；线性输入是 `gains[group_index] * spikes`，不是只枚举非零脉冲。反向必须保留全部 N 个输入分量，包括静默细胞。 |
| `src/cnsbridge/learning/training_window.py:55–75` | `backward_clip_seconds` 从目标/损失构造前开始，包含 `loss.backward()`、参数梯度 NumPy 复制及裁剪。约 79–86 秒是该组合阶段，不是已测的单个 SpMV 或全部稀疏内核累计时间。 |
| `training_window.py:66–69`、`joint_probe_worker.py:143–148` | 保存的是 theta/readout 参数梯度；现有 `gradients_before_clip.npz` 或 `evidence.npz` 不能直接称作传入 `_FixedSparseMatvec.backward` 的 N 维上游梯度。 |
| `src/cnsbridge/learning/training_worker.py:117–118`、`training_contract.py:87–91` | 当前训练固定 CPU float32、Torch intra-op 4、interop 1。CPU 状态约束也见 `trainable.py:206–231`。GPU 未参与是实现事实，不是对 MPS 硬件全部能力的判定。 |
| `training_contract.py:50–58,171`、`training_checkpoint.py:60–64,227–228` | 现有 run 与完整训练 checkpoint 绑定源码及线程配置。直接改核心或线程后继续旧 run 会破坏来源/恢复契约，运行结束也不会解除历史绑定。 |

本轮协调者提供的现场采样为：10 核 M5，活动 worker 三秒窗口 CPU 约 266.4%，每两帧各 50 ms、共 1000 神经步，前向约 20 秒。该 CPU 数值约相当于同时使用 2.66 个核心，不等于整机 266.4% 或资源必然不足；单次样本也不是稳定利用率。审计者未另行重采进程，避免把历史记录当成第二份实测。

## 有依据的候选与尚未验证的机制

- 旧[正向内核实验](forward-kernel-run-20261019-r1.md)在同一机器配置下测得 Torch 约 74.6 ms、SciPy 约 15.3 ms，三个合成向量结果精确相等。它支持试测相同 API 在伴随方向的成本；转置后行分布不同，不能直接把该倍数应用到当前反向。
- 当前全窗有顺序相关的递归状态和大量重复的稀疏矩阵乘向量。CPU 稀疏遍历、内存访问、行负载不均、小算子调度以及线程开销都可能限制利用率。未做本轮内核计时/采样归因前，不判断哪个占主导，也不把 80 秒全部解释成 Torch SpMV。
- 旧内核报告保存的构建信息为 Torch 2.14.1、Accelerate BLAS、`USE_MKL=OFF`、`USE_OPENMP=ON`；这些是既有运行环境记录，不是本轮新导入检测。此前交接中关于 native CSR 实现的分析可作线索，不替代当前伴随方向实测。
- 4 线程为现有契约，增加线程是否更快尚未知；尤其不能只追求活动监视器百分比更高。

## 当前最小有用实验

采用[反向内核基准 Draft](backward-kernel-benchmark.md)，只测 `base_weights_adjoint` 的 Torch CSR 与共享同一 data/indices/indptr 的 SciPy CSR。固定 1/2/4/8/10 线程、interop 1，三个 zero/恰好 1%/稠密合成数值向量；每方法每输入预热 1 次，再交错计时 5 次。总计 180 次稀疏乘向量调用，不含构造，不运行神经场景。SciPy 路径须包含每次 NumPy 视图与 Torch 包装成本。

现有 `examples/fullgraph_matvec_benchmark.py` 可复用的边界：

- 43–63：纯标准库 canonical/file/source hash 辅助函数；新脚本仍须登记自己的完整实际导入闭包，不能照搬旧脚本身份。
- 66–131：先登记后受监督启动、独占新目录、来源快照与预算；新脚本需遵守当前工作区 OS 锁。
- 176–218：原始 counts 摘要、矩阵构造和共享数组/dtype/地址核验。
- 220–270：固定输入、交错计时、每 case 保存输出及独立数值比较。

保留旧脚本及 G29/R13 字节；新 `examples/fullgraph_adjoint_benchmark.py` 复用稳定辅助能力，不复制训练器。固定 `rtol=1e-5, atol=1e-5` 只用于本批量级约 1 的合成输入，记录所有 exact/max_abs/RMS 差。所有线程输出均与 4 线程参照比较，完整保留负结果。SciPy 必须数值全过且 4 线程三输入中位时间全部低于 Torch 才称核级候选；线程设置只作为单独核级候选报告，不立即改训练。

预算仍为一次新登记、600 秒最大墙钟、进程树 RSS `min(6 GiB, available−1 GiB)`、0.1 秒采样、1 GiB 系统额外预留。等待当前 G 段退出，持有 `data/.continuous-training.lock`，保留 Noria/Docker；构图和所有数值工作均在受监督 worker。时间和 RSS 是上限，不是预测；不预报实际加速。墙钟与 process CPU 时间同时保存，不能把 CPU 秒/墙钟误称为整机占用率。

本批为隔离离线脚本 S2；候选若进入核心将是 S3，须另立可实施规格。当前小实验满足后停止，不自动扩大为训练调参或后端迁移。

## 候选通过后的独立门槛

以下不属于当前实施范围，不现在运行；必须先冻结来源、预算和比较规则。

1. **真实上游向量。** 在隔离的实验 worker 中，预登记完整 G 首窗的两帧输入/初值/空白预处理，保持原 Torch 4 线程数值路径。使用仅观察的输出 Tensor hook 或等价显式实验接线记录真正传给伴随算子的梯度；观察器返回原梯度，不修改生产源码。可预先固定反向调用序号 0/249/499/749，记录实际神经步、dtype、stride、幅度/非零计数及完整 N 维数组。缺失就报告缺失，不改挑非零或“典型”向量；这些是新隔离诊断采集，不能冒充来自原训练 run。挂钩/复制有开销，采集窗口时间不得直接称作无观测器训练时间。
2. **真实向量的内核比较。** 使用保存的精确上游值原尺度比较；不只按当前 `atol=1e-5` 判定微小梯度，否则把小梯度全清零也可能通过。可预登记先要求逐位相等；不相等则保留原始差异，再按新独立数值规格判断，不事后放宽容差。零/稀疏/稠密探针仍标记为合成覆盖。
3. **完整一阶反向语义。** 小型独立代数覆盖有符号非对称边、自环/孤立节点、float32/float64、零输入但非零上游梯度及非连续输入。现有 `tests/test_trainable_core.py:122–199` 已提供合适的独立 dense 参考和递归静默梯度检查。候选须证明所有输入分量、增益/代理梯度和延迟语义一致；不能以“未发放”删梯度。CPU NumPy 路径会切断高阶自动微分，当前训练使用普通一阶 `backward()`，未发现 `create_graph=True` 调用，文档也未承诺二阶能力；候选必须显式维持这一边界。
4. **完整两帧窗口。** 保持每帧 50 ms、完整背景/感觉映射、全部 1000 神经步及既有损失/裁剪/Adam 次序。基线与候选从完全相同的参数、Adam、LIF `v/g/refractory/pending/step`、感觉相位/时钟和输入开始，串行执行，不把量化/重排/删边与算子替换混在一起。先核对全部 DN 特征、动作、loss、原始及裁剪后各参数梯度、全部 Adam 矩与更新参数，以及全部末态。初值/离散状态/输入要求 exact；当前首窗对照已冻结 `rtol=1e-6, atol=1e-7`（`training_math.py:154–174`），应沿用并报告 exact/maxdiff，完整状态差异另按预登记规则判定，不能只比较动作。前向未变且未更新前的状态应 exact；所有有效脉冲/时钟差异必须单列，不能用浮点 allclose 隐去。
5. **来源与采用。** 数值/性能通过仍只是该隔离窗口的证据，需保留实验专属实现身份；不能伪装成现有 G checkpoint 或重写旧 run。核心采用、历史恢复、新 run 起点与后续状态连续性另行明确后，才可改新训练路径。不能直接给旧 G/frozen/R 换线程或伴随实现继续。

## 本次检查与停止

风险 S0：仅新增本审计文档，阅读相关源码、冻结协议和既有报告，检查本文件差异；未跑测试、build、lint、Torch/NumPy 数值导入、完整图或训练，没有 commit。本轮结论是值得测伴随内核和线程数，不是已经找到性能主因或已经提速。
