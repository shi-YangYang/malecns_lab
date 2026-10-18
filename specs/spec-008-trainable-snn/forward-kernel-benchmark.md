# 完整图前向内核候选实测

状态：软件已独立[PASS](acceptance-forward-kernel-benchmark.md)，协调者提交后运行B/C。此子批只比较同一完整权重的计算成本，生产模型仍使用已有 Torch 实现。

## 目标与范围

当前完整图前向约16.5s/200神经步。先比较 Torch CSR 乘向量与共享其数组的 SciPy `csr_array` 乘向量，再决定是否替换生产前向。不能从内核计时推断完整动力学、学习能力或实时性。

实施位置：`examples/fullgraph_matvec_benchmark.py`，作为可重复的离线性能实验；复用 `learning.resource_guard`，不新增业务框架、不修改生产模型、不引入依赖。报告及本子规格在当前 Spec 下，实际数组/采样/登记在新的 `data/forward_kernel_20261019_r1/`。

## 冻结协议

- 完整 `data/malecns_v2/full_reference.npz`，经 `load_training_graph` 验证；用现行 TrainableLIF 构造固定权重，全部节点和边保留。float32 CPU、Torch 4线程/inter-op 1线程、原0.275mV尺度；不训练、不飞行。
- SciPy `csr_array((values.numpy(), col_indices.numpy(), crow_indices.numpy()), shape=..., copy=False)` 只在计时循环前构造；逐项核实 data/indices/indptr 与 Torch NumPy 视图共享内存、dtype和顺序不变，不修改任一数组。
- 固定三个长度N输入：全零；种子1901的 NumPy default_rng 选择恰好1%节点（floor(N/100)）赋值1；同一生成器随后生成 uniform(-1,1) 的float32稠密向量。保存三个输入及每种实现结果。
- Torch路径 `torch.sparse.mm(W, x[:,None])[:,0]`；SciPy路径包含每次 `x.detach().numpy()` 和 `torch.from_numpy(csr @ ...)` 的桥接成本。每路径每输入先1次预热，随后每输入5轮交错计时；偶数轮Torch先，奇数轮SciPy先。使用perf_counter，不计输入构造及输出复制/落盘。
- 保存所有原始时间、每输入中位数、最大绝对差/RMS差、`rtol=1e-5, atol=1e-5`逐元素比较、输出dtype/shape/有限性。容差不用于证明脉冲轨迹等价；不因失败放宽。三输入均通过且各中位耗时SciPy更低才进入生产候选；否则保留负结果并定位。
- 构造耗时、图/模型身份、源码SHA、包版本、线程配置、完整原始计数前后摘要、共享数组与dtype证据单独记录。保存输出和原始数组是证据，不以自报pass代替独立审查。
- 先由轻量父进程在全新目录登记固定协议、脚本/关键模型源码SHA与预算，再启动子进程导入Torch/加载图。子进程核对登记身份。采用额外1GiB预留、最多6GiB进程树RSS、600秒、0.1秒采样；保持Noria/Docker，不并行全图作业。终态区分监督完成、数值比较、性能候选，资源失败不自动重跑。

## 分工和验证

风险S2：离线实验脚本复用已验收监督，不改生产核心。实施者只负责脚本与实施报告，可做轻量 `--help` 检查；不得在实施期执行完整图。新的独立验收者审查协议实现与资源边界，软件PASS后协调者提交并运行一次。实际结果另由新验收者核对原始数组、时间与监督记录，不重复已通过运行。

## 验收

- [x] A：脚本实现以上固定协议，独立软件审查通过；随本状态更新提交。
- [ ] B：新目录完整图实际运行有可追踪终态，原图不变，数值和资源完整报告。
- [ ] C：独立核对实际证据，根据结果决定下一步，不把核级通过当完整模型通过。

若进入生产候选，须另立数值等价子规格并验证完整P1及带背景动力学；清理被替代实现，不形成永久配置分叉。整体AC3–AC6仍未完成。
