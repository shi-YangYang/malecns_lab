# Acceptance — 完整图伴随内核与线程实测 r1

## Result

PASS

新的独立实际验收者核对 [Ready 协议](backward-kernel-benchmark.md)、[软件 PASS](acceptance-backward-kernel-software.md)、[实施报告](implementation-backward-kernel.md)、[本次运行报告](backward-kernel-run-20261019-r1.md)与 `data/adjoint_kernel_20261019_r1` 的保存证据。B、C 的一次完整图运行、原始数值和候选判断满足本批范围。SciPy 及 Torch 8/10 线程是核级候选；**本结论不批准直接改写既有训练来源，也不代表整个训练已提速或模型已学会控制**。

## Spec Coverage

- **完整图、身份及来源**：独立重算 registration、protocol、manifest、model、training graph 的 canonical ID；42 个 manifest 文件清单与实际目录精确一致且 SHA 全部通过，worker 的产物摘要亦逐个核对。16 份登记源码的快照与当前文件摘要一致，实际导入来源由登记闭包覆盖，原模型两份源码身份一致。母图文件只做流式 SHA，得到 `98f83158ed12f0c836fb085c2e286e54f9458630c2a158bee3164af52648f6f3`，没有解码其完整矩阵。worker/model 保存的图身份与协议一致：166700 节点、25582938 边、124177617 突触计数、full_reference，逻辑图和原始计数摘要匹配。
- **伴随矩阵及共享存储证据**：绑定源码实际使用原 `TrainableLIF.base_weights_adjoint`，不是正向 W 或筛选后的发放细胞子图。检查保存的 166700×166700、25582938 项、float32 data / int64 indices / indptr、全部地址/shape/stride/只读及顺序字段，Torch/SciPy 前后摘要、指针和输入摘要自洽。原图有序计数前后 SHA 同为 `4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565`。**这些内存地址、共享关系和原图前后摘要是受绑定 worker 的运行记录；本验收没有重建矩阵、重验已退出进程的活指针或独立重算逻辑图。** 独立文件 SHA、输入和输出重算与此证据层次分开。
- **输入与固定调用顺序**：独立使用 NumPy default_rng(1901) 重建完整 zero、无放回选择的 1667 个 sparse 位置及其顺序、同一生成器随后生成的 float32 uniform dense。与 `inputs.npz` 逐元素精确一致，前后原始数组摘要一致；所有 NPZ registration/protocol 身份正确。核对 1/2/4/8/10 线程、每种输入一次预热及五轮交错调用的全部 180 条次序，interop 始终记录为 1，终态恢复 4 线程。
- **独立数值比较**：直接读取 15 个输出 NPZ 中全部 180 个长度 166700 的保存向量，验证每条时间记录对应的输出 SHA、float32、shape 和有限性。重新计算每轮 SciPy/Torch、Torch/Torch4、SciPy/Torch4 共 270 组比较，包括预热；全部逐元素精确相同，max-abs 与 RMS 均为 0。固定 rtol=atol=1e−5 未放宽；结果与 worker 全部比较字段一致，没有仅信任其 pass 布尔值。
- **计时与候选**：从原始记录重新计算全部 30 个方法/case 的五次墙钟与进程 CPU 时间中位数及其比，核对每次 CPU 秒/墙钟比，全部与保存摘要一致。SciPy 在 4 线程设置块的 zero/sparse/dense 耗时比分别为 `0.2190635369840131`、`0.2190833600895497`、`0.2178072899936428`，对应核级吞吐约 `4.5649 / 4.5645 / 4.5912` 倍。三输入均更快，满足预登记候选门槛。Torch 8/4 比为 `0.791880–0.809290`、10/4 比为 `0.805406–0.813091`，两者也满足门槛；1/2 线程均较慢，4 线程是自身基线，不列为候选。没有丢弃负结果或使用最好单次结果。
- **资源与终态**：父 `completed / worker_exited / returncode 0`，worker `protocol_complete / kernel_evidence_recorded`，manifest 中父结果与实际文件一致。预算按 available 4638179328 − 1 GiB，得到 RSS 上限 3564437504 bytes；6 GiB 申请、600 秒、0.1 秒采样设置保留。重新汇总 125 行资源 CSV：峰值 RSS 1481129984 bytes，最低 available 3770482688 bytes，均满足预算；父墙钟 13.376378916997055 秒，worker 13.100856083998224 秒。源码中同一工作区 OS 锁覆盖登记与终态，未见设备、Noria 或 Docker 操作路径。
- **报告口径**：运行报告全部 15 case 的舍入中位数、CPU 时间比、资源数值及身份与上述证据一致。明确区分现场 CPU/系统 swap 短采样与保存资源 CSV，反向/裁剪总耗时与孤立 SpMV，构造时间与包含摘要检查的相位时间，合成数值向量与真实训练上游梯度；没有将 4.6 倍解释为端到端训练或 GPU 结果。

## Validation

- Risk Level: **S2**，仅本次实际数值证据的独立验收。
- Evidence Reviewed / Checks Performed: 读取项目治理、当前 Spec/Plan、任务交接、Decision 011/012、Ready 子协议及软件/实施报告；核对当前绑定脚本的矩阵方向、固定计算调用、计时范围、摘要和来源登记语义。
- 唯一数值审查命令：`.venv-train/bin/python data/adjoint_kernel_review_20261019_r1/review.py`，exit 0，约 0.258 秒。独立实现使用标准库与 NumPy，从保存数据重算；没有导入 Torch/SciPy/项目模型，没有运行 matvec、模型构造或旧软件测试。
- 审查脚本、结果和其 SHA 清单位于独立同级目录 `data/adjoint_kernel_review_20261019_r1/`，原始实验目录完全只读。结果 `review.json` 记录 manifest/source 复核、180 条记录、270 组比较、30 组中位数、候选和资源汇总，未写入封存运行目录。
- 未重复已经通过的软件测试、没有全项目检查、没有生产代码修改或 commit。本验收不是模型训练验收。

## Issues

没有阻止本批通过的未解决问题。共享缓存运行记录与独立可重算数组的证据层次已在本报告明确；已向协调者反馈同样口径供运行报告引用。

## Regression Risks

固定线程块顺序、每 case 五轮和单次执行受后台负载影响，不能据此选出 8/10 的唯一赢家。合成向量不代表真实上游梯度，CPU 时间比不代表整机利用率。完整自定义 autograd、静默细胞梯度、两帧 DN/动作/原始与裁剪梯度/Adam/参数/末态等价、旧运行来源迁移和端到端成本均未在本批验证。当前生产反向、G/frozen/R 注册和训练线程保持原实现；本次核级通过只支持下一批单独设计上述验证。

## Required Rework

无。满足本批实际验收范围，停止检查；整体 AC3–AC6 继续开放。
