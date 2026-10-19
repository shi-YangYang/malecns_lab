# 完整连接组 MPS 成本、数值与条件迁移

状态：A 阶段软件和一次完整图测量已完成；速度/资源通过，数值门槛未通过，B/C 未启动。2026-10-19。以下保留原登记条件，不追改容差。

## 背景与范围

用户明确要求先测完整连接组在当前 Mac GPU 上的内存、速度、CPU 数值差异，验证没有问题后切换训练。已有 2×2 诊断只证明本机 Torch 2.14.1 的 MPS COO 前向/输入反向可运行，CSR 不支持，不能外推完整脑。PyTorch 仍是训练框架；本批不实施 SciPy 反向候选。

保留完整 full_reference：166700 节点、25582938 有向非零边、124177617 绝对突触计数；文件 SHA `98f83158ed12f0c836fb085c2e286e54f9458630c2a158bee3164af52648f6f3`，逻辑图 SHA `af12b8dc8775f828740221f92c870d76450266759bca6f1540ff5e27002ac156`，原始有序计数 SHA `4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565`。不裁图、不稠密化、不改权重精度或动力学。

## A：完整图稀疏算子实测

新增隔离的 examples 入口和核辅助模块；复用原完整图 loader、TrainableLIF 构造与 resource_guard。原 G29/R13、前向/反向历史基准源码保持字节不变，不改旧 run。

- float32，Torch CPU 4 线程/interop 1；权重系数 0.275。直接从原模型 `base_weights` 与 `base_weights_adjoint` 构造 canonical COO，两个方向均保留全部坐标与值。GPU 缓存一次构建，记录转换和传输耗时；逐块回读坐标/值，与 CPU COO 精确核对，原图计数前后摘要相同。
- CPU 前向为现行共享 SciPy CSR，CPU 伴随为现行 Torch CSR；GPU 两方向为驻留 MPS COO 的 `torch.sparse.mm`。禁止 CPU fallback，记录环境 `PYTORCH_ENABLE_MPS_FALLBACK=0`；参数/输入/输出实际设备须检查。
- 输入固定为原伴随基准 `make_inputs(N,1901)` 的 zero、1% sparse、dense。每方向/输入/方法 1 次预热 + 5 次正式调用，偶数轮 CPU 后 GPU、奇数反序，预热 CPU 后 GPU，共 72 次。保存所有输出；GPU 计时前后 synchronize，排除输入传输、结果回读和存档；另报告这些成本，不能混称端到端训练速度。
- 在相同完整 W/WT 缓存上，再执行一次显式固定矩阵自定义 autograd：dense 输入、dense 上游梯度，前向 W x，反向 WT grad，不计算或申请逐边梯度。保存并与现行 CPU 两方向结果核对，覆盖完整 N 维输入梯度。它不是神经网络 BPTT 或 Adam 验证。
- 保存每次输出与 CPU 对应结果的 exact、max-abs、RMS、相对 L2/Linf、超容差元素数、零支持差数、非零符号差数。以 float64 计算统计；预登记逐元素 `rtol=1e-5, atol=1e-5`、形状/float32/全部有限。三种输入包含抵消，不因结果好坏挑选。
- A 数值通过需要全部输出/自定义 autograd 比较及图/缓存/输入/来源完整性通过。性能候选需要三种输入各自 GPU 前向+伴随中位总时间均严格小于现行 CPU 对应总时间；单方向退化单列。不通过则不迁移，不放宽容差或删失败输入。

## 资源与证据

轻量父进程先注册协议、源码闭包和运行环境，在独占 `data/.continuous-training.lock` 下用新目录运行一次；拒绝覆盖和重用已消费 worker。完整图/模型仅在受监督 worker 加载。资源遵循申请 6GiB RSS、额外系统预留 1GiB、实际 RSS 上限 min(6GiB, available−1GiB)、600s/0.1s。不操作 Noria/Docker。

MPS 单独记录 current_allocated_memory、driver_allocated_memory、recommended_max_memory、set_per_process_memory_fraction 的实际预算。GPU allocator cap 不高于已登记有效 RSS 预算且不超过 recommended max；同时仍由父进程监测系统可用内存与进程树 RSS。worker 以 0.1s 周期及关键阶段记录 MPS 两种用量和采样峰值；采样失败如实失败，不伪称峰值。统一内存各指标口径不同，不把 RSS 与 driver 直接相加。这些监控不构成 OS 硬限额保证。

保存资源 CSV、GPU 内存 CSV、原始计时、输入/全部输出/梯度、实际设备、转换校验、图与模型身份、源码快照/摘要、worker/父进程终态、产物清单。异常/资源中止保留部分证据。parent completed 不是科学通过。

## B/C：条件训练迁移

A 通过后，才建立保持原动力学、输入映射和读出的 MPS 候选，使用新的真实后端与源码身份，验证完整两帧/1000 步训练窗口的 DN/动作/神经末态、离散发放/不应期、原始和裁剪梯度、Adam 与参数增量，以及实际时间与资源。固定容差、比较锚点和候选接口在运行 B 前登记；不能将 A 容差直接当作 SNN 递推充分标准。

B 通过且完整窗口比 CPU 更快才执行 C：建立关联旧终态 checkpoint 的新训练 run，保留所有参数、Adam 矩、递归状态、帧游标和已完成训练预算，验证跨进程恢复后继续一小段。现有 joint60/160、frozen10/160 不是可重置的进度。不得覆盖旧记录/伪造旧 backend，不开启硬件。条件失败只报告失败和具体阻碍，保持现有 CPU 训练路径。

## 实施与验收

A 为 S2：隔离离线探针，模块定向检查，不运行项目全套。一个实施 Agent 负责新增脚本和相关最小检查，新的独立验收 Agent 静态核查及已有软件证据后，由协调者执行一次完整图；独立实际验收复核保存证据，不重跑模型。阶段接受后中文 conventional commit。

- [x] A 软件：完整图、固定比较、资源与来源约束正确实现，见[软件独立 PASS](acceptance-mps-probe-software.md)，提交 b4b06de。
- [x] A 实测：保存真实内存、速度、差异和候选结论，见[一次实测](mps-fullgraph-run-20261019-r1.md)及[独立记录验收 PASS](acceptance-mps-fullgraph-run-r1.md)。A 数值门槛 FAIL，性能更快不能覆盖此失败。
- [ ] B：仅 A 通过后，完整训练窗口数值与速度准入。
- [ ] C：仅 B 通过后，保留训练状态的 GPU 迁移与实际续训。

无待用户决定的问题；具体工程接口由证据决定，当前用户授权覆盖条件满足后的迁移。
