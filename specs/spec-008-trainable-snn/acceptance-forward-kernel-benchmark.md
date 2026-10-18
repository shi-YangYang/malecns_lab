# Acceptance — 完整图前向内核基准软件

## Result

PASS

本次为未参与实施的新 Agent 对 [冻结协议](forward-kernel-benchmark.md) A 的独立软件审查。通过仅说明脚本可按该协议进入一次受监督实测；A 的提交由协调者完成，B/C 的实际运行与证据审查尚未发生。本报告不宣称 SciPy 已提速、实际共享成功或完整模型数值等价。

## Spec Coverage

- 完整图与同一权重：直接使用现有 `load_training_graph` 和 `TrainableLIF`，没有子图、弱边过滤、回退或生产模型修改。CPU float32、4 / 1 Torch 线程、0.275 mV 均绑定固定协议。原始 body IDs 与 CSC indptr/indices/data 按 `<i8` 顺序计算前后摘要，并与已有图身份校验；磁盘图文件也有前后 SHA。
- 共享 CSR：在计时前只构造一次 `csr_array`，来源为 Torch 固定权重的 NumPy 视图。逐项记录 data / indices / indptr 的共享内存、dtype、shape、stride、地址、值与顺序、前后 SHA；不满足存储契约立即结束，不把复制或降精度作为候选。
- 确定输入：种子 1901；无放回选择恰好 `N // 100` 个位置赋 1；稠密输入由同一生成器在 choice 后生成 uniform(-1,1) 并转换 float32；同时保存输入和选择位置。全零输入独立保留。
- 计时：每输入每方法一次预热，随后五轮按偶数 Torch 优先、奇数 SciPy 优先交错。SciPy 计时表达式包含 `detach().numpy()` 及 `torch.from_numpy`，输出复制与保存发生在计时外。三十条原始顺序/时间、全部五轮输出及预热输出保存，可独立复算中位数和数值比较。
- 数值与候选边界：固定 `rtol=1e-5, atol=1e-5`，比较全部元素，记录 dtype / shape / 有限性 / 最大绝对差 / RMS。仅三输入均通过、各 SciPy 中位数严格更低且图/权重/源码完整性成立时标为候选；数值或速度负结果不会伪装生产许可。监督终态、协议采集完成与候选标志分开。
- 登记与资源：父进程导入链只有标准库、psutil 和既有轻量 `resource_guard`；在 worker 重型导入前保存协议、源码快照及 SHA、图文件 SHA 与预算。新目录 `exist_ok=False` 和独占 `worker.claim` 防止重复覆盖。worker 验证登记、当前源码、快照、图文件与固定预算，再加载 NumPy / SciPy / Torch。
- 运行边界：复用现有进程树监督，额外 1 GiB、最多 6 GiB RSS、600 秒、0.1 秒采样不变；没有 Noria / Docker 管理、设备连接、训练更新、自动重跑或并行完整图启动逻辑。维持不并行其他全图作业由协调者在实际启动前确认。

## Validation

- Risk Level: S2，离线实验、身份及资源监督交互；生产核心未变。
- Evidence Reviewed / Checks Performed: 直接逐段阅读新增脚本（含未跟踪文件），核对 Ready Spec、实施报告、当前 handoff diff、`resource_guard.py`、轻量包导入、`trainable_graph.py` 及 `TrainableLIF` 权重构造；读取项目规则、实施目标与相关 Decision 009 / 011 / 012。
- 实施者已记录唯一 `--help` 检查成功，本次没有无理由重跑；没有加载完整图、执行 benchmark、运行测试套件或修改实现。
- 脚本保存的 NPZ、原始计时、身份、摘要与监督 CSV 足以供实际运行后的独立检查；真实输出是否符合协议仍须 B/C 审查，不能由本次静态审查代替。

## Issues

无阻断软件验收的问题。

## Regression Risks

- 采样监督是既有观察机制，不是操作系统硬内存隔离。
- 五轮交错样本只支持本机本次内核比较，不能推出稳定的整体训练倍数；浮点容差内相等也不证明阈值动力学或梯度等价。
- 当前软件通过不覆盖后续改生产前向；如实际结果支持候选，仍按另立的数值契约检查完整 P1 和背景动力学。

## Required Rework

无。协调者处理本报告并提交软件步骤后，可按固定协议在新目录运行一次；不追加已通过软件检查。
