# 完整图反向稀疏运算与线程数成本对照

状态：Ready，已通过[独立协议审查](acceptance-backward-kernel-protocol.md)。仅离线内核成本实验，不修改当前训练的模型、线程、来源或游标。

## 背景和目标

用户观察活动监视器中CPU/GPU利用率偏低，要求判断当前训练是否过慢。2026-10-19当前完整图joint反向阶段3秒采样：worker CPU约266.4%（100%=一个核心），Mac为10核；实际登记CPUfloat32、Torch4线程/interop1。最新已完成四窗的两帧前向约20s、反向/裁剪约79–86s。源码前向使用共享SciPy CSR，反向仍为Torch CSR乘向量；当前GPU未参与，不据此声称MPS硬件不能做任何此类任务。

先测同一完整伴随矩阵的Torch/SciPy运算及CPU线程设置，判断算子选择和线程数的实际影响。不预设CPU占满就是更快，不把合成测试向量称为训练中采集的真实梯度，不从内核吞吐直接宣布完整训练或部署速度。

## 固定输入、矩阵与方法

- 完整`data/malecns_v2/full_reference.npz`，通过现有`load_training_graph`和`TrainableLIF`构造同一float32权重；保留166700节点/25582938边/124177617计数。原始权重尺度0.275，实际`base_weights_adjoint`为W转置的CSR，使用当前原构造实现。不得把正向W的成绩误当反向。
- 从该Torch伴随CSR的values/col_indices/crow_indices构造唯一共享存储SciPy `csr_array`，在计时前完成；核实全部指针、shape、dtype、顺序/数值以及只读视图，记录构造时间和完整数组摘要前后不变。
- 三个长度N输入固定顺序zero/sparse/dense。seed1901，NumPy default_rng；zero全零，sparse恰好floor(N/100)个位置为1（无放回choice），随后同一生成器生成uniform(-1,1)float32稠密向量。保存实际输入/选中索引。这些是数值探针，不能解释成生物刺激或真实反向梯度分布。
- Torch线程依次为1、2、4、8、10，interop始终1；每个线程块显式调用set_num_threads，设置与必要预热不计入计时。每个输入/方法先预热1次，然后测5次；偶数轮Torch先、奇数轮SciPy先。固定顺序全部报告，不能挑最好一轮或因慢而跳过某个线程块。结束恢复4线程，并记录配置。
- Torch路径：`torch.sparse.mm(adjoint, x[:,None])[:,0]`；SciPy路径：`torch.from_numpy(shared_adjoint @ x.detach().numpy())`。计时包含每次输入NumPy视图和结果Torch包装，矩阵共享视图只构建一次；无GPU、无稠密N×N矩阵、无自动线程搜索。
- 计时使用perf_counter和process_time，分别保存每轮墙钟和进程CPU秒；报告中位耗时及CPU秒/墙钟比，不把它当成全系统CPU百分比。保留每轮/预热输出及dtype/shape/有限性。每个case保存后释放，不把所有结果和多个完整模型同时常驻。

## 数值和候选规则

对所有输入、线程、重复输出按元素比较Torch/SciPy以及Torch各线程相对4线程的结果；固定`rtol=1e-5, atol=1e-5`，记录exact、max_abs/RMS差与失败项。全零输入也完整计算，稀疏输入的零值位置不剪掉任何伴随输出。

SciPy候选门槛：全部数值检查通过，4线程设置下三类输入的SciPy中位时间都严格小于原Torch4线程对应时间。线程候选分别报告相对Torch4线程三输入的时间比；只有三输入数值合格且都更快才称该线程设置为核级候选。未通过则保留负结果，不放宽容差或替换数据。本文不选择或部署训练后端。

核级通过只允许设计后续完整反向等价试验。任何后续候选必须验证实际自定义autograd边界、包括静默细胞的全细胞梯度、完整两帧窗口的DN/动作/原始与裁剪梯度/Adam/参数和全部末态，并与已保存G首窗对照；不能改变神经时间窗或只验非零发放细胞。当前训练用普通一阶backward；不承诺NumPy候选支持高阶自动微分。

## 来源、资源和运行纪律

- 新例程`examples/fullgraph_adjoint_benchmark.py`；复用原前向实验的纯canonical/file-hash辅助函数，独立登记本次协议、`resource_guard`和当前工作区OS锁。历史前向脚本、G29、R13均不改；若实际实现需要调整复用边界，先在Plan中说明，不复制整个训练器。
- 新目录先由轻量父进程登记固定协议、图文件SHA、脚本及全部实际导入依赖来源SHA和快照、runtime；父进程不得导入Torch或加载图矩阵。worker核对登记、加载后核对并记录图逻辑身份，在新claim下运行一次，拒绝覆盖/重复运行。
- 同一`data/.continuous-training.lock`覆盖登记、worker和终态；等待当前G段完成后才能运行。1GiB额外预留、申请6GiB、实际min(6GiB,available−1GiB)、600秒/0.1秒监督；Noria/Docker不操作，所有构图在受监督worker。
- 保存模型/图/配置、原始计数前后SHA、共享缓存证据、输入、所有输出、原始计时、CPU时间、文件SHA、phase、监督/worker终态和资源CSV。资源完成、数值通过、核级性能候选分别报告，不信任单个pass字段。
- 不启动神经场景训练或硬件、不重跑旧前向实验、不改已有G/frozen/R注册。新的source身份只属于此性能诊断；后续迁移必须明确来源版本和现有训练证据的兼容策略，不能直接改绑定文件继续旧run。

## 实施和验收计划

风险S2，隔离的离线实验脚本，复用已有资源监督；不改生产模型。单一实施者新增必要脚本/说明与实施报告；先做最小CLI/登记逻辑检查，禁止在软件实施阶段加载完整图。只有具体失败或协议缺口才扩展相关验证，不跑全项目或重复旧已通过检查。

新的独立软件验收后提交，协调者在`data/adjoint_kernel_20261019_r1`登记并执行一次完整图。新的独立实际验收核对实际数组、数值、时间、来源和资源；不重跑模型。每个已接受步骤提交，整体AC3–AC6保持开放。

## 验收标准

- [x] A：固定协议和轻量父入口、共享完整伴随缓存、线程/计时/保存与失败边界实现，经[独立软件验收](acceptance-backward-kernel-software.md)。4个不同定向检查通过，软件提交404b62c。
- [x] B：一次完整图有真实终态和原始证据，实际矩阵/输入不变，资源和数值如实报告，见[实际运行](backward-kernel-run-20261019-r1.md)。
- [x] C：新[独立实际审查](acceptance-backward-kernel-run-r1.md)核验180次输出/270组比较全部精确相同；SciPy和Torch8/10线程满足核级候选。整个训练尚未优化，真实autograd和完整窗口等价待下一批。
