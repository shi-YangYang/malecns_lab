# Acceptance — 完整图伴随内核基准软件

## Result

PASS

新的独立验收者审查 [Ready 协议](backward-kernel-benchmark.md)及本批实际新增文件。验收标准 A 的软件范围满足；B、C 的完整图运行与实际证据审查仍未执行。本结论允许按协议提交后登记一次完整图实验，不代表当前训练已经提速或任何候选已获得性能证据。

## Spec Coverage

- **轻量登记与来源**：父入口仅使用标准库、psutil 和既有纯 hash／资源辅助函数，不导入数值 helper。独立协议包含完整图文件、逻辑图、原始计数及规模身份；登记绑定本地导入闭包、源码快照、解释器／库版本／线程环境。worker 在导入和结束时核对实际本地源码覆盖，并重查来源、图文件及原始计数不变。原前向脚本只被复用纯 hash 函数，没有修改其全局协议或字节。
- **完整伴随缓存**：通过原 `load_training_graph`、`TrainableLIF` 创建全部图及其 `base_weights_adjoint`，没有改为正向矩阵或过滤发放细胞。`shared_adjoint` 在计时前一次构造 SciPy CSR，严格核对 CPU float32、int64 索引、shape、nnz、数组地址、共享内存、stride、顺序／数值和只读视图；完整数组摘要、指针及只读属性在实验后复核。实际完整图共享是否成立由后续运行证据决定。
- **固定刺激、线程与调用**：输入生成严格采用 seed 1901、同一 default_rng、zero、无放回 floor(N/100) 个 1 和随后生成的 uniform float32 dense，保存输入与选中索引。线程块固定 1/2/4/8/10、interop 1，每 case 两方法各一次 warmup 加五次交错调用，共 180 次，结束恢复 4 线程。Torch 和 SciPy 路径与协议一致，计时包含 SciPy 每次 NumPy 视图与 Torch 包装，不含输出复制和写盘；每 case 写盘后释放输出。
- **完整原始证据和判据**：全部 warmup 与重复输出、每次 wall/process CPU 时间、输出 shape/dtype/有限性及摘要保存。所有输出逐轮比较 SciPy/Torch、Torch/Torch4，并额外比较 SciPy/Torch4；固定 1e-5/1e-5 容差，保存 exact、max-abs、RMS 和失败列表。候选仅在全部数值检查通过且三输入对应中位时间都严格低于 Torch4 时为真；完整性失败也会关闭候选。没有采用最好单次或跳过负结果。
- **独占、失败和资源**：同一 `data/.continuous-training.lock` 覆盖目录声明、登记、监督及终态。原子 `mkdir(exist_ok=False)` 在失败结果写入边界之外，已有目录保持不变；worker 需要登记父 PID、已持有的工作区锁及新的独占 claim。所有构图发生在受监督 worker，采用 1 GiB 预留、6 GiB 申请／有效 min 预算、600 秒和 0.1 秒采样。资源完成、数值结果与性能候选分别记录，父 manifest 覆盖原始证据；没有设备、Noria 或 Docker 操作。
- **范围与兼容**：本批仅新增两份隔离示例、一份相关测试及实施报告。审查时 tracked diff 为空，旧训练核心、G29/R13 和历史前向工具未改。没有生产反向替换、旧 run 迁移或现有训练线程修改。

## Validation

- Risk Level: **S2**，隔离离线实验工具及直接相关失败边界。
- Evidence Reviewed / Checks Performed: 阅读项目治理、当前 Spec/Plan/交接、Decision 011/012、Ready 子协议、协议验收和完整实施报告；逐段审查 `examples/fullgraph_adjoint_benchmark.py`、`examples/adjoint_benchmark_kernel.py`、`tests/test_adjoint_benchmark.py`，并核对原 `trainable_graph.py`、`trainable.py`、`resource_guard.py` 和历史前向工具的实际调用语义。新增文件不依赖 git diff 展示，均直接读取。
- 实施者已有 4 个不同定向检查的通过记录：初始 3 项通过；原子目录修改后仅重跑 2 项相关登记检查，其中新增一项真实文件流式 SHA 登记／篡改拒绝。审查了其测试内容、作用边界和报告，没有发现需要重复运行或扩大检查的具体缺口。
- 本验收仅静态审查和工作区差异检查；没有导入 Torch/NumPy、执行测试、构建、完整图加载、基准或模型重跑，没有修改业务代码。没有把实施者的执行记录表述为本验收重新执行。

## Issues

未发现阻止本批按协议进行一次完整图测量的问题。

## Regression Risks

软件及小型代数 fixture 不验证完整 MaleCNS 的真实数值、资源与速度。线程固定顺序、有限重复和后台负载限制仍存在；合成向量不代表实际训练梯度。即使内核结果合格，实际 autograd、静默细胞梯度、两帧训练窗口及来源迁移仍需后续独立规格和证据，不能直接修改旧绑定训练来源。

## Required Rework

无。本批验收范围满足，停止检查。下一步由协调者提交后串行登记一次完整图，再安排新的独立实际验收。
