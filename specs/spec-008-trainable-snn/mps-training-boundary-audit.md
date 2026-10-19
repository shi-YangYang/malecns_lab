# MPS 完整图训练与迁移边界审计

状态：**Draft / 只读设计审计；未实施，未运行完整图 GPU。** 2026-10-19。

用户已授权先比较完整连接组 GPU 的内存、速度和 CPU 数值，验证通过后切换训练。本稿给出内核门槛之后的最小模型与迁移路径；不是性能结果，也不提前批准迁移。原 G29、R13 与历史基准源码及旧 run 保持原字节，Noria / Docker 保持运行，额外系统预留保持 1 GiB。

## 当前事实

- `data/mps_compatibility_20261019_r1.json` 记录本机 PyTorch 2.14.1、MPS 可用、fallback=0。2×2 dense 与 COO 的前向/输入反向通过；CSR 构造失败。该证据不涵盖完整图、当前 LIF 运算集合或训练更新。
- 实际训练仍是 PyTorch 参数、代理梯度、Adam；固定矩阵前向是 SciPy CPU CSR，反向是 Torch CPU CSR。完整图为 166,700 节点、25,582,938 有向边。更换存储布局不授权过滤任何节点、边、符号或计数。
- 本稿只读取源码、既有 JSON 与项目规范，没有导入 Torch、加载图、数值复算或启动训练。G joint=60/160、frozen=10/160 为协调者交接状态；迁移执行前必须重新验证实际末尾 seal / checkpoint，而非相信此处数字。

## 为什么不能直接 `.to("mps")`

| 边界 | 当前代码事实 | 隔离候选必须处理 |
| --- | --- | --- |
| 固定稀疏权重 | `brain/trainable.py` 生成前向/伴随 CSR；SciPy 缓存通过 `.numpy()` 共享 CPU 存储 | 构造完整、coalesced MPS COO 的 W 与 Wᵀ，一次构造后缓存；固定边值不训练，只有输入向量传梯度 |
| 脑状态与动力学 | `reset` 在 CPU 建张量；`step` / `_validate_state` 明确拒绝非 CPU；tonic 亦限制 CPU | 新后端保留原积分、阈值、复位、18 步延迟与整数时钟；在 MPS 分配/验证状态 |
| 其他算子 | gain/tonic 用 `nextafter`、tanh/sigmoid、布尔索引、int64 时钟、gather 与 where | 稀疏乘法通过不代表这些算子通过；定向小检查须覆盖实际组合，禁止 CPU fallback 掩盖缺项 |
| 感觉输入 | `DriveEvents.to_torch` 每步产生 166700 维 CPU 向量；SpatialDrive 本身使用 NumPy float64 相位 | 保留原 CPU 事件生成/相位；只把原 float32 增量搬到 GPU。初版不同时重写随机数/编码，避免另一个数值变量 |
| 策略 | `NeuralPolicy._initialize` 精确检查原脑类型；参数/缓冲校验与 forward 仅适配 CPU | 独立候选策略类型、实际身份与设备校验；保持完整 R1–R6、全部 DN、两半窗均值和原读出 |
| 窗口与裁剪 | `training_window` / `joint_probe_worker` 的 snapshot 和 state digest 直接 `.numpy()`；裁剪以 NumPy float64 求范数 | 边界用 detach→CPU→NumPy；保持小梯度数组的原稳定裁剪、原系数舍入和独立复算 |
| Adam | 精确参数/源身份与 CPU 状态检查；foreach/fused/capturable 均为 false | 新绑定校验，保持命名、对象顺序、超参和矩；确认实际 MPS Adam 状态布局，不能假定 step 必须和参数同设备 |
| 持久化 | 原 policy/train checkpoint 固定 schema/backend/runtime/source；恢复只构造 CPU policy | 新 schema/运行身份和显式迁移 lineage；旧档读取校验可复用，不能给新模型伪装旧 ID |

`step` 中多次 `bool(isfinite(...))`、`count_nonzero` 等 CPU 判断若直接移到 GPU，会频繁等待 GPU；单次内核快并不能保证完整窗口快。首轮可保留严格校验获得真实基线，若实测同步为瓶颈，再单独登记验证频率/异步错误标记的变更，不在迁移时顺手隐藏检查。

## 建议的最小模块边界

1. **`brain/mps_trainable.py`**：完整图 MPS LIF、固定 COO 一阶 autograd、设备状态校验。复用原 `TrainableLIFConfig`、`LIFState`、`surrogate_spike`、graph/map 校验与 metadata 工具。局部重新表达原约 33 行 `step`，对照操作顺序；不要复制整套图 loader。显式 `backward = Wᵀ @ gradient`、矩阵返回 `None`，不分配逐边训练参数或逐边梯度，不只对本步发放细胞求导。
2. **`learning/mps_policy.py`**：独立 `nn.Module` 策略与真实 identity。可在一次 CPU 原工厂校验后消费其 graph/map/config/初始小参数，再释放旧脑两个 CSR 及 SciPy view，构造 MPS 脑；避免双份全图缓存同时长期存在。另一可接受做法是在新工厂显式复用已验证 loader/map 并保留原构造检查。不能 monkeypatch 原全局、修改 `__class__` 或跳过完整图准入。新策略保持相同输入、读出、状态时间和参数名。
3. **`learning/mps_window.py`**：两帧更新与小参数/梯度快照边界，复用原 `ADAM_OPTIONS` / `optimizer_contract`、`training_math.clipping` / `validate_window_numbers`。只替换设备搬运和严格候选绑定，保持先建正则、递推、loss、反向、裁剪、Adam、detach/清梯度的顺序。第一轮只支持当前 float32、fixed_lamina_8mv、joint/frozen 契约，不同时拓展 dtype/背景/逐边学习。
4. **只有模型对照通过才增加 `learning/mps_checkpoint.py` 与薄运行入口**：新封存/恢复和迁移。原资源监督与父运行工具可作为 helper，源闭包绑定实际使用文件；没有必要复制整个 CPU 研究/评估目录。若之后形成共享抽象，必须另立源身份迁移，不能在本批偷偷改 G29。

新身份明确记录 device/backend/layout、Torch/runtime、actual source hashes 与 parent CPU 模型/运行关联；graph/map/spatial/drive 仍为真实 full_reference。候选没有设备入口，旧保存器必须继续拒绝它。先做诊断产物，不能把未经验证的 NPZ 命名成可续训 checkpoint。

## 内存与速度应如何测

- 一份 COO 的两个 int64 坐标加 float32 值：`25,582,938 × (16+4) = 511,658,760 bytes`，约 0.477 GiB；W 与 Wᵀ 若各有独立存储约 0.953 GiB。这只是静态有效载荷，**不包含**原图/构造副本/coalesce 工作区、MPS allocator、神经轨迹或反向临时量；不能当峰值承诺。
- 两帧共 1000 神经步。一份每步保留的 N 维 float32 轨迹已是 `166,700 × 4 × 1000 = 666,800,000 bytes`，约 0.621 GiB；实际 autograd 会保存若干张量并可能复用/释放，不能简单假设只有一份，也不能仅由这项公式宣称一定超限。现有 CPU RSS 实测不能直接作为 GPU 峰值。
- 原 `resource_guard` 采样 process-tree RSS 与 system available。新增 worker 侧 `current_allocated_memory`、`driver_allocated_memory`、`recommended_max_memory` 的阶段/周期记录和预登记限制；RSS、MPS allocator 与统一内存统计存在不同/重叠口径，分别报告，不能相加作为总量。系统可用低于 1 GiB 仍中止。
- 保留独占锁、600 s、`min(6 GiB, available−1 GiB)` RSS 上限，GPU allocator 的预算还需在 Ready Spec 明确，不能仅由 RSS 通过宣称全部 GPU 内存受控。GPU 监控也需在构造/coalesce、前向与反向覆盖；单纯窗口末端采样可能漏峰值。
- `PYTORCH_ENABLE_MPS_FALLBACK=0`；不修改 high-watermark 绕开分配保护。计时边界同步 MPS；分别报告转换/上传、warm-up、前向、反向/裁剪、Adam/保存、整个监督墙钟。CPU 小数组裁剪、感觉编码和存档仍可合法存在，但须如实计入端到端耗时。
- 不同时常驻 CPU 与 GPU 全模型；比较串行运行并落盘同一输入/参数/状态。CPU 使用已登记 4/1 线程，GPU 首轮不同时改窗口或数学。单内核提速不触发切换；完整窗口真实 wall 和内存才是切换依据。

## 核门槛之后的最短验证路径

### B：完整模型窗口对照

内核 gate 通过后，先用小型非对称数值夹具验证所有候选设备算子、一阶 input/θ 梯度与延迟/静默细胞梯度；这是接口验证，不是主脑替代。随后登记一个**完整两帧 / 1000 步**窗口的 CPU 和 MPS 配对；按现有来源完整载入图，不裁图、不缩窗冒充原训练。

优先选已保存 CPU 起止状态的真实训练边界，而不是重新初始化：CPU 首窗只有起始态摘要，不便描述其 GPU 数值差。当前 G 的窗口 59 checkpoint → 已保存窗口 60 是候选锚点，可完整给定非零 Adam/θ 和 recurrent state，并使用已经存在的下一窗证据。执行前验证对应实际 seal/source/dataset/window 链，不在稿中猜文件名或 hash。若需要同期 CPU 性能，串行在新诊断目录重算该固定下一窗；这有明确的 GPU 比较目的，不能写入旧 run。

比较必须覆盖：输入/参数/矩/起始完整状态；全部 DN 特征、三轴动作、loss；raw/post-clip 梯度、裁剪系数；Adam m/v/step、参数实际 delta；末态 v/g、所有 pending、refractory；感觉 phase/events 与全部时钟。保存 exact、max-abs、RMS、relative L2/Linf、零支持/符号差数；范数/差统计用 CPU float64。不能只以动作接近或参数仍有限判定等价。

**门槛应预先冻结，不能沿用 SciPy-only 的“前向必然 exact”理由。** GPU 改了前向加法顺序及设备算子，硬阈值可能把小差扩大为脉冲分歧。建议把离散发放/不应期、phase/events/时钟要求精确，神经浮点状态/特征/动作和梯度/参数 delta 采用分别登记的严格尺度门槛；不能看完结果再放宽。若发放路径明显不同，保留报告，不能仅靠容差扩大把它当作无影响迁移。绝对阈值不得掩盖小梯度或 Adam 下溢。

尤其不能改用 float32 `clip_grad_norm_`：历史首窗 θ 原梯度范数约 7.8e31，原代码特意用 float64 稳定归约。保留同一 CPU 小数组裁剪；不要同时改变训练算法来使 GPU 结果通过。

### C：通过后迁移与实际续训

1. 重新读取 G 当前真实完成边界，校验旧 source/runtime/登记/累计 prefix/checkpoint；为 GPU 建新 run，记录 parent run/checkpoint/window 和完整 source hashes。原 run 不再追加 GPU 窗口。
2. 精确保留 theta、读出、gain/rho、命名 Adam m/v/step、v/g/18 项 pending/refractory、drive phase/events、scene/frame/optimizer 游标与已用学习预算。CPU→MPS float32/int64 转移后的返拷数组必须精确；不 reset、不预处理、不清 Adam、不重放前缀。新 model/policy/state ID 必须真实更新，来源关系显式记录。
3. 用迁移后的相同非零边界串行比较 CPU/MPS **下一窗**；完整窗口门槛与资源/速度 gate 通过才写新的 GPU 完成 seal。该窗计入共同 160 窗预算一次，不把比较试算重复计为新训练机会。
4. 新进程加载 GPU checkpoint 再执行下一窗，验证恢复完整性与累计 cursor；可与同边界保存的直接续算结果比较。随后才把日常入口切向 GPU 新 run，保留明确 fallback 为用户选择的原 CPU run；运行失败不得自动回 CPU 并混入同一证据链。
5. joint / frozen / 未来 R 的科学比较要登记共同后端和预算，不能只给主模型更多选择机会。当前两支前缀长度不同，不能悄悄声称全程同后端配对；需要在 GPU 比较协议中记录各自的 CPU 前缀及迁移点，或另起共同 GPU 研究 run，前者不能冒充从零纯 GPU 训练。既有 CPU 进度必须保留。

## 阻塞与停止条件

当前未解决的是实测能力，不是缺用户授权：完整 COO 内核资源/吞吐/差异；实际 MPS 算子与一阶反向；完整窗口峰值和端到端收益；阈值递归/梯度/Adam 差异；可校验的新身份/持久化和跨进程恢复。任何 gate 失败就报告具体失败和已保存证据，保留 CPU 正式训练，不能先把默认入口切换再补验证。

本稿 S0，仅检查新增文档与代码边界；没有运行测试/数值作业或修改业务代码。后续实施按跨模型、优化器、状态持久化的 S3 范围做定向验收，不为本稿重跑原 CPU 检查。
