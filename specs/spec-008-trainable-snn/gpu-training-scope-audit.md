# GPU 训练边界审计（2026-10-19）

状态：只读架构审计；建议待协调者纳入新的修复 Spec。本文件不授权运行或表示软件/数值验收通过。

## 范围与结论

用户要求修复前进输出饱和、方向错误和梯度过大，再继续训练，并把训练运算全部迁到 GPU。本审计只追踪 CPU/MPS 边界、可行接口与历史迁移，不重复训练，也不负责判定饱和和梯度问题的根因。

当前后端确实是 `mps_forward_backward_cpu_adam`，不能称为全部训练在 GPU。建议把**感觉编码、脑网络、读出、损失、反向、稳定范数/裁剪、Adam 参数与状态更新**全部置于 MPS；CPU 保留文件读写、图资产检查/初始缓存构造、进程调度、日志与独立保存证据验证。操作系统、Python 调度和文件系统仍然会使用 CPU，这是执行边界而非数值回退。

优先沿现有 `brain/`、`senses/`、`learning/` 职责复用，不复制第二整套训练器。先保持可复用的图缓存、数据读取和资产检查字节不变；GPU 感觉状态、GPU 优化器、严格历史导入是适合新增的独立模块。只有确需改变的共同训练编排或数值核心才在封存旧源后修改，不为“兼容旧哈希”复制整个框架。新的数值和优化器使用新协议、新 ID、新 run。当前 r3 已完成 160 窗，旧 CLI 不支持额外 epoch 或从 MPS 终档建立新实验。

## 当前计算落点

| 环节 | 当前设备/实现 | 建议与理由 |
|---|---|---|
| 原始图、类型/ID 映射、CSR→COO、值与坐标核对 | CPU NumPy/SciPy/Torch 构造，随后上传完整 MPS COO | 保留一次性资产准备；不缩图，不在每步以 CPU 执行图乘 |
| 固定完整图 W / Wᵀ 乘法、LIF、延迟队列、代理梯度 | MPS；所有节点参加伴随传播 | 保留 GPU 路径和完整拓扑；根因修正需单列数学契约 |
| 图片目标色掩膜与 12×16 占有率 | `target_mask()` 使用 NumPy | GPU 张量实现阈值与占有率；训练仍须明确当前输入只是指定红色标记，并非通用目标识别 |
| 感觉相位/事件 | NumPy float64，每个神经步执行 `SpatialDrive.step()` | 必须迁移或预计算；当前每步还构造长 166,700 的 CPU dense 向量并上传 |
| activation checkpoint 重算 | 神经运算 MPS，但重新执行上述 CPU 编码 | GPU 驱动应成为显式纯张量状态；重算不能改变外部相位/计数/时钟 |
| DN 汇总、线性读出、tanh、动作 MSE 与 gain 正则 | 已是 MPS | 保留；不能将迁移优化器当成修复饱和的证据 |
| raw 梯度 | MPS 产生，每窗复制到 NumPy | GPU 保留 raw 梯度用于裁剪；只有证据副本在边界下载 |
| 裁剪 | CPU NumPy float64 范数，float32 系数乘梯度 | 使用 MPS float32 的缩放范数，避免先平方 1e30 级梯度溢出 |
| Adam | CPU master 参数、m/v/step；更新后复制回 MPS | MPS 参数作为唯一 master；所有矩及 step 也在 MPS |
| 末态/参数/Adam 归档和独立数值检查 | CPU NumPy、NPZ、SHA、JSON | 可保留，明确是保存/验收；不得用验算结果替代 GPU 更新结果 |
| 训练场景/标签 | CPU 离线生成并封存，执行中读取 | 当前任务为离线模仿学习；环境推进并非本次已存在的在线可微训练计算 |

具体源码：`mps_policy.py`、`mps_recompute.py`、`mps_window.py`、`mps_optimizer.py`、`mps_state.py`、`mps_worker.py`；编码在 `senses/spatial_drive.py` / `target.py`。

旧 61/62 窗记录的 CPU 裁剪＋Adam 只有约 0.0004 秒/窗；不能宣传迁移这部分就会显著缩短约 27 秒的整窗计算。每神经步 CPU 编码、上传、Python 循环和状态检查同步更值得关注，但本审计没有做性能实测，不能给出其实际耗时比例。

## MPS 精度与优化器选型

MPS 不支持 float64 张量。官方 PyTorch [OperationUtils.mm](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/mps/OperationUtils.mm) 的 `ScalarType::Double` 明确拒绝该转换；本地当前 Torch 为 2.14.1。不能直接把现有 float64 相位和范数 `.to("mps")`，也不能允许静默 CPU fallback。

本地安装源码还有两个必须区分的事实：

- `torch/optim/optimizer.py::_get_capturable_supported_devices()` 不含 MPS；不能把 `capturable=True` 当作 MPS 全设备 Adam 的通用方案。
- `torch/utils/_foreach_utils.py::_get_fused_kernels_supported_devices()` 包含 MPS；`torch/optim/adam.py` 的 fused 路径把 step 初始化在参数设备，m/v 也随参数驻留，并调用设备 fused Adam。因此优先验证内置 `torch.optim.Adam(..., fused=True, capturable=False)`；不能仅由设备列表推断实际所有算子/恢复路径已通过。

若内置 fused 在当前版本的实际定向检查不支持所需行为，才考虑局部显式 MPS 张量 Adam。不要先实现一个重复优化框架。原 CPU `foreach=False, fused=False` 的逐项舍入顺序不等同 fused MPS；优化器 schema/身份与验收参考须相应版本化，不要求伪造逐位 CPU 等价，也不沿用旧 `training_math.py` 的 exact clipping 断言冒充新后端通过。

### 稳定 GPU 裁剪

对每个原有参数组，先求 `a=max(abs(g))`；在安全分母下求 `u=g/a` 与 `q=sqrt(sum(u²))`。用 `coefficient=min(1,(1/a)/(q+eps/a))` 等等价缩放形式获得系数，避免直接 `sum(g*g)` 溢出。零梯度显式处理；原始非有限梯度必须报错而不是经 `nan_to_num` 隐藏。极小 a、最大有限梯度和多张量组也需要专门边界检查。

记录 a、q、clip 系数、裁剪前后非零数、更新前后参数与 m/v；若原始范数超出 float32 可表示范围，用这对缩放量或 log 范数记录，不让仅用于报告的乘积溢出污染系数。CPU float64 只用于对保存值独立复算。这个数值实现避免范数溢出，并不能修复反向图本身的梯度爆炸或小梯度下溢。

## 感觉编码两种实现的取舍

### 推荐：显式 GPU 编码与 GPU 状态

增加有独立职责的 MPS 感觉实现，继续沿用登记的全部 3,377 个输入和 pixel-cell 映射。state 包含设备 `phase`、`event_counts` 和时钟；输入图像 uint8 上传后在 MPS 完成红色阈值及 cell pooling，每步只在 GPU scatter 到完整 N 维输入。注册初始化相位来自哪个种子/算法及 float64→float32 的转换；不得将原 float64 drive ID 原封不动套到新状态。

新 drive 使用 float32 时，跨阈值事件时刻可能与旧 float64 不同，这是实际数值变更。应保存同一画面/初相位下的事件计数与时刻差异、相位误差，再以新 drive ID 运行。不能将仅参数复制成功说成旧神经感觉完整状态逐位续接。新训练若从场景起点进入下一轮，可以按登记规则重置神经/感觉状态；参数和 Adam 的继承或修复性重置必须另行明确。

activation checkpoint block 必须显式传入并返回 drive 张量状态，禁止依赖可变 NumPy 闭包；反向重算保持相同输入、同一次 forward 末态，不推动外部游标。保存时才把真实设备状态下载成 NPZ。

### 备选：一次性编译旧感觉事件表

训练数据固定且感觉映射不可训练，因此可在训练前用旧 float64 编码器生成不可变事件表，训练期间将事件表上传并只做 GPU 神经与优化运算。每两帧 1,000 步×3,377 输入的 bool 表约 3.22 MiB；320 帧全表约 0.50 GiB，宜逐窗/逐场景读取，不能建立 160,000×166,700 的 dense 表。

这能保持旧事件规则并消除重算时 CPU 编码，但应如实称为“离线 CPU 预编码，训练内核全 GPU”，不称编码也在 GPU。需要绑定图像、原 drive ID、起止相位/事件数、每步事件表哈希，且未来实时摄像头仍要单独解决 GPU 编码。用户当前要求全部训练 GPU，推荐前一方案，不自行把本备选包装成同一承诺。

## 最小接口与来源迁移

建议按职责改现有文件，必要新增 GPU 感觉模块与历史导入模块即可：

1. GPU 编码：`initial_state()`、`encode_frame(frame, state)` / `step(mask, state)`；state 的设备、dtype、时钟和身份是协议的一部分。
2. GPU 优化器：直接绑定 policy 命名参数；`step()` 不接收 NumPy raw 梯度，不持有 CPU master；`snapshot()` 才下载参数与完整 Adam；启动时断言参数、梯度、m/v/step 全部位于 MPS。
3. 窗口：MPS target/valid、loss、raw grad、stable clipping、Adam；统一 GPU 完成后采集保存副本，不能将 CPU oracle 结果写回设备。
4. 历史导入：读取封存资产，验证来源 manifest、checkpoint/prefix/数据集/完整图，返回参数和状态；不依赖“当前源必须等于旧源”才能检查历史。
5. 新协议：登记 parent run/checkpoint、继承参数/矩清单、任何重置、修复后的数学式、感觉 dtype、优化器后端、epoch/窗口预算与新源码闭包。

当前 `mps_contract.validate_run()` 递归调用 CPU `validate_run()`，两者把历史 `implementation_sha256` 与**当前工作树**源码比较。改 42 份绑定文件中的任意一份都会让旧父 run 校验失败；`policy_identity()` 也读当前源码。继续累积 `mps_v2_*` / `mps_v3_*` 并全量复制训练器不是合理长线方案。

建议修复前将已验收 r3、其 CPU 父和相关未完 frozen/R 的准确源码闭包保存到不可变 source manifest + Git commit/archive，并在旧源码仍可验证时封存父 checkpoint 导入凭证。新历史读取器按**归档源**和严格文件/数组/游标/前缀哈希验证，不将旧 run 重标为新实现，也不为检查历史执行归档中的任意 Python。原 r3 全部运行数据、失败记录及旧 commit 保留；新实验源可以正常维护。旧 frozen/R 若以后继续，应使用对应归档源码或显式新迁移，而不是默认让新源码接管其旧 run。

这个迁移既不能简化成关闭 source 检查，也不能只记录一个 parent ID 而不校验实际文件。第一轮新协议可只支持明确的旧 r3 终档导入，不必泛化成任意版本插件系统。

## 验证与风险

属于 S3：训练数学、共享来源检查、完整状态与优化器持久化均会改变；不需要无边界运行整个项目测试。建议按依赖顺序一次完成：

- 对 GPU 编码固定刺激、空白/边缘、跨帧/重算恢复与旧 float64 差异做定向检查，公布新的事件差异，不把差异隐藏为容差。
- 对实际 MPS stable norm/clip、内置 fused Adam 的多步 m/v/step、极大/零梯度、归档恢复做独立小型算术检查，fallback=0；新融合舍入误差的验收阈值事先登记。
- 对修复梯度的核心小型数值场景检查、完整窗口断点重算一致性和命名参数梯度设备检查；完整参考图仍须实际短窗证据，fixture 不能替代。
- 先完整图诊断/一次更新，再新进程恢复一窗；记录全部输入、动作、raw/postclip 梯度、参数/矩/状态、同步计时及 RSS/MPS 资源。通过新独立验收后才启动登记的续训批次。
- 保持 1 GiB 系统预留、共享数值锁、Noria/Docker 不动、所有数值任务串行。GPU 不保证吞吐提高，更不自动证明控制学习成功。

验证阶段未执行任何 Torch/MPS/NumPy 数值作业、测试或训练；没有修改业务源。此报告基于只读代码/文档检查与官方源查证。
