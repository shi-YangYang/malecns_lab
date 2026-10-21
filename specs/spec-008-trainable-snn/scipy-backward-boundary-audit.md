# SciPy 伴随接入与完整更新窗口边界审计

状态：**Draft / 只读设计审计，未 Ready、未实施、未运行新数值实验。** 2026-10-19。

本稿在既有[完整伴随内核实测](backward-kernel-run-20261019-r1.md)之后，限定下一批为独立候选的一阶 autograd 与历史 G 首个两帧更新等价检查。原 G joint、frozen、R 生成器和两个历史基准脚本保持原字节；不授权旧 run 迁移，也不以本批代替完整训练、研究对照或实飞。

## 结论与最小实现边界

有可行的隔离接入路径，但现有实现没有可注入的 matvec 回调。建议新增明确的 `LaminaTonicLIF` 子类，仅覆盖约 33 行 `step` 和伴随缓存/autograd；由独立工厂先完成原 `NeuralPolicy.from_reference` 的全部校验与读出初始化，释放旧脑对象及缓存，再构造候选脑并更新真实身份。保留原 `NeuralPolicy` 对象的全部前向、感觉与读出方法，无需复制 `_initialize` 或整个训练器。

真实候选身份不能直接通过旧 `compute_window`。其优化器快照走旧参数身份校验，明确固定 Torch 伴随后端及源清单。因此还需要一个窄的候选 Adam 绑定层和约百行的两帧窗口编排，复用原数学与前向；不修改旧全局函数、不伪造原 model/policy ID、不使用 `production=False` 绕过证据准入。当前只做一次诊断窗口，可保存不可续训的诊断数值档，无需提前复制整个 checkpoint/runner 系统。

| 现有位置 | 已核实的边界 |
| --- | --- |
| `brain/trainable.py:83–99,254–255` | `step` 直接调用模块级 `fixed_sparse_matvec`；没有 `self.matvec`。只覆盖同名实例属性不起作用。全局替换函数或替换函数 `__globals__` 会污染旧模型，不可采用。 |
| `brain/trainable.py:225–257` | 需覆盖的局部步骤包含原验证、积分、代理脉冲、复位、延迟入队。候选必须逐语句保持原浮点求值次序，仅替换固定矩阵乘法调用。不能改成仅发放索引路径。 |
| `brain/tonic.py:29–81,94–106` | 子类可复用原初始化、全部 L1–L5 固定背景及 `rho=0` 逻辑；其 `_initialize` 会设置旧身份，因此候选必须在构造完成后、任何 state 生成前封存新身份。 |
| `learning/neural_policy.py:125–158` | `_initialize` 要求精确原脑类型，不能直接把新子类传进去。不要复制整份构造或通过更改 `__class__` 绕过。 |
| `learning/neural_policy.py:203–257` | 构造后的 `_validate_parameters`、`_drive_for`、`initial_state`、`forward_tensor` 不要求精确原脑类型；保留真实候选 model/policy ID 后可直接复用。 |
| `learning/training_window.py:14–20,34,83–86` | 开始及结束的 `optimizer_snapshot` 经旧身份门槛；`compute_window` 没有注入快照/优化器回调。原函数不能直接用于真实新后端身份。 |
| `learning/training_optimizer.py:28–70,94–107` | `_policy_parameters` 依赖 `policy_checkpoint._arrays`，不是纯优化器算法。需新严格参数准入，仍使用原 `ADAM_OPTIONS`、`optimizer_contract`、`validate_moments` 和精确 `torch.optim.Adam`。 |
| `learning/policy_checkpoint.py:70–91,127–195,198–210` | 固定 backend、model/policy 字段集合、源码清单及精确 `NeuralPolicy` 类型。新后端必须被旧 reader/writer 拒绝；不能暂写旧 identity 后通过校验。 |
| `learning/training_math.py` | `clipping`、`validate_window_numbers`、一般步 Adam 算术是角色无关的数组运算，可复用；它们不负责证明候选身份或来源。`compare_r2` 缺少裁剪梯度、Adam 和完整末态，不能作为本批唯一检查。 |
| `learning/training_checkpoint.py:114–177` | `_state_metadata`、`_state_arrays`、`_validate_state`、`_build_state`、`_declarations`、`_no_gradients` 可复用；完整 save/load 会进入旧参数及优化器格式，不可直接使用。 |
| `learning/joint_probe_worker.py:27–52` | `snapshot`、`raw_counts_digest`、`state_digest` 可直接用于候选。`state_id` 含 model/policy ID，数值相等不意味着新旧 `state_id` 相等。 |
| `learning/training_evidence.py:72–121`、`training_worker.py:39–138` | 原窗口 seal/扫描/worker 绑定旧 schema、旧 checkpoint、原训练身份。只用于验证只读历史锚点，不能把候选结果写回原链。 |

## 建议的显式候选构造

拟议名称仅供 Spec 冻结，不是已有 API：`brain/scipy_adjoint.py` 的 `ScipyAdjointLamina`，`learning/scipy_adjoint_policy.py` 的 `from_reference_candidate`。

1. 在受监督 worker 中走原 `NeuralPolicy.from_reference`，使用原 G 首窗配置、完整图、固定背景、CPU float32 和读出种子。构造不调用 `initial_state`、不生成 state 或 optimizer。
2. 校验原 policy identity 与已验证历史锚点精确一致，保存原 model/policy identity、图/map/config，以及小型 theta/readout/gain/rho 初值证据。保留原 `graph` 对象；不复制原始大矩阵。
3. 删除 policy 对旧脑的注册引用；不得保留局部旧脑、buffer、CSR 视图、状态或优化器引用。随后构造子类脑。可以在软件检查中用弱引用证明旧脑/缓存对象已释放；RSS 不是对象释放的充分证明，也不要求内存分配器立即归还所有页。图、map、读出继续保留。
4. 新脑先调用原 `LaminaTonicLIF` 初始化，再为 `base_weights_adjoint` 建立一次只读共享 SciPy CSR。使用其现成 data/col_indices/crow_indices，不做第二次转置、索引窄化或额外完整权重副本；验证 shape/dtype/地址/顺序/只读及内容对应关系。缓存所有权随新脑存活，不能每个反向步骤重新构造。
5. 自定义一阶 `Function` 的 forward 保持原 SciPy 正向；backward 对完整 N 维原尺度上游梯度计算 `torch.from_numpy(adjoint_csr @ gradient.detach().numpy())`，矩阵参数返回 `None`。支持非连续 CPU 向量，不把 `.detach()` 误称为二阶可微。显式声明只支持当前一阶训练；对高阶使用应清楚拒绝，而非产生貌似有效的二阶结果。
6. 子类仅覆盖局部 `step` 调用此 Function；原 `reset`、状态验证、增益、背景、代理梯度及延迟语义继承。注意原 `LaminaTonicLIF.for_numerical_test` 通过 `__new__ + _initialize` 绕过 `__init__`，不能原样继承后漏建候选缓存/身份；测试需要私有明确数值工厂并执行同一候选完成步骤。
7. 把新脑注册回 policy，核对参数名仍恰好 `brain.theta/readout.weight/readout.bias`、初始数值与步骤 2 精确一致，map/感觉/读出和图身份保持。旧脑与新脑缓存不同时常驻。全图构造会做两次顺序初始化，需独立记录这项启动成本，不能藏入每窗提速。
8. 封存新 model schema/backend、旧数值来源说明、新源码摘要与 model ID，再把真实新 model identity 嵌入新的 policy schema/source/ID；原 model/policy ID 只能作为显式 baseline 关联字段。图/map/spatial/drive 仍为实际 G/full_reference，不能伪标 research_control。新身份完成前不允许 reset/forward/optimizer。

本方案采用新实例的明确脑替换，不修改全局、不改 `__class__`、不把方法挂到已开始训练的实例上。当前 `NeuralController` 只检查 `isinstance(policy, NeuralPolicy)`，所以不能声称候选在内存级被所有设备入口绝对隔离；本批只提供离线诊断入口、旧持久化拒绝候选，且不新增设备调用路径。

## 窗口与证据 API

建议新模块只包含严格候选身份/参数校验、Adam 对象/顺序绑定与快照，以及 `compute_candidate_window`。窗口维持原 `training_window.compute_window` 的所有操作顺序：前向前建立正则、两帧递推、缩放 MSE、`loss.backward()`、复制原始梯度、原稳定分组裁剪、原 Adam、完整 state detach、清梯度、原 NumPy 独立复算。源清单绑定这些实际复用 helper，不能只绑定新增文件。

新优化器校验必须明确固定候选类型、schema/backend/source、baseline 来源和真实嵌套 ID；检查三参数的对象身份、顺序、requires_grad、重复/额外参数、精确 Adam 配置和矩。不能把旧 `_arrays` 调用删掉后只相信自报 hash。可在构造期使用原校验验证真实 baseline，再严格验证候选仅有登记的身份/后端差异；不能递归删除任意字段后“比较相等”。

诊断只支持固定 joint 首窗、`before_steps=0`，输出完整 evidence、末态数组和命名 Adam 矩，明确 `purpose=offline_backend_equivalence`，不冒充原 training checkpoint 或可续训档。若此批另要证明第二窗/恢复，应在 Spec 中明确增加该切片，不能靠档名或保存数组宣称恢复已验证。

## 实际首窗比较：固定对象与建议门槛

历史锚点为 `data/neural_training_joint_20261019_r1/segments/000001/windows/000001/`。本审计使用标准库读取 JSON、ZIP 成员及 NPY 头，**没有读取数值数组进行复算**。可用证据包括：

- `evidence.npz`：输入、初值、全部 2628 维 DN 特征、三轴动作、三组原始/裁剪梯度、更新参数/gain/rho、全部命名 Adam 矩。
- `checkpoint/training.npz`：166700 维 v/g/refractory，完整 18×166700 pending、3377 维感觉 phase/events、全部 Adam。末态 frame=2、脑/感觉 step=1600。
- `event.json.start_state.values`：初始 600 步预处理后全部状态数组的 dtype/shape/SHA；历史起始态只有这些摘要时，不编造其逐数组误差。候选保存实际起始数组可供未来复核，但与历史的当前比较只能是摘要。
- window ID `491c6d0b6b841610121c7889a3a5761b047c06c542fdc4f244dff67ea2f9895e`；evidence 文件 SHA 声明 `c618ec6dbca47b3bdd1b510a4629f47507ae09db992d81bc8e43543aeb17566a`；checkpoint ID `1387b7b3da06366fc0b0530bd6de3b0228db99684cdd56580a989038e75dc375`。实际运行前必须验证真实文件及父 seal/登记/终态链，不仅信任本稿中的声明。

历史 JSON 记录首窗 theta 原始梯度范数约 `7.785561213505507e31`，裁剪系数约 `1.2844288816809274e-32`，4259 项梯度非零；其中 2714 项有非零梯度但 Adam 二阶矩为零。它们是保存的历史报告值，本审计未独立重算。这个跨度说明统一 `atol=1e-7` 不足以检验小梯度和下溢后的矩。

以下建议在运行前冻结，失败后不能放宽原登记：

| 比较对象 | 建议固定要求 |
| --- | --- |
| 初值、输入、shape/dtype、整数/布尔状态、感觉 phase/events、时钟 | 精确相同；policy/model ID 应各自真实有效且因后端而不同，不强行要求总 `state_id` 相同。 |
| 起始状态 | 全部 `values` 子项摘要精确一致，drive/spatial ID 与时钟精确一致；只允许预先列明的新 model/policy ID 差异。 |
| DN、动作、完整末态浮点值 | 要求精确相同。该首窗前向实现和初值未变，末态在 Adam 前算出，反向仅影响参数，不应改变这些值。没有保存的每步全图轨迹不作精确等价声明。 |
| raw/post-clip 三参数梯度、Adam 一阶/二阶矩 | 全部有限；零支持集合精确、非零符号精确；逐项 `abs(new-ref) <= 1e-6*abs(ref) + 2^-149`，用 float64 计算；相对 L2 与相对 Linf 差均 <=1e-6。参考全零时要求候选全零；避免分母任意加大 epsilon。`2^-149` 仅允许 float32 最小次正规数级舍入，不能掩盖零支持变化。 |
| 更新参数 | 原 `rtol=5e-6, atol=1e-9` 仍作为绝对参数检查；另比较 `after-before` 的零支持/非零符号及 `rtol=5e-6, atol=2^-149`，并报告相对 L2/Linf，以免仅靠接近原参数值掩盖更新差异。计算差值时先转 float64。 |
| gain/rho、损失与裁剪记录 | rho 保持精确零；gain 通过原公式独立检查并报告新旧差异；损失沿用原 `rtol=1e-6, atol=1e-7`，裁剪系数和稳定 float64 norm 独立复算并报告相对差，不能只比较最终动作。 |
| Adam 算术一致性 | 原 `validate_window_numbers` 的独立检查不变：moment rtol=1e-6/最小次正规数 atol，参数 rtol=5e-6/atol=1e-9，裁剪结果按原复算精确相同。它与上述新旧对比是两个不同门槛。 |

每一字段保存 exact、max-abs、RMS、参考范围/范数、相对 L2/Linf、零支持差数、非零符号差数；统计先转 float64，不能在 float32 对巨大梯度平方。首窗全过只证明这个窗口；不能外推所有后续更新位级一致。

## 一阶边界与真实上游梯度

软件检查以独立小型非对称有符号矩阵为数值夹具，覆盖自环/孤立点、零输入但非零上游梯度、非连续向量、float32/float64、静默细胞递归代理梯度。独立 dense 参考核对输入梯度和 theta 梯度；小电路仅证明新 autograd 接线，不替代完整图。不得用硬阈值有限差分作为代理梯度参考。

本批必须执行真实完整窗口 `loss.backward()` 并比较所有参数梯度；另外记录 N 维上游梯度不是完成窗口等价的必要条件。若希望同时回答“实际传入伴随算子的向量是否也一致”，可在**新候选 Function 内**加只读观察器，预登记反向调用序号 0/249/499/749，保存传入向量及 SciPy 结果，退出训练计时后对这些原尺度向量单次计算原 Torch 4 线程伴随结果。不修改/重排传入梯度，不以非零程度挑选；未到达的序号如实记录缺失。它验证的是候选这次运行的真实上游向量，不是原历史 run 未保存的梯度轨迹。

观察器会增加复制/写盘成本，因此有观察器的窗口不能直接声称无观察器的端到端耗时。更简洁的下一批可以暂不加观察器，以完整参数梯度/Adam/末态等价为核心门槛；若失败，再针对具体偏差追加隔离采集，不扩大为每步全图录像。

## 性能报告与未来迁移

保持原 CPU float32、4/1 线程、两帧各 50ms/总 1000 步及初始 600 步；本批只改变伴随后端，不同时改线程或模型参数。报告构造、预处理、两帧前向、反向/裁剪、更新及保存/监督总时间。历史首窗 `compute_seconds=95.44689095799913`、`backward_clip_seconds=76.4704832080024` 只是跨时间的已保存参照，不是同期交错计时或稳定总体提速保证；新工厂的双次顺序构造需单独计量。

继续使用相同 OS 锁、1GiB 额外预留、`min(6GiB, available−1GiB)` RSS、600s/.1s 监督；原训练/生成器/数值检查退出后才执行。Noria/Docker 和旧封存资产保留。固定来源闭包包含候选、原 G29 中实际调用的 helper，以及两份历史基准中真正复用的源码；不修改后重新补历史 SHA。

未来可以单列迁移协议：验证旧终态 checkpoint/prefix/source 后，创建新 run 与新身份，精确保留 theta/readout、全部 Adam、完整 recurrent state、场景/帧游标及学习预算；新状态只更新身份关联，原数值不重放/重置。先在固定非零旧矩的保存边界，串行比较旧/新下一窗，再验证新进程恢复。迁移 lineage 必须记录旧 run/checkpoint 和新源版本，禁止重写原 G run、旧 policy ID 或旧 source snapshot。joint 与 frozen 是否迁移、R 科学比较的共同后端与预算须一致登记；不能只有某个更有利分支得到新选择机会。本批不实现任何迁移。

## 审计范围与停止

风险 S0：只新增本 Draft，阅读治理、原源码和既有实际报告/JSON/ZIP 头，检查新增文档；未导入 Torch/NumPy/SciPy、未加载完整图或数值数组、未运行测试/训练、未修改代码和数据、未 commit。协调者期间运行的 frozen 续段不属于本审计。下一步由协调者把选定范围和门槛形成 Ready Spec，再安排实施与新的独立验收。
