# 可训练完整连接图核心

本模块是 Spec 008 的 P0 数值核心，身份为 `cnsbridge.trainable_lif.v1`。它提供完整参考图加载、内部类型共享增益和脉冲代理梯度。P0的小电路检查与P1完整图实测分开报告；2026-10-19已完成一次完整图内部更新及恢复/重载检查，详见文末。该结果不代表已经学会跟随或可实时控制设备。旧冻结 LIF/V2 未被修改。

缓存固定伴随后，[同协议r4](../specs/spec-008-trainable-snn/p1-run-20261019-r4.md)及独立验收通过：反向由291.70s降至13.79s，整套探针83.69s，采样峰值1.745GiB。全部有效增益与神经响应精确相同，梯度/theta末位差另报；单次非隔离运行不能保证固定加速倍数。无梯度前向仍约16.5s/20ms模拟窗，通路诊断和任务训练须分别测量。

## 环境与入口

训练使用独立 `.venv-train`，不修改既有 Tello `.venv`：

```sh
python3 -m venv .venv-train
.venv-train/bin/python -m pip install -e '.[train,dev]'
.venv-train/bin/python -m pytest tests/test_trainable_core.py -q
```

`train` 固定 PyTorch 2.14.1；核心只允许 CPU float32/float64。具体安装版本和定向检查记录见 `specs/spec-008-trainable-snn/implementation-p0.md`。尚未验证 MPS 稀疏计算，没有设备回退逻辑。

```python
import torch
from cnsbridge.brain.trainable import TrainableLIF

# 必须事先明确感觉编码对应的原始 body ID；此处空集合仅演示零输入接口。
model = TrainableLIF.from_reference(
    "data/malecns_v2/full_reference.npz", input_body_ids=[], dtype=torch.float32,
)
state = model.reset()
# 以下仅展示 API；全图运行须先按 P1 登记时间窗和资源预算。
increment_mv = torch.zeros(model.graph.n)
spikes, state = model.step(increment_mv, state)
state = state.detach()  # 截断反传，保持电压、突触状态、队列和时钟。
```

`from_reference` 必经既有 `load_reference` 的图、注释和来源校验。不会裁边、裁节点、忽略自环或用小图替代。`for_numerical_test` 是明确标记为 `numerical_test_only` 的独立测试入口，不是设备或任务模型。模型身份记录图来源、原始整数计数摘要、完整节点/边/突触计数、类型分组、外部驱动 body ID、计算精度、参数、语义和实现文件摘要。

原始有符号 int64 计数保留在只读 `model.graph.counts`；计算权重另存 CPU CSR 浮点矩阵，不会覆盖原始资产。float32 可能舍入较大的整数或计算结果；身份记录精度。原始无背景模型的 `state_dict()` 只包含训练参数 `theta`，不重复保存原始大图；checkpoint 必须同时保存并校验 `model.identity`，不能把裸参数文件当作独立可复现模型。

## 数值契约与来源

参考 [Shiu 等的官方 model.py](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py)，2026-10-17 核对的内容 SHA256 为 `fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e`。使用其方程、默认参数、严格阈值、输入细胞无不应期、复位清除 g 这些设计依据。这是方程启发的工程模型，不宣称逐时间步复现 Brian2，亦不宣称 MaleCNS 真实生理参数已知。

单位 mV/ms，默认 `v_rest=v_reset=-52`、`v_threshold=-45`、`tau_m=20`、`tau_syn=5`、`refractory=2.2`、`delay=1.8`、`weight=0.275`、`dt=0.1`。时间步必须整除不应期与延迟；不自动取整。

```text
dv/dt = (v_rest - v + g) / tau_m
dg/dt = -g / tau_syn
```

每步 k 的固定顺序：

1. 队首到期增量加入 g。
2. 非不应期细胞精确积分耦合线性方程一个 dt；不应期细胞 v/g 冻结，但到达的增量保存在 g。
3. 指定外部输入细胞的 v 加上显式输入增量。核心不生成 Poisson 噪声、默认驱动或视觉特征；非输入细胞传入非零增量会报错。
4. 满足严格 `v > threshold` 且允许发放时产生二值脉冲；发放后 v=reset、g=0。
5. 非输入细胞在 `k + refractory_steps` 再次允许积分和发放；输入细胞没有不应期。发送的连接增量在 `k + delay_steps` 开始到达。

设 `a=exp(-dt/tau_m)`、`b=exp(-dt/tau_syn)`，活动细胞的解析积分为：

```text
v_next = v_rest + a*(v-v_rest) + g*tau_syn/(tau_m-tau_syn)*(a-b)
g_next = b*g
```

时间常数相等时，g 对 v 的系数采用极限 `dt/tau_m*a`。测试以独立的 2×2 矩阵指数和手写事件时间表核对，未用核心自己的输出作参考答案。

与官方实现的边界：官方依赖 Brian2 调度、PoissonInput 和不应期时间判定，本模块将其明确为上述整数步协议；外部输入由调用者提供，不采样 Poisson；官方 reset 字符串包含 `w=0`，但其神经元方程未声明 w，本模块不引入这个未定义神经元状态；本模块增加工程代理梯度和共享增益。以上不是生物学习规则。

## 学习参数与反向规则

按突触前细胞原始 `type` 注释共享内部增益：

```text
gain = 1 + 0.5*tanh(theta)
delayed_increment = W_base @ (gain[group(pre)] * spikes)
```

theta 初始为 0。缺失、空串与全空白注释进入组 0，其余原始字符串不去空格、不改名，按 Unicode 排序。IEEE 浮点饱和可能把端点舍入为 0.5/1.5，因此以 `nextafter` 保护严格内部端点，身份明确记录该舍入保护。正增益保持源符号；未知递质原始符号假设仍通过参考图来源公开。

前向脉冲严格二值，反向使用 `1 / (scale * (1 + abs((v-threshold)/scale))**2)`，默认 scale=1 mV。复位门、不应期门停止梯度，递归传播脉冲保留代理梯度。稀疏线性反向对所有细胞计算 `W_base.T @ gradient`，不只选取实际发放索引，因此静默细胞的潜在代理梯度不会被丢弃。固定矩阵没有梯度，不构造 N×N 密集图、不按步重建边权。

模型构造时从原始 CSC 计数分别生成前向 `(post, pre)` CSR 和伴随 `(pre, post)` CSR；释放前向临时 SciPy CSR 后才创建伴随缓存。每步反向直接复用伴随 CSR，避免逐步转置和格式转换；两个方向保持全部边、符号、计算精度与权重尺度。伴随缓存为非持久化 buffer，不增加学习参数或 checkpoint 内容；P2c 继承相同构造与反向。完整图 float32 缓存额外静态数组约 0.287 GiB，实际内存和性能变化须通过同协议完整图对照实测，不能由小图数值检查推断。

`run(inputs, state)` 的输入是每步一个 N 维 mV 向量，可传迭代器。返回脉冲序列和完整状态；连续两段传递状态与一次运行一致。`reset()` 重置动态状态而不重置训练参数；`state.detach()` 截断全部状态（含延迟队列）的梯度。跨模型状态、非法精度/形状、非有限参数/输入/状态会明确报错。

## 可选 L1–L5 背景工作状态

`brain.tonic.LaminaTonicLIF` 是独立身份 `cnsbridge.lamina_tonic_lif.v1` 的数值研究变体。原 `TrainableLIF` 与 P1 继续严格无背景、只有 theta；不会默认启用此变体。背景候选针对完全静息时抑制性 R1–R6 输入无法产生第一批非输入脉冲这一[方程推论](../specs/spec-008-trainable-snn/signal-observability.md)，尚无完整图传播或任务收益实测。

生产构造 `LaminaTonicLIF(graph, input_body_ids, mode="learnable")` 或 `from_reference(...)` 保留全部节点、边与原始符号，选择全部精确原始类型 L1、L2、L3、L4、L5。每组必须有注释成员；缺失、空类型和相似字符串不推断为这些类型，且不能与外部输入细胞重叠。仅显式 `for_numerical_test(..., tonic_types=("L1",))` 可选择保持 L1–L5 顺序的非空子集；这种小电路有 test-only 身份，不是替代脑。

```text
offset[type] = 16 * sigmoid(rho[type]) mV
integrated_v += (1-exp(-dt/tau_m)) * offset
```

初始 rho=0，对应8mV；`nextafter` 保持有限 rho 的偏移严格在(0,16)内，非成员严格0。**8mV 是工程初始化假设，不是已测果蝇生理常数。** 此项只进入活动细胞的膜电位解析积分，不直接增加 g，不扩大 input_mask，也不取消 L1–L5 不应期。没有下降神经元动作偏置。`tonic_offsets_mv` 返回每类型偏移，`reset()` 仅清动态状态、保留学到的 rho。

`mode="learnable"` 将 rho 注册为参数；`mode="fixed"` 将其注册为不参与优化器的 buffer。两者初始化前向相同，但身份不同、不能交换状态。严格零背景对照应使用原 `TrainableLIF`，不能用极负 rho 近似。变体身份包括模式、类型顺序、成员 body ID 摘要、数量、公式、初值、范围和源码摘要。整个图的统一 LIF 仍是建模近似。

默认8mV下，无突触单细胞从−52mV到严格超过−45mV首次发放为约41.6ms。数值检查使用独立指数解核对电位、首次发放、复位、不应期和 rho 梯度；该结果不是完整图观察。未来可观测性实验须预先登记足够预热及观察窗，在同一协议比较空白与位置/尺度变化；背景自发发放本身不证明视觉传播。零背景、固定背景、可学习背景分别报告；比较冻结/可训练内部增益时保持背景条件一致。

**本批不支持变体 checkpoint。** `learning.checkpoint` 只接受原 gain-only 动力学 schema 和精确的 theta 参数集合：保存、恢复和生产加载均拒绝 tonic，包括 fixed 模式，拒绝额外参数及 metadata 参数键。在创建目录或修改参数前报错，避免静默遗漏 rho。变体的完整持久化在后续策略批次实现；不要将裸 `state_dict()` 当作具备来源验证的 checkpoint。共享核心源码改变会改变模型身份，旧源码身份的 checkpoint 不自动兼容。

## P1：登记完整图内部更新探针

工具仅做数值可行性验证，不连接设备、不解释飞行动作。固定协议来自 [P1 子规格](../specs/spec-008-trainable-snn/p1-fullgraph-probe.md)：全部原始 `R1-R6` 每 67 步接受 68.75 mV 增量，dt=0.1 ms，共 200 步；固定 `L1` 最终膜电位对 -50 mV 求 MSE，Adam lr=0.01 仅更新一次内部类型 theta。没有可学习输入或动作读出。相同初态输入分别重放更新前、更新后、恢复零 theta 和重新加载 checkpoint；同时复核原始完整图整数计数摘要。

```sh
PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.probe \
  --graph data/malecns_v2/full_reference.npz \
  --out data/trainable_probe_NEW_RUN
```

输出必须是新目录。父进程先写 `registration.json`，再启动 CPU float32 / 4 线程子进程。墙钟预算 600 s，RSS 上限 6 GiB；启动前按可用内存减去系统额外预留 **1 GiB** 收紧上限，若无剩余预算则拒绝启动。运行中每 0.1 s 采样进程树 RSS、系统可用内存和墙钟；超过预算会终止子进程，写明失败原因。预留按用户 2026-10-19 的要求调整，见 [Decision 012](../.ai/decisions/012-one-gib-system-reserve.md)；既有 4 GiB 登记和失败记录保持原值。采样监督不是操作系统级硬内存隔离。

`result.json` 是父进程最终状态；`worker_result.json` 保存完成数值流程的阶段时间、梯度、参数和响应变化、恢复比较及图身份。正常完成还会有 `resources.csv`、`worker.log`、`model_identity.json`、`baseline.npz`、`update.npz`、`responses.npz` 和 `checkpoint/`。仅有日志、checkpoint 或子进程退出码 0 都不能判为完成。损失上升会如实记录，不通过换目标制造下降；非零内部参数和下游响应变化也只证明一次数值更新，不能证明学会目标跟随。

新登记与数值结果使用 v2 证据模式。登记摘要绑定固定协议、资源配置和实现摘要；worker、三个 NPZ 和 checkpoint 都关联同一登记/协议身份，结果还记录全部数值产物的 SHA256。通用资源监督的 `completed` 仅表示进程正常退出。随后 `probe_evidence` 只读小型 JSON/NPZ，核对身份、摘要、维度、有限值、单次 Adam 与增益关系，并从保存数组复算参数/响应变化、损失和两次恢复判据；通过后才把 P1 最终状态设为 `passed`。它不导入 Torch、不重载完整图或重新仿真。缺失、损坏、自相矛盾或来自其他登记的证据写为 `invalid_probe_evidence`；软件夹具校验通过不构成完整图运行证据。

checkpoint 由 `parameters.npz` 与 `checkpoint.json` 组成，禁止 pickle。保存原始精度 theta、完整模型/图/类型分组/输入集合身份和数组摘要。`learning.checkpoint.load_checkpoint(directory, reference=...)` 必须匹配另行验证的完整参考图，拒绝身份、精度、形状或有限性不符。数值夹具只可用 `restore_checkpoint` 恢复到身份相同的夹具，不可进入生产加载。

2026-10-17 首次实际登记 `data/trainable_probe_20261017_p1/`：可用内存 **3,674,931,200 bytes**，低于 4 GiB 预留，结果 `resource_preflight_failed` / `not_launched`。没有加载全图、计算梯度或生成模型，**P1 的完整图学习验收仍未通过**。首轮 v1 原始文件保持不变；原工具的 16 项检查未覆盖完整证据验证，独立审查的缺口及本次修复见 P1a 返工记录。后续只有资源条件变化且工具验收完成后，才能在新目录登记真正的全图计算，不能重复该失败作为进展。

上述为首次失败的历史状态。2026-10-19按用户指定的1GiB预留执行新登记 `data/trainable_probe_20261019_p1_r3/`，完整166,700节点和25,582,938边实际完成一次内部增益更新：**采样峰值1.830GiB、总墙钟362.64秒**，其中反向291.70秒。全部11,752组theta均为候选变量，本次仅R1–R6组实际变化，gain从1变为0.99500018；809个L1最终电位改变，MSE从163.01246降至161.55836。归零恢复、checkpoint重载和原始图摘要检查通过，父进程证据验证后返回 `passed`。

四次前向各模拟20ms，不能把这次短窗峰值当作长轨迹训练的固定需求，也不能将直接下游电位优化说成神经策略或多跳视觉通路已经可用。完整指标、身份、原始记录位置和证据边界见[P1实测报告](../specs/spec-008-trainable-snn/p1-run-20261019-r3.md)。
