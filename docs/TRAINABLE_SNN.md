# 神经计算与策略接口

本文说明可训练完整图的数值、输入与有状态读出契约。完整图资产见[数据模型](data-model.md)，训练与持久化见[训练手册](training.md)，实际阶段结果见 [Spec008 历史](../history.md#spec-008)。本文中的 API 示例不授权自动运行完整脑。

<a id="trainable-snn"></a>
## 完整图可训练核心

`brain.trainable.TrainableLIF` 的身份为 `cnsbridge.trainable_lif.v1`。生产 `from_reference` 必经完整参考图、注释与 provenance 校验，保留全部节点、边、自环和原始符号，不按活跃度裁剪。`for_numerical_test` 具有 `numerical_test_only` 身份，只供数值检查，不能成为缺图回退。

独立训练环境为 `.venv-train`，依赖由项目 `train` extra 固定；不要改已有设备环境。CPU 核心允许 float32/float64，MPS 是单独身份的后端，见[GPU 训练](training.md#gpu-training)。数值 CPU 核心与已经删除的 CPU X2 仿真部署不是同一对象。

```python
import torch
from malecns_lab.brain.trainable import TrainableLIF

# input_body_ids 必须来自已登记的真实输入集合；空列表仅演示接口。
model = TrainableLIF.from_reference(
    "data/malecns_v2/full_reference.npz", input_body_ids=[], dtype=torch.float32,
)
state = model.reset()
spikes, state = model.step(torch.zeros(model.graph.n), state)
state = state.detach()
```

原始有符号 int64 计数保存在只读 `model.graph.counts`；浮点计算矩阵另存，float32 舍入不覆盖原始资产。模型身份包含图来源、原始计数摘要、节点/边/突触总数、类型分组、输入 body ID、精度、参数与实现 SHA。gain-only 核心的 `state_dict()` 仅含 theta；裸参数文件不构成可复现模型。

### 方程、单位与事件顺序

方程参考 [Shiu 等官方实现](https://github.com/philshiu/Drosophila_brain_model/blob/main/model.py)，核对版本 SHA256 为 `fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e`。这是方程启发的工程模型，不声称逐步复现 Brian2 或测得 MaleCNS 生理参数。

单位 mV/ms：`v_rest=v_reset=-52`、`v_threshold=-45`、`tau_m=20`、`tau_syn=5`、`refractory=2.2`、`delay=1.8`、`weight=0.275`、`dt=0.1`。步长必须整除延迟与不应期，不自动取整。

```text
dv/dt = (v_rest - v + g) / tau_m
dg/dt = -g / tau_syn
a = exp(-dt/tau_m), b = exp(-dt/tau_syn)
v_next = v_rest + a*(v-v_rest) + g*tau_syn/(tau_m-tau_syn)*(a-b)
g_next = b*g
```

时间常数相等时，g 的积分系数取极限 `dt/tau_m*a`。每步 k 固定顺序：

1. 队首到期增量加入 g。
2. 活动细胞解析积分；不应期 v/g 冻结，但到达增量保存在 g。
3. 仅指定输入细胞接受显式外加 mV；非输入细胞非零输入报错。核心不自行生成视觉、Poisson 或默认驱动。
4. 允许发放且严格 `v > threshold` 时产生二值脉冲，随后 v=reset、g=0。
5. 非输入细胞在 `k+refractory_steps` 再次允许积分/发放；输入细胞无不应期，连接增量在 `k+delay_steps` 到达。

官方未定义的神经元状态 w 不在此引入；整数步调度、输入事件、共享增益与代理梯度均是明确的工程协议。

### 参数与梯度

```text
gain = 1 + 0.5*tanh(theta)
delayed_increment = W_base @ (gain[group(pre)] * spikes)
```

theta 初值 0，按突触前原始 type 共享。缺失、空串和全空白注释进组 0，其余原始字符串不修剪、不改名，按 Unicode 排序。`nextafter` 防止浮点舍入触达 0.5/1.5 端点，正增益保持源符号。

前向脉冲严格二值；反向代理为 `1/(scale*(1+abs((v-threshold)/scale))**2)`，默认 scale=1 mV。复位门与不应期门停止梯度，传播脉冲保留代理梯度。稀疏反向计算全部细胞的 `W_base.T @ gradient`，不删除静默细胞潜在梯度，不构造密集 N×N 图。

CPU backend 为 `scipy_cpu_csr_forward_torch_csr_adjoint`：从原始 CSC 建 `(post,pre)` 前向 CSR 与 `(pre,post)` 伴随 CSR；SciPy 前向只读视图共享 Torch 存储。每步复用固定矩阵和全细胞向量，伴随为非持久化 buffer，不增加参数或存档。支持连续/非连续 CPU float32/float64；不承诺二阶梯度。

`run(inputs,state)` 接受每步 N 维 mV 向量；`reset()` 只清动态状态，`state.detach()` 截断包括延迟队列在内的全部梯度但保留值与时钟。跨模型身份、错误精度/形状、非有限输入/参数/状态均拒绝，源码变更也不自动兼容旧 checkpoint。

### L1–L5 背景变体

`LaminaTonicLIF` 使用独立身份 `cnsbridge.lamina_tonic_lif.v1`，精确选择原始类型 L1–L5；各组必须存在，不能与外部输入重叠。测试入口才允许显式类型子集。

```text
offset[type] = 16 * sigmoid(rho[type]) mV
integrated_v += (1-exp(-dt/tau_m)) * offset
```

rho=0 对应 8 mV，这是工程初始化而非生理常数；`nextafter` 保持偏移严格在 (0,16)，非成员为零。不增加 g，不扩大输入集合，不取消不应期，也不向 DN 添加动作偏置。`learnable` 将 rho 注册为参数，`fixed` 为 buffer，两者身份不可交换；严格零背景用原核心。

背景发放不证明视觉传播。抑制性 R1–R6 加全网静息的零背景模型可能无法启动非输入脉冲，不能靠输出偏置掩盖。旧 gain-only `learning.checkpoint` 拒绝 tonic；完整固定背景策略由 `policy_checkpoint` 保存，不能把两种格式混用。

<a id="neural-interface"></a>
## 完整输入输出与空间事件

`load_policy_neuron_map(graph, reference_path)` 接收已通过 `load_training_graph` 的完整图，从同一 NPZ 绑定 `meta/body_ids/annotation_fields` 及缺失数组，沿用 annotation hash 的顺序、dtype、shape 和字节契约。这是元数据检查，不能替代之前的全图验证。

| 集合 | 精确规则 | 当前资产数量 |
|---|---|---:|
| 输入 | 原始 `type == R1-R6` | 3,377 |
| 输出 | 原始 `superclass == descending_neuron` | 1,314 |

成员由资产推导，不硬编码数量，不用 DN 名称前缀，保留未知类型与所有侧别。map 身份绑定来源、图、注释、实际 body ID/索引和规则；不替换 V2 原有 33 细胞探针，不裁图。

`build_spatial_input_map` 对应 `target_mask` 的 12×16 红色占用图：行向下、列向右，格号 `row*16+column`。原始 rootSide 分组，缺失单独一组，空串与缺失分开；其余字符串按 Unicode 排序。组内 body ID 升序后，共用 `PCG64(1701)` 顺次置换；第 j 个成员分配 `j % 192` 格。每侧接收同一整幅图，格覆盖计数/空格随身份保存。

这是工程映射，不是解剖眼柱或双眼标定。映射身份绑定算法、种子、NumPy 版本及每细胞格号，返回数组不可改写。

`SpatialDrive` 默认 `dt_ms=0.1`、最大 150 事件/s、每事件 68.75 mV。相位由独立 `PCG64(1702)` 按 body ID 升序生成；每步：

```text
phase += occupancy[pixel_cell] * max_rate_hz * dt_ms / 1000
若 phase >= 1：发放一次事件，phase -= 1
```

单步可能超过一次事件的配置拒绝；mask 须为有限 `[0,1]` 且形状正确，失败不改状态。空白帧无新事件但仍推进时钟，相位跨帧保持。`reset()` 恢复相同初相位；`export_state/restore_state` 绑定映射、驱动、dt、步数和各细胞事件计数，不能只恢复相位值。

`DriveEvents` 默认只含本步输入索引与增量；`to_dense()` 或 `to_torch(dtype=...)` 才创建全节点向量，非输入节点为零。CPU 接口支持 float32/float64；GPU 感觉使用独立数值身份。一个视频帧不等于一个神经步，控制器必须明确帧时长。

<a id="neural-policy"></a>
## 有状态三轴策略

```text
BGR → target_mask → SpatialDrive → 完整 LIF
                  → 所有 DN 前/后半窗电位均值 → 物理尺度 → Linear → tanh → vx/vz/wz
```

`NeuralPolicy.from_reference` 校验全图与输入输出；当前特征为 2×1,314=2,628 维。背景显式选择 `zero_background` 或 `fixed_lamina_8mv`；后者 rho 固定 0。可训练参数严格为 `brain.theta`、`readout.weight`、`readout.bias`，不改原始拓扑。`policy_id` 绑定图、选择/顺序、编码、相机、精度、时间、初始化和源码；学习后的数组值由参数存档另行绑定。

| `NeuralPolicyConfig` 字段 | 契约 |
|---|---|
| `frame_ms` | 每图推进神经毫秒，dt 整除且为偶数步 |
| `precondition_ms` | 新场景空白预处理，可为 0，dt 整除 |
| `control_dt_s` | 每动作的环境秒数，不等于神经时间或墙钟 |
| `readout_seed` | 独立 CPU Generator 初始化，不改全局 RNG |

构造不运行神经动力学。每场景调用 `initial_state()`，脑和驱动一起重置并在 no_grad 下完成空白预处理；跨帧保持返回状态：

```python
state = policy.initial_state()
result = policy.forward_tensor(bgr_frame, state)
action_si = result.action          # shape (3,), vx/vz/wz
dn_features_mv = result.features_mv
state = result.state
```

`forward_tensor` 不自动关闭梯度。特征为每步复位后电位均值，先前半窗全部 DN，再后半窗全部 DN，各半窗按 body ID 排序；读出输入 `(features-v_rest)/(v_threshold-v_rest)`。线性初始权重标准差 `0.01/sqrt(2*N_DN)`，bias=0，tanh 后乘 `[0.3,0.2,0.5]`，对应 m/s、m/s、rad/s，FLU 正 yaw 向左。随机初始化没有正确动作含义。

检测误差、估计距离、教师、scene ID 和真值均不直接进入读出。训练损失使用张量；转成 `BodyVelocity` 的 Python 浮点会断开梯度。感觉前端不训练。

`PolicyState` 保存全节点 v/g、不应期、完整延迟队列、事件相位/累计数及两种步号；均应等于 `precondition_steps + frame_index*frame_steps`。TBPTT 使用 `state.detach()`，不重置数值、队列、相位或时间；跨场景才重置。

### 环境适配与用途边界

`NeuralController` 实现 `FollowPolicyInput -> BodyVelocity`，首次调用和每个新场景前显式 `reset()`；核对相机、`control_dt_s` 与时间，不自动继续旧场景。在 no_grad 下计算，再转为动作；其它轴为零。

controller 不使用 observation 的误差/距离/有效性；外层 rollout 负责丢失目标保护，分别保存 raw、executed 与 intervention。`neural_time_ms` 含预处理，`control_time_s` 从场景 0 开始，未 reset 时均为 None。

该类型检查不是研究角色的安全隔离器；研究图工厂产出的策略仍只能用于[离线研究](offline-research.md#research-policy)。状态和存档的真实身份必须匹配，不能因维数相同而交换 G/R 状态。API、数值检查、内部更新与合格设备控制是不同完成条件。
