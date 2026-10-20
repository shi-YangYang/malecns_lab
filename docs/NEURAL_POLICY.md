# 有状态三轴神经策略接口

`malecns_lab.learning.neural_policy` 将图像接入完整神经网络，再由所有下降神经元的活动产生 `vx/vz/wz` 张量。当前完成的是可微接口和显式小电路数值检查；尚未训练出目标跟随策略，没有本接口的完整图任务成绩或设备效果证据。

## 路径和模型身份

```text
BGR 图像 → target_mask（12×16 红色占用）→ SpatialDrive → 完整 LIF 状态
                                                        ↓
                               所有 DN 的前半窗/后半窗电位均值
                                                        ↓
                                    固定物理尺度 → Linear → tanh
                                                        ↓
                                           vx / vz / wz（SI）
```

生产入口 `NeuralPolicy.from_reference` 先调用完整图及元数据校验，再选择全部精确 `R1-R6` 输入和全部 `descending_neuron` 输出。当前参考数据对应 3,377 个输入、1,314 个 DN，特征宽度为 2,628；这些数量来自现有图和元数据，不在策略中硬编码。未知类型的 DN 也保留，不按响应或侧别筛选。

背景必须显式选择 `zero_background` 或 `fixed_lamina_8mv`。后者仍是工程背景假设：固定 L1–L5 平衡偏移 8 mV，`rho=0` 不训练。策略可训练参数严格为 `brain.theta`、`readout.weight`、`readout.bias`；完整原始连接和神经群组保持。`for_numerical_test` 仅接受显式 `numerical_test_only` 图与映射，不是缺失真实数据时的回退。

独立 `policy_id` 绑定脑模型、完整选择、空间与事件映射、相机、精度、时间配置、特征顺序/尺度、初始化种子与相关源文件 SHA。theta 和读出学习后的数值不进入结构身份。空间映射是可追踪的工程编码，不代表测得的视网膜拓扑。

## 明确三种时间

`NeuralPolicyConfig` 必须提供四项，不预设一个尚未验证的训练窗口：

| 字段 | 含义 |
|---|---|
| `frame_ms` | 一幅图像推进的神经模拟毫秒，须能被 LIF dt 整除且包含偶数步 |
| `precondition_ms` | 场景开始的空白预处理神经毫秒，可为 0，须能被 dt 整除 |
| `control_dt_s` | 环境一次动作推进的秒数，与神经时间分别登记 |
| `readout_seed` | 独立 CPU Generator 的初始化种子，不改变全局 Torch 随机状态 |

实际计算墙钟由运行方测量，与上述两种模拟时间不同。固定背景预处理不自动意味着达到稳态。过去短刺激实验的撤除后响应不能证明任何默认控制窗口足够；实际训练必须另行登记窗口和计算预算。

生产配置示例是一个构造函数，不会执行预处理或飞行。`protocol` 应来自之后冻结的训练协议，不能直接把数值测试里的短窗口用于完整图研究：

```python
import torch
from malecns_lab.brain.trainable import TrainableLIFConfig
from malecns_lab.learning.neural_policy import NeuralPolicy, NeuralPolicyConfig
from malecns_lab.senses.spatial_drive import SpatialDriveConfig
from malecns_lab.senses.target import TargetCamera

def build_policy(protocol):
    return NeuralPolicy.from_reference(
        "data/malecns_v2/full_reference.npz",
        background=protocol["background"],
        config=NeuralPolicyConfig(**protocol["policy_config"]),
        camera=TargetCamera(**protocol["camera"]),
        drive_config=SpatialDriveConfig(**protocol["drive"]),
        lif_config=TrainableLIFConfig(**protocol["lif"]),
        dtype=torch.float32,
    )
```

## 训练张量与完整状态

构造时不运行神经动力学。每个场景显式调用 `state = policy.initial_state()`：脑与事件驱动一起重置，在 `no_grad` 下完成空白预处理。之后每幅图像使用：

```python
result = policy.forward_tensor(bgr_frame, state)
action_si = result.action         # shape (3,): vx, vz, wz
dn_features_mv = result.features_mv  # shape (2 * N_DN,)
state = result.state
```

`forward_tensor` 自身不关闭梯度。`features_mv` 是每神经步复位后的膜电位均值，依次排列前半窗全部 DN、后半窗全部 DN，各半窗内按 body ID 升序。读出实际输入为 `(features_mv - v_rest) / (v_threshold - v_rest)`；这是物理尺度，不是从验证/测试数据拟合的统计量。

读出只有一层线性权重和 bias，初始化权重标准差 `0.01/sqrt(2*N_DN)`、bias=0。tanh 后按 `[0.3, 0.2, 0.5]` 缩放；轴单位为 m/s、m/s、rad/s，FLU 正 yaw 向左。初始化随机方向没有正确动作含义。检测误差、估计距离、教师动作、场景 ID、环境真值都不进入读出；感知前端不训练。

外部训练器可直接以三轴张量计算动作损失，优化上述参数。`BodyVelocity` 会转换 Python 浮点，不能放在这条反向路径中。本批没有训练循环、数据拟合或闭环优化。

`PolicyState` 保存所有神经元的膜电位、突触状态、不应期、完整延迟队列，以及不可变事件相位/累计次数和两个步号。神经/驱动步号都必须等于 `precondition_steps + frame_index * frame_steps`。跨帧不得重新调用 `initial_state()`；跨场景必须重新调用。TBPTT 边界使用 `state = state.detach()`，仅切断状态梯度，数值、队列、相位及时间全部保持，也不会再次预处理。不同身份或不一致的状态会报错，不静默清零。

## 环境边界

`NeuralController` 是已有 `FollowPolicyInput -> BodyVelocity` 的独立适配器。创建 controller 不预热，首次调用及每个新场景必须先 `reset()`。它按 policy 登记的相机与 `control_dt_s` 检查输入时间，时钟不符报错而非自动继续旧场景。

```python
from malecns_lab.learning.neural_policy import NeuralController
from malecns_lab.learning.follow_task import rollout

controller = NeuralController(policy)
controller.reset()
result = rollout(scene, controller, policy_name=policy.policy_id,
                 camera=policy.camera, dt_s=policy.config.control_dt_s)
```

此示例需先有已登记策略和场景；不能据此认定有合格控制器。controller 在 `no_grad` 下推进神经网络，随后转换 `BodyVelocity`，其他轴为零。它忽略 `observation` 的误差/距离/有效性；现有 rollout 仍负责目标丢失保护和分别记录原始动作、实际执行动作及介入。神经策略不能把外部保护效果算作自身能力。

`controller.neural_time_ms` 包含预处理；`controller.control_time_s` 从场景 0 秒开始。两者在未 reset 时为 `None`。原始动作时间、计算墙钟及实际设备速度尚需后续日志/部署测量。

## 当前边界

本接口只支持 CPU float32/float64。旧 `learning.checkpoint` 是独立的 gain-only 格式，不保存新读出或全部策略配置；不要用它保存“完整策略”。新的策略 checkpoint、恢复训练、实际完整图学习、冻结脑/急性恢复/直接视觉/拓扑对照和设备接入仍待后续子批。

本次只运行显式五节点电路检查：电位时间聚合与线性代数、状态/相位连续性、梯度和联合更新、精度/身份/时钟拒绝、独立随机初始化、固定背景与部署边界。它证明软件接口的约定，不能代替真实连接组学习结果。规格见 [P2e](../specs/spec-008-trainable-snn/p2e-neural-policy.md)。
