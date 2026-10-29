# 训练数据、运行与恢复

本手册收拢历史目标跟随训练工具的操作与存档契约；实际结果和未完成项统一见 [Spec008 历史](../../history.md#spec-008)。它不是 X2 多行为按钮的验收说明，也不授权恢复已暂停的训练。

所有命令从项目根目录执行，输出必须为新目录。旧 run、checkpoint 和 `sources` 快照按原来源保留：恢复要求匹配源码/运行时，不能用当前模块名改写旧身份或直接续接耗尽预算的 run。完整脑串行运行，保留 Noria/Docker；通常申请 6 GiB，按 `min(6 GiB, available−1 GiB)` 收紧，每段 600 秒、0.1 秒采样。具体登记优先，采样监督不是 OS 硬隔离。

<a id="neural-training-data"></a>
## 仿真教师序列

`learning.neural_data` 只运行[速度级标记环境](offline.md#follow-task)，不加载脑或设备。它固定收集按 ID 排序的四个 `train-sine-*` 完整场景，默认各 8 s/80 帧，不允许随机逐帧划分。

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.neural_data \
  --out data/neural_training_sequences_new
```

默认相机 160×120 BGR，环境步长 0.1 s，响应常数 0.25 s；API 可给 `TargetCamera`，CLI 可用 `--dt-s / --response-tau-s`，全部进入身份。采集行为来自 `visual_teacher`，监督另由同帧 `truth_teacher` 生成，不能把执行动作当正确标签。

| `sequence.npz` 字段 | dtype / shape | 含义 |
|---|---|---|
| `frames` | uint8 `[N,H,W,3]` | 同时刻无损 BGR |
| `time_s` | float64 `[N]` | 连续环境时间 |
| `teacher_raw` | float64 `[N,3]` | 未限幅真值教师 |
| `teacher_target` | float64 `[N,3]` | `limit_follow_action` 后的监督 |
| `executed_action` | float64 `[N,3]` | 视觉策略实际执行动作 |
| `valid` | bool `[N]` | 同帧检测有效性，作为动作损失掩码 |

轴为 `vx/vz/wz`，FLU，单位 m/s、m/s、rad/s，上限 0.3、0.2、0.5。无效帧保留像素和原标签，仍推进神经状态，仅屏蔽损失，不能删帧或改零标签。原始 `summary.json/trajectory.jsonl` 保留真值与介入；神经策略只接收图像。

`manifest.json` 最后封存，绑定场景、train split、相机/时钟、教师、源码、文件 SHA、数组模式及计划/实际样本数。只有全场景完整且无环境 failure 才为 ready；失败为 collection_failed，未封存目录不可加载。

```python
from malecns_lab.learning.neural_data import read_neural_data
dataset = read_neural_data("data/neural_training_sequences_new")
```

读取使用 `allow_pickle=False`，检查字段、时钟、限幅、valid、源码与真实 rollout；重放轻量环境逐帧核对像素、真值与状态，防止只改 split/hash 把验证材料冒充训练。数组不可写。数据格式不含神经特征或模型结果。

<a id="joint-probe"></a>
## 两帧联合更新探针

`learning.joint_probe` 验证一次动作损失能否同时更新内部 theta 与读出，不是完整训练或闭环。当前 r2 稳定分组裁剪协议依赖已封存 r1 比较锚点；这是既有协议的复现入口，不是可任意选帧的新实验启动器。

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.joint_probe \
  --graph data/malecns_v2/full_reference.npz \
  --dataset data/neural_teacher_20261019_r1 \
  --out data/joint_update_new
```

封存数据同级须保留 `joint_update_20261019_r1/`；登记绑定其原 registration/result 与比较文件摘要，核心、策略、数据、checkpoint、监督器来源须一致。无裁图、缩窗或换帧参数。r1 部分数值文件后来才冻结摘要，不能称它们由旧终态清单证明。

固定使用 `train-sine-combined` 帧 0/1，两帧均有效；60 ms 空白无梯度预处理后，连续两个 50 ms 图像窗，中间无 reset/detach。所有 DN 两半窗电位进入三轴线性读出。

Adam 仅一步：theta lr=0.001，读出 lr=0.0001；两帧归一化动作 MSE 加 `1e-4*mean((gain−1)^2)`。theta 与 weight+bias 分别裁剪 L2≤1：先验证原始梯度有限，float64 求组范数，再将 `min(1,1/(norm+1e-6))` 转 float32 乘原梯度。固定 rho 不训练。

baseline、updated、acute_restore（原 theta/新读出）、full_restore（全部原参数）、checkpoint_reload 五分支各自 fresh state。更新前与 r1 核对初值精确相同、动作/特征/梯度 `rtol=1e-6, atol=1e-7`；超差即停，不事后放宽。重载前释放旧图和计算图，避免两份全图常驻。

结果严格区分 `protocol_complete`、`joint_numeric_update` 和 `internal_action_effect`。协议完整可退出 0，即使 loss 变差或动作影响为阴性；未知为 null。父进程不加载脑，独立复算损失、组裁剪、float32 Adam（含二阶矩下溢）及恢复关系；exit 0/checkpoint 存在均不足以判完成。

保留 registration、输入/DN 顺序、初参数、原始梯度、update、r1 比较、实际 Adam 数组、五分支响应、参数 checkpoint 与原始监督终态。这里的 `adam_state.npz` 是一次更新证据，缺少完整状态与游标，不可作续训档。

更早的 `learning.probe` 只验证 gain-only 内部数值更新：全部 R1–R6 每 67 步加 68.75 mV，共 200 个 0.1 ms 步，L1 最终电位对 −50 mV 的 MSE，Adam lr=0.01 更新一次 theta。它没有动作读出：

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.probe \
  --graph data/malecns_v2/full_reference.npz --out data/trainable_probe_new
```

输出 registration、baseline/update/responses、gain-only checkpoint 与监督终态；`probe_evidence` 不载脑，核对保存数组并复算更新前后、归零恢复和重载，只有父证据通过才为 passed。下游电位变动不代表已学会动作。

<a id="training-runner"></a>
## 有界连续训练

`learning.train_snn` 是早期 CPU 数值 runner，生产仅接受完整参考图与封存四场景，一个 epoch 为 160 个两帧窗口。每次显式处理 1–4 窗后退出，不自动跑全轮或重试。GPU 分支见下节；示例不会使旧来源兼容当前代码。

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.train_snn \
  --graph data/malecns_v2/full_reference.npz \
  --dataset data/neural_teacher_20261019_r1 \
  --out data/neural_training_joint_new --mode joint --windows 1

PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.train_snn \
  --run data/neural_training_joint_new --windows 1
```

首次门槛是两个独立 OS 进程各一窗，核对恢复后才扩大至每次最多 4 窗。续跑不能换图、数据、模式或源码。`frozen_brain` 使用新目录、原 theta=0 和同读出种子，只训读出，不能复用 joint 最终参数。

每帧神经 50 ms，dt=0.1 ms；场景初 60 ms 预处理。窗内两帧连续反向，更新后 detach；跨场景才重置活动，参数/Adam 保留。无效标签仍推进状态；全无效窗跳整个更新，第一窗全无效拒绝零步存档。

`run.json` 固定图/数据/协议/模式/来源；`segments/` 保存登记、资源、恢复值和原始监督结果；每窗依次封存 `evidence.npz → event.json → prefix.jsonl → checkpoint/ → window.json`。缺最后 window manifest 的尾部不计进度。`progress.json` 从真实封存链重算，`epoch_complete` 仅指训练预算完成。

只有资源/进程失败且有效封存前缀获准时，下一次显式调用才能恢复；数值/协议失败、坏父链、重复窗或损坏存档不静默回滚。OS 文件锁防并行；锁文件存在不等于进程仍活着。资源采样时钟不足可报 `clock offset unresolved`，不伪造覆盖。

<a id="gpu-training"></a>
## MPS 训练与数值版本

`learning.train_gpu` 将感觉掩码/事件、完整脑、读出、损失、反向、稳定裁剪及 Adam 参数/m/v/step 放在 MPS；CPU 负责资产、调度、I/O 与独立验算。禁用 CPU 算子 fallback。它与 X2 仿真 GPU 渲染是不同计算链。

修复协议使用稳定 logit 损失、100 步递归反向范围、全部 1000 步前向及原始 theta 梯度 L2 上限 1000。每 50 步分段重算限制中间张量内存，100 步 credit horizon 限制梯度传播，两者不能混称。该协议继承旧参数但清零 Adam，新场景用 GPU float32 感觉，不是旧协议第 161 窗。

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m malecns_lab.learning.train_gpu --initialize \
  --parent-run data/neural_training_mps_20261019_r3 \
  --out data/neural_training_gpu_new --credit-horizon 100 --theta-norm-limit 1000

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m malecns_lab.learning.train_gpu --gate --run data/neural_training_gpu_new

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m malecns_lab.learning.train_gpu --run data/neural_training_gpu_new --windows 1
```

gate 是可丢弃的负标签联合更新，核对内部梯度、参数变化与固定特征纠错，不污染正式初参。首窗后新进程恢复第二窗，通过实际门槛才可每次最多 4 窗，总新增预算 160 窗。`initialization/gate/segments` 分别保存来源、门槛与更新/恢复证据；验算下载设备值但不回写优化器。失败保留并阻断继续。

历史 `learning.train_mps` 后端为 `mps_forward_backward_cpu_adam`：脑/读出前后向在 MPS，感觉相位 NumPy float64，CPU Adam 和 float64 范数裁剪，不能称全链路 GPU。迁移从完整 CPU 末档保留参数、Adam、神经/感觉状态、前缀及累计 160 窗预算，不重放旧窗或重新预热；新场景才重置活动。

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m malecns_lab.learning.train_mps --cpu-run /path/to/compatible_cpu_run \
  --out data/neural_training_mps_new --windows 1
```

MPS current/driver 与 RSS 分别记录，不能相加为总物理内存；allocator 受登记预算约束。CPU/GPU 不承诺逐位训练等价，但自身有限性、算术、文件身份和精确恢复约束不放宽。历史 `neural_training_mps_20261019_r3` 与 `neural_training_gpu_recovery_20261019_r1` 均已耗尽登记预算，不再续跑；效果与方向错误边界见历史。

<a id="checkpoints"></a>
## 存档类型

| 模块/格式 | 保存内容 | 用途限制 |
|---|---|---|
| `learning.checkpoint` | gain-only theta 与模型身份 | 拒绝 tonic/完整策略；不能续训 |
| `learning.policy_checkpoint` / `cnsbridge.neural_policy_checkpoint.v1` | theta、读出 weight/bias 与完整 policy 身份 | `deployment_parameters_only`，不含 Adam/活动状态 |
| `learning.training_checkpoint` / `cnsbridge.training_checkpoint.v1` | 参数、实际 Adam、全部状态与游标 | `detached_two_frame_tbptt_continuation`，匹配 CPU 来源/运行时 |
| GPU runner 存档 | 各后端自己的参数/状态/来源与封存链 | 不改文件名混用 CPU 格式；由对应 runner 校验 |

### 策略参数

```python
from malecns_lab.learning.policy_checkpoint import (
    save_policy_checkpoint, read_policy_checkpoint,
    restore_policy_checkpoint, load_policy_checkpoint,
)
save_policy_checkpoint(policy, "data/policy_new", run_identity=registered_identity)
metadata, arrays = read_policy_checkpoint("data/policy_new")
restore_policy_checkpoint(policy, "data/policy_new")
loaded = load_policy_checkpoint("data/policy_new", "data/malecns_v2/full_reference.npz")
```

`parameters.npz` 严格为 theta/readout_weight/readout_bias；最后写 `checkpoint.json`，绑定 dtype/shape/SHA 与全部 policy 身份。固定 rho=0 的 8 mV 背景由身份约束，不作为训练参数保存。`run_identity` 是有限 JSON 数据并参与 ID。

轻量读取不构造脑；生产加载再次验证完整参考图、成员和构造身份，拒绝 numerical_test_only/research 角色。恢复先验证全部字段与目标，再执行拷贝，错误不应部分修改模型。已有目录拒绝覆盖，NPZ 禁止 pickle。源码变更需显式迁移/重验，不能改 hash 伪装兼容。

这类存档不保存 Adam、隐藏状态、延迟、感觉相位、随机游标或样本进度；加载后 `initial_state()` 开新场景，不能声称中途精确续训。

<a id="training-continuation"></a>
### CPU 训练续跑

`make_training_optimizer(policy, mode=...)` 固定参数名称与顺序。joint：theta lr=0.001、读出 lr=0.0001；frozen_brain：theta 不入优化器，仍保存其值和动态。CPU float32 Adam 固定 betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad/foreach/fused=False；maximize/capturable/differentiable/decoupled_weight_decay 均 False，rho 不入组。

保存顺序：两帧连续前向 → 反向与登记裁剪 → Adam step → `state.detach()` → `zero_grad(set_to_none=True)` → 写日志前缀 → `save_training_checkpoint`。活动状态来自更新前参数，后续携带它前进是明确的 TBPTT 近似，不重演前缀。

`context` 恰含以下字段；`expected` 恰含表中前四个外部期望摘要，读取/恢复不可省略：

| 字段 | 约束 |
|---|---|
| `training_run_id / protocol_id / dataset_id / dataset_manifest_sha256` | 各为 64 位小写十六进制摘要 |
| `scenes` | 按训练顺序的 `{scene_id,frame_count}`，名称唯一，帧数正偶数 |
| `epoch / scene_index / scene_id / next_frame` | 索引从 0 开始，next_frame 偶数且 2≤值≤场景长度，等于 state.frame_index |
| `completed_windows` | 按场景与 next_frame 独立推导，每窗两帧 |
| `optimizer_steps` | 正整数≤completed_windows，等于每个活动参数实际 float32 Adam step |
| `parent_checkpoint_id / log_prefix_sha256` | 首父档 null，其余为真实父档和日志摘要 |

真实数据顺序、日志文件和父链由 runner 核验，摘要字符串本身不证明执行。

```python
from malecns_lab.learning.training_checkpoint import (
    save_training_checkpoint, read_training_checkpoint,
    restore_training_checkpoint, load_training_checkpoint,
)
save_training_checkpoint(policy, optimizer, state, new_directory,
                         mode="joint", context=context)
archive = read_training_checkpoint(directory, expected=expected)
state, context = restore_training_checkpoint(policy, optimizer, directory, expected=expected)
# 释放旧完整图后，才用 load_training_checkpoint(directory, reference_path, expected=expected)
```

恢复不预热、不推进帧、不再次应用更新；先完整验证/准备张量再修改对象，目标参数梯度必须 None。`read_training_checkpoint` 依赖 Torch，轻量监督父进程不应用它判断终态。

根 `checkpoint.json` 最后封存，绑定 `training.npz`、`policy/checkpoint.json`、`policy/parameters.npz`、源码、Python/NumPy/SciPy/Torch、CPU 架构与线程。只接受相同运行时设置；不保存图矩阵或可执行对象。

状态包含全节点 v/g、int64 refractory_until、完整到达序延迟队列（默认 18 槽），float64 感觉 phase、int64 event_counts、映射/驱动/dt/时钟。Adam 按稳定参数名保存 step/exp_avg/exp_avg_sq；非负零二阶矩合法，保留 float32 下溢。

缺失/未知/重复字段、坏摘要、非有限、错误模式/时钟/父链均拒绝。未封存尾部保留但不能恢复，禁止覆盖旧目录。首版仅接受至少一次真实更新后的边界；未来加入随机算子、噪声或 dropout，必须升级恢复契约。
