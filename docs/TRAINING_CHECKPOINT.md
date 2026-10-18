# 神经训练续跑存档

`learning.training_checkpoint` 保存一个已经完成更新的两帧 TBPTT 边界，包含全部模型参数、实际 Adam 状态、全部神经活动状态和下一帧游标。它为[有界连续训练 runner](TRAINING_RUNNER.md)提供存档能力；本页记录的 P3c1 验证没有新增完整图训练或学习收益结论。

它与 [部署参数存档](POLICY_CHECKPOINT.md) 的用途分开。已有 P3b `adam_state.npz` 仅为单次 Adam 的证据，缺少隐藏状态、感觉相位和游标，不能直接作为续训档；部署参数档也会被续训读取入口拒绝。

## 固定 API

```python
from cnsbridge.learning.training_optimizer import make_training_optimizer
from cnsbridge.learning.training_checkpoint import (
    save_training_checkpoint,
    read_training_checkpoint,
    restore_training_checkpoint,
    load_training_checkpoint,
)

optimizer = make_training_optimizer(policy, mode="joint")
```

`policy` 为既有 `NeuralPolicy`。生产构造仅接受经过验证的完整参考图；`for_numerical_test` 单独用于小电路检查，不能通过生产加载入口。

两种模式的参数名称、绑定顺序和选项固定：

| 模式 | theta | 读出 weight/bias |
| --- | --- | --- |
| `joint` | requires_grad=True，Adam lr=0.001 | requires_grad=True，Adam lr=0.0001 |
| `frozen_brain` | requires_grad=False，不进入 Adam | requires_grad=True，Adam lr=0.0001 |

两模式采用 CPU float32、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、amsgrad=False、foreach=False、fused=False；maximize、capturable、differentiable、decoupled_weight_decay 均 False。固定背景 rho 维持 0，不进入参数组。`make_training_optimizer` 显式设置 theta 的 requires_grad；保存和恢复则要求现有模式正确，不隐式切换。冻结模式仍保存全部 theta 和神经动态，不能作为内部学习完成的证据。

保存前，调用方完成以下顺序：

1. 同场景连续处理两帧，使用当前完整状态。
2. 反向、按已登记训练协议裁剪梯度，然后 `optimizer.step()`。
3. `state = state.detach()`，保留每个状态值，仅截断梯度历史。
4. `optimizer.zero_grad(set_to_none=True)`。
5. 写完对应日志前缀，再封存：

```python
metadata = save_training_checkpoint(
    policy, optimizer, state, new_directory,
    mode="joint", context=context,
)
```

损失和梯度裁剪不由此模块决定。活动状态由更新前参数计算，更新后的参数携带它继续下一窗口，这是显式 TBPTT 近似；没有用更新后的参数重演前缀。场景内不 reset；场景末状态保存后由后续 runner 显式开始新场景。

## 上下文和身份

`context` 必须恰好包含以下字段：

- `training_run_id`、`protocol_id`、`dataset_id`、`dataset_manifest_sha256`：64 位小写十六进制摘要，由调用方从登记记录提供。
- `scenes`：按训练顺序排列的 `{scene_id, frame_count}` 列表。名称唯一且非空，帧数为正偶数。
- `epoch`、`scene_index`、`scene_id`、`next_frame`：epoch 和 scene_index 从 0 开始；next_frame 是当前场景下一未处理帧，范围为 2 到场景帧数，必须为偶数并等于 state.frame_index。场景末允许 next_frame==frame_count，保留当前场景末状态。
- `completed_windows`：由 epoch、前面各场景帧数和 next_frame 独立推导。每窗两帧。
- `optimizer_steps`：正整数，不大于 completed_windows；每个活动参数的 float32 scalar Adam step 必须与之相等。允许无效标签窗口只推进状态而未更新，因此不能仅由 optimizer_steps 推断帧数。
- `parent_checkpoint_id`：前一已封存档的摘要；首个封存边界为 null。
- `log_prefix_sha256`：已写日志前缀的摘要。

所有读取和恢复函数都要求 `expected` 参数，且它恰好包含上述前四个摘要。不能省略外部期望身份而只信任档案自报身份。

本模块交叉核对 expected 摘要和档案身份，并核对声明场景与状态游标的一致性。**实际场景顺序/帧数是否符合封存数据集、日志摘要是否对应真实日志前缀、父链是否存在且连续，由[连续训练 runner](TRAINING_RUNNER.md)与登记和真实文件核对。** 摘要字符串本身不证明训练已执行。

## 读取和恢复

```python
archive = read_training_checkpoint(directory, expected=expected)
# 不构造图或模型；TrainingArchive 包含 metadata、parameter_metadata、
# parameters、optimizer_arrays、state，以及返回独立字典的 context 属性。

state, context = restore_training_checkpoint(
    policy, optimizer, directory, expected=expected,
)

# 已释放旧模型后，生产入口从完整参考图重新构造：
policy, optimizer, state, context = load_training_checkpoint(
    directory, reference_path, expected=expected,
)
```

恢复不调用 initial_state/precondition，不推进帧，也不再次应用上次更新。恢复到既有对象前先校验全部文件、nested 参数语义、模式/参数绑定、Adam 矩、完整状态、游标、源版本和运行时，再准备全部张量，最后修改参数和优化器。目标对象的参数梯度也必须为 None。

读取模块依赖 Torch；轻量资源监督父进程不应导入它来判断终态。不要同时常驻两份生产图来做恢复对照。运行前后的资源预算和 1 GiB 系统额外预留继续由监督器负责。

## 文件与失败行为

一个新目录包括：

```text
checkpoint.json           最后封存的根 manifest
training.npz              全部状态与命名 Adam 数组
policy/checkpoint.json    复用现有部署参数身份与语义验证
policy/parameters.npz     theta、读出 weight、bias
```

格式为 `cnsbridge.training_checkpoint.v1`，purpose 为 `detached_two_frame_tbptt_continuation`。固定文件名，NPZ 使用 allow_pickle=False，不保存巨型图矩阵或可执行对象。manifest 绑定所有文件的字节 SHA、各数组 dtype/shape、必要源码版本、Python/NumPy/SciPy/Torch 版本、CPU 架构和线程数。当前只支持相同运行时和线程设置下恢复，不自动迁移版本。

神经状态含全节点 v、g、int64 refractory_until、全部按到达顺序排列的延迟队列；默认动力学为 18 槽。感觉状态含 float64 phase、int64 event_counts 和驱动/空间身份、dt、时钟。Adam 保存稳定参数名下的实际 step、exp_avg、exp_avg_sq；有限且非负的零二阶矩合法，保留 float32 下溢。

未知或缺失字段、重复 NPZ/JSON 字段、摘要/源码/模式/时钟不一致、坏矩或状态都会被拒绝。禁止覆盖已有目录。写入失败可留下部分文件但没有根 manifest；这种目录不可恢复，保留供诊断，不自动删除。

第一版只接受至少一次实际 Adam 更新后的边界；初始化零步需从已登记初始化重建。当前无随机采样、噪声、dropout，也不消费训练全局 RNG；感觉相位必须恢复。未来加入随机算子须升级续训契约。

## 本批验证边界

51 项定向数值检查通过：joint/frozen 各自比较连续两次更新与首步保存、销毁对象、重建、恢复再更新，参数、Adam、全部状态、动作、特征和游标精确相同；包含非零延迟事件和感觉相位。还覆盖坏档晚期拒绝无部分恢复、更新边界校验、固定参数绑定、未封存目录与生产角色隔离。

这些结果是小电路持久化证据，不是完整图续跑、长期训练有效性或无人机行为证据。后续 runner 的协议与软件验证见 [P3c2](../specs/spec-008-trainable-snn/p3c-training-runner.md)及[使用说明](TRAINING_RUNNER.md)；真实完整图分段运行仍待独立证据。
