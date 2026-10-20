# 完整输入输出集合与空间事件接口

本接口为完整 MaleCNS 可训练网络准备输入集合、独立输出集合和可重复刺激。没有动作读出、背景驱动或设备连接，也没有完成完整脑的响应／学习验证。数值核心见 [TRAINABLE_SNN](TRAINABLE_SNN.md)，本批登记见 [P2b](../specs/spec-008-trainable-snn/p2b-mapping.md)。

## 图与成员身份

`brain.policy_groups.load_policy_neuron_map(graph, reference_path)` 接收已经通过 `load_training_graph` 验证的完整 `TrainingGraph`。它从同一 NPZ 懒读 `meta`、`body_ids`、`annotation_fields` 和全部注释／缺失数组，沿用 `annotation_hash` 的排序、dtype、shape 和字节摘要契约。来源、注释摘要和 body ID 顺序必须与已验证的图一致；这里不重新读取 `data/indices/indptr`，所以这一步是元数据绑定检查，不能替代前面的完整图校验。

- 输入：原始 `type` 精确等于 `R1-R6` 的全部细胞。
- 输出：原始 `superclass` 精确等于 `descending_neuron` 的全部细胞；包含缺失类型和所有侧别，不使用 `DN` 名称前缀筛选。
- 保留两组的实际 body ID、图内索引、原始 `rootSide/somaSide` 及缺失标记。空集合、类型分组不一致和身份错误明确拒绝。

`PolicyNeuronMap.identity` 包含来源、训练图身份、注释摘要、成员和规则；`map_id` 是这些内容的摘要。输出集合当前仅供观察／后续读出，不自动产生动作，不替换参考资产原有的 33 细胞群组，也不剪除任何图节点。登记时元数据统计为 3,377 个输入、1,314 个输出；构造时按资产推导，不硬编码计数。本批没有重新运行该完整资产。

数值夹具使用独立的 `numerical_test_policy_map`，身份为 `numerical_test_only`；生产入口拒绝这类图。夹具只检查接口，不能代替完整脑实验。

## 工程空间映射

`senses.spatial_drive.build_spatial_input_map(policy)` 将全部输入细胞映射到 [target_mask](../src/malecns_lab/senses/target.py) 产生的 12×16 占用图。行从上到下、列从左到右，格号为 `row * 16 + column`。

映射固定为以下算法：

1. 缺失 `rootSide` 独立成组；其他原始字符串按 Unicode 排序，空字符串与缺失分开。
2. 每组成员按 body ID 升序排列；所有组依次使用同一个 NumPy `Generator(PCG64(1701))` 随机流置换成员。
3. 置换后的第 `j` 个成员分配到 `j % 192` 号格。每组所有格的成员数最多相差 1。

每一侧组都接收同一整幅相机图。这是工程编码，不是解剖视网膜位置、眼柱对应或双眼标定。少于 192 个细胞的组会留下未覆盖格；`identity.coverage` 明确保存每格计数、未覆盖格列表和数量。映射身份还保存算法、种子、NumPy 版本和每个输入 body ID 的格号，调用者不能改写返回的数组。

## 事件与连续状态

`SpatialDrive` 默认协议为 `dt_ms=0.1`、最大 150 事件/秒、每事件外加 68.75 mV。占用值 `[0,1]` 线性转换为事件率；参数是工程刺激协议，不是生理标定。显式配置改变会产生不同的驱动身份，研究运行应在登记中记录配置。单步最大事件数超过 1 的配置直接拒绝，不能静默丢事件。

每个输入细胞的初始相位由独立 `PCG64(1702)` 按输入 body ID 升序生成 `[0,1)` 均匀数。每个神经步：

```text
phase += occupancy[pixel_cell] * max_rate_hz * dt_ms / 1000
若 phase >= 1：产生一次事件，phase -= 1
```

相位跨图像帧连续保留。`step(mask)` 先检查形状、有限值和 `[0,1]` 范围；非法输入抛错且不改变相位、步数或累计事件数。合法空白帧没有新事件，相位保留，神经步时钟仍向前。目标检测有效性应由任务外层处理，不能从空白占用图自动推断“目标丢失后的任务已成功”。

`reset()` 重置到同一初始相位和零计数；`export_state()` 返回不可变状态；`restore_state(state)` 在完整验证后恢复。状态绑定空间映射、驱动配置、dt、神经步数和每输入细胞事件计数，错误恢复不会部分修改当前状态。状态的 `phase`、`event_counts` 可作为普通 NumPy 数组存储；恢复时仍需保留身份字段，不能只导入相位数值。

单步返回 `DriveEvents`，只包含本步发放输入的图内索引和电压增量。按需调用 `to_dense()` 或 `to_torch(dtype=...)` 才分配完整节点向量；所有非输入细胞严格为零。PyTorch 返回 CPU float32/float64 张量，不调用模型或硬件。

```python
from malecns_lab.brain.trainable_graph import load_training_graph
from malecns_lab.brain.policy_groups import load_policy_neuron_map
from malecns_lab.senses.spatial_drive import SpatialDrive, build_spatial_input_map

# 这一加载会验证完整图；只在资源和实验登记条件满足后执行。
graph = load_training_graph("data/malecns_v2/full_reference.npz")
policy_map = load_policy_neuron_map(graph, "data/malecns_v2/full_reference.npz")
drive = SpatialDrive(build_spatial_input_map(policy_map))
# mask = target_mask(frame, camera)；每次 step 对应一个神经 dt。
# events = drive.step(mask)
# voltage = events.to_torch(dtype=model.theta.dtype)
# spikes, state = model.step(voltage, state)
```

外部图像的帧时长与神经步数必须由后续实验协议明确：一个新视频帧不等于一个 0.1 ms 神经步。这里没有自动选择控制频率、跳过神经时间或向输出细胞添加偏置。

## 当前科学边界

当前 R1–R6 递质符号在核心中为抑制性，其他细胞从静息初始化且无背景驱动。仅把视觉刺激接入 R1–R6，并不能保证下降神经元出现脉冲；代理梯度也不能证明前向信号到达输出。见 [可观测性记录](../specs/spec-008-trainable-snn/signal-observability.md)。本接口保留该静默问题，不用输出偏置、目标误差直连动作或额外背景电流掩盖它。
