# 完整神经策略参数存档

`learning.policy_checkpoint` 的模式为 `cnsbridge.neural_policy_checkpoint.v1`。它保存完整策略的三个可训练参数：`brain.theta`、`readout.weight`、`readout.bias`。旧 `learning.checkpoint` 是仅保存内部增益的独立模式，两者互相拒绝，不能替换文件名来混用。

```python
from cnsbridge.learning.policy_checkpoint import (
    save_policy_checkpoint, read_policy_checkpoint,
    restore_policy_checkpoint, load_policy_checkpoint,
)

# policy 已通过 NeuralPolicy.from_reference 创建；只写新的目录。
save_policy_checkpoint(policy, "data/policy_r1", run_identity={"protocol_id": "已登记协议的 ID"})
metadata, arrays = read_policy_checkpoint("data/policy_r1")  # 不加载完整图
restore_policy_checkpoint(policy, "data/policy_r1")          # 恢复到已有兼容策略

# 生产加载必须再次验证完整参考图，禁止 numerical_test_only。
loaded = load_policy_checkpoint("data/policy_r1", "data/malecns_v2/full_reference.npz")
state = loaded.initial_state()  # 新场景的显式 reset；不是恢复训练中途状态
```

以上示例中的 `policy` 和协议 ID 必须来自实际调用方；存档工具不会自动训练、初始化神经状态或运行控制器。

## 文件与身份

目录中有非 pickle 的 `parameters.npz` 与最后写入的 `checkpoint.json`。NPZ 严格包含 `theta`、`readout_weight`、`readout_bias`，JSON 记录各字段对应参数、shape、dtype、文件 SHA256，以及全部 `policy.identity`。

策略身份绑定完整图与角色、神经元选择及顺序、空间编码、事件驱动、固定背景、相机、神经/控制时钟、全部 DN 特征的顺序与归一化、动作尺度、精度和源文件版本。固定 `rho=0` 对应的 8 mV 背景由身份约束，不能额外保存成可训练参数。`run_identity` 可选，必须是可 JSON 化的有限数据，参与存档 ID。

轻量读取先校验模式、嵌套 canonical ID、身份字段的关联和当前源码，再重建轻量输入映射/驱动身份、检查文件 hash 与参数语义，不构造完整脑。模型字段集合、有界增益、梯度和事件顺序必须符合当前版本契约；固定 tonic 的公式、范围、积分、成员类型/数量/摘要格式也分别核对，不能靠重算 ID 接受不同动力学。成员实际身份仍由生产参考加载再次核实。恢复会先验证所有身份及全部目标参数，再执行拷贝；错误相机、时间窗、背景、角色、顺序、精度、来源或畸形数组均不应导致部分参数被恢复。

生产加载除上述步骤外，再调用原有 `NeuralPolicy.from_reference` 核实全图，构造结果的完整身份必须相等。数值电路只允许通过明确 fixture 的已有策略做软件往返检查，不能进入生产加载。

## 本格式的用途边界

清单明确写入 `purpose=deployment_parameters_only`。不保存 Adam 动量、学习率调度、TBPTT 神经状态、延迟队列、输入相位、随机游标或样本进度，因此**不能据此宣称训练可从中途精确续跑**。从这些参数开始一个新训练过程与恢复中断训练是两件事；完整训练恢复由后续独立协议定义。

保存会先验证 CPU、精度、严格参数集合和有限性，再创建目录。已有目录一律报错。读取不执行 pickle；旧资产保持不变。源码版本变化可能使旧存档不兼容，需要显式迁移/重验，不能通过改 hash 伪装成相同策略。
