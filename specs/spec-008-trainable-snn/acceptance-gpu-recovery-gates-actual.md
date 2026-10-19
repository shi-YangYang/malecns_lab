# Acceptance — 全 GPU 修复分支完整图实际门槛

## Result

**PASS**。本验收者未参与实现，独立核对 [集成规格](gpu-recovery-integration.md) 要求的完整图 initialize、可丢弃负目标 gate、正式首窗及新进程第二窗。协调者已分别收取 session **7199 / 14692 / 40286 / 96128** 的 exit 0；保存的四份父终态、worker 终态及监督退出结果一致。

本结论允许按已冻结协议每段最多四窗继续剩余 **158 窗**，不是新 160 窗训练完成或学会目标跟随。没有验证闭环、泛化、实飞或连接组性能优势。

实际目录：`data/neural_training_gpu_recovery_20261019_r1`。独立只读审查脚本与结果：`data/gpu_recovery_gate_review_20261019/review.py`、`review.json`。

## Spec Coverage

- **完整图、协议和来源：通过。** 独立重算 run、protocol、policy、drive、optimizer、checkpoint、job 和 seal 的 canonical ID；run 为 `601aad8e5ef0fd52034952d4790dc2c812d2063508c746ec45fd7e1dbec41b81`，protocol 为 `eff4bc9e9033ca27e982e4b58d3c9229c5299be4fd2b89f9a263600c5980825c`。54 份当前绑定源码 SHA 匹配，其中旧 r3 的 42 份保持不变。实际读取完整参考 NPZ 文件复核 SHA `98f83158ed12f0c836fb085c2e286e54f9458630c2a158bee3164af52648f6f3`；四个 worker 的实际 policy 与登记相等，raw counts 前后均为 `4ce30eec9ecc64c7bb2723f9dee084b40cea08793375f40ab71cacc2592b0565`。完整图身份仍为 166,700 节点、25,582,938 条边、124,177,617 突触计数；输入 3,377 个、DN 输出 1,314 个。没有重建或删图。
- **参数继承与新状态：通过。** 旧第 160 窗档与前缀匹配，genesis 中 theta、读出权重和偏置与父数组逐位相同；全部九个 Adam 数组为零。新 phase 为登记 PCG64 初始化转 float32 后的精确值，事件计数为零，脑/感觉时钟均为 600，frame=0。genesis ID 为 `ca61d78953740b0fb3763c611494d346b2be6cf162239c46679e88e7e4bdc901`，参数没有从零重建。
- **可丢弃纠错 gate：通过。** 图像、教师、时间和索引与密封训练集 yaw 场景第 78–79 帧逐位匹配，教师 vx 均为负。实际完整脑 forward/backward 后 theta 原始 L2 为 **0.07613792855**，有限且低于 1000，**5,014** 项内部参数变化；vx 原始 bias 梯度 **+0.3282814622**，2,628 个 vx 权重梯度均非零。固定实际 features 的更新后 vx logits 从 **[1.43134415, 5.31309986]** 降至 **[1.28804743, 4.70904207]**，vx logit loss **3.533785049 → 3.154177824**，总 logit loss **1.723703183 → 1.515350529**。两帧 logit 仍为正，`actual_action_sign_reversed=false` 正确。
- **gate 的证据边界：准确。** 本 gate 从新 genesis 神经状态出发，其 features 与旧第 160 窗末期状态不同，不能把这些 logits 说成旧约 9.6 的完全饱和值。旧饱和特征的解析导数证据属于 [r2 诊断](acceptance-gradient-probe-r2-actual.md)；本 gate 新增的是完整图真实联合更新对错误正 vx 的一步纠错证据。它不证明动作已经后退。
- **正式训练不继承 gate 更新：通过。** 首窗恢复的全部 **18 个数组**与未修改 genesis 逐位相同，包括三份参数、九份 Adam 与六份完整神经/感觉状态。第二个独立 worker 恢复的全部 18 数组与首窗档逐位相同；包含 18 槽 pending、refractory、phase、事件计数及所有时钟。两个 worker PID 为 **21731 / 22048**，与初始化、gate PID 也各不相同。首两窗选择 combined 场景第 0–3 帧；脑/感觉时钟 **600→1600→2600**，frame **0→2→4**，新 Adam step **0→1→2**。
- **实际更新、数学与驻留：通过。** 保存 arrays 独立重算 logits、tanh 动作、logit loss、原动作 MSE、读出解析梯度、稳定分组裁剪、Adam m/v/step 及参数更新；均满足预登记容差，非零梯度裁剪前后支持与符号保持。gate 与两个正式窗中，参数、raw/clipped gradients、九份 Adam 状态、phase/counts、logits/loss 的实际设备记录全部为 `mps:0`。结合已验收并 SHA 绑定的计算路径，确认新训练数值使用 MPS；CPU 证据验算未回写优化器。
- **存档和日志链：通过。** 检查全部 terminal/seal 绑定文件 SHA、checkpoint NPZ SHA、数组类型/形状/有限性、完整状态 digest、before/after 参数与矩、每窗事件及逐字节 prefix。gate 不计入正式优化步骤或 prefix。末档为 `034b17b458fff6059d78d970fd4f4083339885a6cf551a6991011cbdc5681278`，末前缀为 `26606588ba3e0e2efe659c335d91786569bd45221b75f2719420e1eb9d75d1f8`；正式进度为 **2/160**。

## 实测

| 阶段 | theta 原始稳定 L2 | theta 实际变化项 | 窗口 compute | 监督墙钟 |
|---|---:|---:|---:|---:|
| initialize | 不执行更新 | 继承父参数 | 不适用 | 18.365 s |
| gate，更新后丢弃 | 0.076137929 | 5,014 | 30.496 s | 41.081 s |
| 正式窗 1 | 0.004985146 | 5,016 | 29.828 s | 40.631 s |
| 正式窗 2 | 0.009769591 | 5,115 | 28.884 s | 39.501 s |

这三个更新的 float64 独立 Adam 参考与实际参数最大绝对差分别不超过 **1.95e-9、1.94e-9、1.11e-8**，均通过原 `rtol=2e-5, atol=1e-8` 组合容差。不是逐位 CPU 等价声明。

四个阶段均登记申请 6 GiB、额外预留 1 GiB；有效 RSS 上限逐次为 **2,315,583,488 / 2,482,176,000 / 2,378,465,280 / 2,506,506,240** 字节，符合 `min(6 GiB, available−1 GiB)`。独立 CSV 统计中最高 RSS 为 **1,649,508,352** 字节，最低系统 available 为 **2,419,212,288** 字节；均满足各自门槛。最高 MPS driver 为 **1,579,122,688** 字节，均低于相应 allocator cap。各阶段均小于 600 秒，0.1 秒采样记录及终态完整。RSS 和 MPS driver 属不同口径，不相加。

## Validation

- Risk Level：**S3**，限新完整图运行的数值、持久化、来源和资源证据。
- 阅读 AGENTS、使命、技术栈、项目规则、验收流程、当前交接、集成规格、软件 PASS、实施报告、根因报告及 r2 实际验收。
- 唯一新增执行检查为上述 stdlib/NumPy 保存证据审查，exit **0**，结果 **PASS**，结束时 `torch_loaded=false`。没有导入 Torch、加载模型、执行优化器、重跑软件测试或训练、build、lint。
- 只新增本报告和独立审查资产，不修改业务或原运行数据、不提交；没有操作 Noria、Docker 或硬件。

## Issues

本实际门槛范围无阻断项。正式首两窗都是正前进教师，vx logit 上升符合该局部标签，不能据此宣称整体饱和问题已经消失；后续负目标阶段的纠错能力仍需按预算记录。

## Regression Risks

100 步 credit 只在本次三个更新确认有限，后续窗口仍可能触发 1000 上限；不得自动放宽。新 float32 感觉、稳定 logit loss、短递归 credit 及 Adam 清零共同构成新协议，不能称旧轨迹无缝续接。一步固定特征下降不等于闭环改善；训练集 loss 也不替代未见场景评估。

## Required Rework

无。实际门槛证据充分，停止追加检查；剩余训练由协调者按已冻结预算串行推进。
