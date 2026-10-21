# 完整图两帧动作联合更新

> 命名迁移：本文操作入口已更新为 `malecns_lab`。涉及旧 run／checkpoint 的目录保留为参数示例和历史证据；不能直接用新源码续跑或重新登记旧实验。须使用匹配的源码身份与新输出目录，精确恢复旧实验需原 commit。见[迁移说明](ARCHITECTURE.md#环境迁移与历史来源)。

当前工具实现 [P3b r2 稳定分组裁剪协议](../specs/spec-008-trainable-snn/p3b-stable-group-clip.md)：一次离线动作模仿损失同时更新完整 MaleCNS 网络的内部 `theta` 与三轴读出。它不是完整训练、神经闭环、未见场景评估或无人机控制入口。

## 运行

必须先完成软件独立验收和提交，保留 Noria/Docker，且没有其他完整图任务并行。每次使用新的输出目录，不覆盖或因阴性自动重试：

```bash
cd /Users/yang/project/malecns_lab
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.joint_probe \
  --graph data/malecns_v2/full_reference.npz \
  --dataset data/neural_teacher_20261019_r1 \
  --out data/joint_update_20261019_r2
```

入口没有裁图、缩窗、改选帧或取消监督的参数。当前 registration / protocol / worker / final result schema 均为 v2；缺少新字段的 v1 记录不能作为本次通过证据。原 r1 统一裁剪软件在提交 `0373c87`，原失败目录保持不变。

必须保留封存数据同级的 `joint_update_20261019_r1/`。新登记固定绑定 r1 的 registration/protocol ID、旧 registration/result 字节摘要和参与比较文件 SHA，并检查旧终态原有的六个文件摘要。r1 因失败未封存数值文件清单；这些比较 NPZ 的摘要是 r2 实施时、r1 独立验收后新增的冻结锚点，不能称为由旧终态清单证明。核心、策略、数据、checkpoint、监督器源码必须与 r1 相同；允许改变的只有本次 joint_probe 优化和证据模块。

父进程绑定完整参考文件的字节摘要和 provenance、当前源码、固定协议及已封存的数据集；只解码参考文件的小型 metadata 成员，不加载图矩阵。全部 Torch 模型、反向和生产 checkpoint 重载位于受监督 worker。申请 RSS 6 GiB，上限取 `min(6 GiB, available−1 GiB)`，系统额外预留 1 GiB，600 秒墙钟、0.1 秒采样。采样监督不是操作系统的硬内存隔离；达到门槛保留失败，不自动重试。

## 固定计算

使用 `train-sine-combined` 的索引 0、1，对应 0 和 0.1 秒；两帧必须有效。`teacher_target` 是同帧经限幅的真值教师标签，独立于采集时视觉策略的 `executed_action`。保存原始 float64 标签；损失按模型 dtype 转成 float32，报告指标再从保存的 float32 动作和实际参与计算的目标/尺度以 float64 归约。

每个分支先进行 60 ms 空白、无梯度预处理，再连续处理两个 50 ms 图像窗。中间没有 reset 或 detach，动作损失可以穿过两帧共 100 ms 的神经计算；梯度不穿过预处理。全部 1,314 个 DN 的两个半窗平均电位进入三轴线性读出，视觉或真值没有直接连到动作。

Adam 仅执行一次：内部 theta 学习率 0.001，读出 weight/bias 学习率 0.0001。动作损失是两帧三轴归一化 MSE，加 `1e-4*mean((gain−1)^2)` 正则；初始正则和其 theta 梯度必须为零。theta 与 weight+bias 两组分别裁剪到 L2 上限 1，合并上限为 sqrt(2)，固定背景 rho 不训练。先检查全部原始梯度有限，再以 float64 归约各组范数，计算 `min(1, 1/(norm+1e-6))`，将系数转成 float32 后乘入原 float32 梯度；两个系数均保存。不能继续称为旧的统一全参数裁剪。

五个分支均从独立 fresh initial state 开始：

| 分支 | 内部参数 | 读出 | 梯度 |
|---|---|---|---|
| baseline | 初始化 | 初始化 | 两帧连通 |
| updated | 更新后 | 更新后 | 关闭 |
| acute_restore | 恢复初始化 | 更新后 | 关闭 |
| full_restore | 恢复初始化 | 恢复初始化 | 关闭 |
| checkpoint_reload | 从存档重载 | 从存档重载 | 关闭 |

反向完成且尚未裁剪/更新时，将当前 baseline 的全部动作/DN 特征、初始参数及原始梯度与 r1 比较；初始参数精确相等，动作/特征/梯度按 rtol=1e-6、atol=1e-7 核对，逐项保留 exact/allclose/maxdiff。输入、DN 顺序、策略身份也须一致。超容差则先保存比较报告并停止，不进入 Adam，不事后放宽。

updated 后立即保存独立完整策略参数存档。做完干预后，旧模型、优化器和全部计算图引用均释放，才调用生产入口加载 checkpoint，避免两份完整图常驻。该存档仍只有部署参数，不含优化器、隐藏状态或训练游标，不能称为续训档。急性恢复包含内部参数影响预处理状态的作用，不单独定位刺激期间的某条通路。

## 结果与证据

`result.json` 分开报告：

- `protocol_complete`：证据齐全、完整图未变、数值有限、恢复/重载和监督终态有效。
- `joint_numeric_update`：内部任务梯度非零，theta/gain、读出 weight/bias 确有变化；附逐项布尔值。
- `internal_action_effect`：updated 与 acute_restore 的全部两帧三轴动作存在超过 rtol=1e-6/atol=1e-7 的差异；同时报告全部 DN 特征比较和模仿损失改善/恶化方向。

协议完成时退出 0，即使内部更新或动作影响是阴性，或 loss 上升。注册、资源、进程或证据失败退出 1；后两项为 `null` 表示尚不能判定。无论哪种结果，都不关闭总 AC3–AC6。

输出目录保留：

- `registration.json`：协议、数据、图文件、源码、环境与预算及 r1 历史锚点的事前绑定。
- `selection.npz`、`dn_ids.npz`、`policy_identity.json`：输入、同帧目标、有效性/时间/索引、全部输出细胞顺序和完整策略身份。
- `initial_parameters.npz`、`gradients_before_clip.npz`、`update.npz`：全部前后参数/gain/rho、裁剪前后梯度及正则梯度；遇到非有限裁剪失败，已写出的原始梯度仍保留。
- `r1_comparison.json`：优化前相同输入的完整比较，即使数值超容差也保留。
- `adam_state.npz`：按 `brain.theta`、`readout.weight`、`readout.bias` 稳定参数名保存实际 `step`、`exp_avg`、`exp_avg_sq`。step 为 float32 标量且等于 1，两个矩为与参数同形的 float32；这是此次优化证据，不是可恢复训练存档。
- 五组分支 `.npz`/`.json`：两帧动作、全部 DN 特征、各分支参数、损失，以及预处理和末状态每一个数组的 shape/dtype/SHA256 和神经/驱动时钟。这些是完整状态摘要，不是全节点每步轨迹。
- `checkpoint/`：训练后的完整策略参数，逐位核对 after 数组。
- `worker_result.json`、`phase.json`、`worker.log`、`resources.csv`：执行环境、阶段耗时/已完成帧和采样资源。worker 自身计时从其模块导入后开始；监督器墙钟包含 Python/Torch 启动、导入和退出。
- `supervisor_result.json`：父进程实际收取的原始监督终态，独立保留；`result.json` 包含父进程 NumPy 独立复核的结论和终态文件摘要。

父复核不实例化模型：重新校验封存数据、选定输入/标签、源身份、图/策略/存档绑定和数组，独立重算损失、float64 组范数/float32 实际裁剪、初始零矩的一次 Adam、动作读出及恢复/干预比较，并核对资源 CSV 与进程终态。Adam 复算按 float32 的矩更新、sqrt/div/add 和参数更新步骤执行，保留二阶矩下溢为零的语义；不能仅套理想的 `lr*g/(abs(g)+eps)` 公式。矩固定按 rtol=1e-6、atol=float32 最小正次正规数核对，参数更新沿用 rtol=5e-6、atol=1e-9。父复核还重做 r1 对比并拒绝伪报通过。它检验已保存证据的一致性，不是重新执行神经计算，也不是对人为全套伪造文件的密码学证明。

若 worker 被预算终止，`worker_result.json` 可能不存在或未封存，已生成的 phase、日志和部分 NPZ 仍留在原目录。下一步依据具体失败分析重新登记协议，不能把不完整证据提升为协议完成。

分组裁剪重新分配内部和读出的更新机会，不能解释 r1 中巨大内部梯度的来源或证明长期训练稳定。即使 r2 完成，也需保留组范数、裁剪后非零量、实际参数更新量及 Adam 二阶矩下溢数量，并分别判断内部数值更新和动作作用。
