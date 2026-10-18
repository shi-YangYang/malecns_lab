# 完整图两帧动作联合更新

该工具实现 [P3b 冻结协议](../specs/spec-008-trainable-snn/p3b-joint-update-pilot.md)：一次离线动作模仿损失同时更新完整 MaleCNS 网络的内部 `theta` 与三轴读出。它不是完整训练、神经闭环、未见场景评估或无人机控制入口。

## 运行

必须先完成软件独立验收和提交，保留 Noria/Docker，且没有其他完整图任务并行。每次使用新的输出目录，不覆盖或因阴性自动重试：

```bash
cd /Users/yang/project/FlyDrones
PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.joint_probe \
  --graph data/malecns_v2/full_reference.npz \
  --dataset data/neural_teacher_20261019_r1 \
  --out data/joint_update_20261019_r1
```

入口没有裁图、缩窗、改选帧或取消监督的参数。父进程绑定完整参考文件的字节摘要和 provenance、当前源码、固定协议及已封存的数据集；只解码参考文件的小型 metadata 成员，不加载图矩阵。全部 Torch 模型、反向和生产 checkpoint 重载位于受监督 worker。申请 RSS 6 GiB，上限取 `min(6 GiB, available−1 GiB)`，系统额外预留 1 GiB，600 秒墙钟、0.1 秒采样。采样监督不是操作系统的硬内存隔离；达到门槛保留失败，不自动重试。

## 固定计算

使用 `train-sine-combined` 的索引 0、1，对应 0 和 0.1 秒；两帧必须有效。`teacher_target` 是同帧经限幅的真值教师标签，独立于采集时视觉策略的 `executed_action`。保存原始 float64 标签；损失按模型 dtype 转成 float32，报告指标再从保存的 float32 动作和实际参与计算的目标/尺度以 float64 归约。

每个分支先进行 60 ms 空白、无梯度预处理，再连续处理两个 50 ms 图像窗。中间没有 reset 或 detach，动作损失可以穿过两帧共 100 ms 的神经计算；梯度不穿过预处理。全部 1,314 个 DN 的两个半窗平均电位进入三轴线性读出，视觉或真值没有直接连到动作。

Adam 仅执行一次：内部 theta 学习率 0.001，读出 weight/bias 学习率 0.0001。动作损失是两帧三轴归一化 MSE，加 `1e-4*mean((gain−1)^2)` 正则；初始正则和其 theta 梯度必须为零。统一全参数梯度范数裁剪到 1，拒绝非有限值；固定背景 rho 不训练。

五个分支均从独立 fresh initial state 开始：

| 分支 | 内部参数 | 读出 | 梯度 |
|---|---|---|---|
| baseline | 初始化 | 初始化 | 两帧连通 |
| updated | 更新后 | 更新后 | 关闭 |
| acute_restore | 恢复初始化 | 更新后 | 关闭 |
| full_restore | 恢复初始化 | 恢复初始化 | 关闭 |
| checkpoint_reload | 从存档重载 | 从存档重载 | 关闭 |

updated 后立即保存独立完整策略参数存档。做完干预后，旧模型、优化器和全部计算图引用均释放，才调用生产入口加载 checkpoint，避免两份完整图常驻。该存档仍只有部署参数，不含优化器、隐藏状态或训练游标，不能称为续训档。急性恢复包含内部参数影响预处理状态的作用，不单独定位刺激期间的某条通路。

## 结果与证据

`result.json` 分开报告：

- `protocol_complete`：证据齐全、完整图未变、数值有限、恢复/重载和监督终态有效。
- `joint_numeric_update`：内部任务梯度非零，theta/gain、读出 weight/bias 确有变化；附逐项布尔值。
- `internal_action_effect`：updated 与 acute_restore 的全部两帧三轴动作存在超过 rtol=1e-6/atol=1e-7 的差异；同时报告全部 DN 特征比较和模仿损失改善/恶化方向。

协议完成时退出 0，即使内部更新或动作影响是阴性，或 loss 上升。注册、资源、进程或证据失败退出 1；后两项为 `null` 表示尚不能判定。无论哪种结果，都不关闭总 AC3–AC6。

输出目录保留：

- `registration.json`：协议、数据、图文件、源码、环境与预算的事前绑定。
- `selection.npz`、`dn_ids.npz`、`policy_identity.json`：输入、同帧目标、有效性/时间/索引、全部输出细胞顺序和完整策略身份。
- `initial_parameters.npz`、`gradients_before_clip.npz`、`update.npz`：全部前后参数/gain/rho、裁剪前后梯度及正则梯度；遇到非有限裁剪失败，已写出的原始梯度仍保留。
- 五组分支 `.npz`/`.json`：两帧动作、全部 DN 特征、各分支参数、损失，以及预处理和末状态每一个数组的 shape/dtype/SHA256 和神经/驱动时钟。这些是完整状态摘要，不是全节点每步轨迹。
- `checkpoint/`：训练后的完整策略参数，逐位核对 after 数组。
- `worker_result.json`、`phase.json`、`worker.log`、`resources.csv`：执行环境、阶段耗时/已完成帧和采样资源。worker 自身计时从其模块导入后开始；监督器墙钟包含 Python/Torch 启动、导入和退出。
- `supervisor_result.json`：父进程实际收取的原始监督终态，独立保留；`result.json` 包含父进程 NumPy 独立复核的结论和终态文件摘要。

父复核不实例化模型：重新校验封存数据、选定输入/标签、源身份、图/策略/存档绑定和数组，独立重算损失、梯度范数/裁剪、首次 Adam 更新、动作读出及恢复/干预比较，并核对资源 CSV 与进程终态。它检验已保存证据的一致性，不是重新执行神经计算，也不是对人为全套伪造文件的密码学证明。

若 worker 被预算终止，`worker_result.json` 可能不存在或未封存，已生成的 phase、日志和部分 NPZ 仍留在原目录。下一步依据具体失败分析重新登记协议，不能把不完整证据提升为协议完成。
