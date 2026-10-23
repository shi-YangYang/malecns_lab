# M3 固定 surrogate 斜率探针实施

2026-10-23。风险 **S3**：新增反向估计器身份，并在共享 MPS step 增加显式 keyword。代码和定向 CPU 检查完成，等待新的独立验收；本报告不表示完整脑探针已经运行或稳定化成功。准入以 [m3-stabilized-contract.md](m3-stabilized-contract.md) 为准。

## 实际变更

- `brain/mps_trainable.py` 增加 `step(..., spike_backward_gain=1.0)`。默认值走原先的 tensor 运算路径；仅显式非默认分支使用 `spikes.detach() + gain * (spikes - spikes.detach())`，保留二值前向并乘 surrogate 导数。没有修改 `brain/trainable.py`、surrogate 曲线宽度、完整图、输入编码、损失尺度或信用长度，没有 monkeypatch、数值清洗或 CPU 脑回退。
- `learning/x2_behavior_policy.py` 将固定 gain、算法说明、loss scale 1、反向身份及 MPS step 来源绑定到新 policy。`mps_identity.py` 原有源码哈希规则也使当前新 model_id 改变；没有把新模型冒充原身份或改写旧档案。新增可选的 50 步边界观察器，注册于 checkpoint 返回之后，不在重计算内部重复登记。
- `learning/x2_behavior_training.py` 和 CLI 新增 `probe-stabilized`，固定 gain 0.1、seed 2401、600 步预热、连续两个 500 步窗口、50 步块、1000 步信用、第二窗损失和一次新 Adam、600 秒预算。原 `probe` 固定 gain 1；两种模式不能通过参数交换导数倍率。参数头仍由原标签 mask 成零，不为制造梯度更换训练样本。
- 首轮来源固定为 `data/x2_behavior_probe_20261023_r1`，绑定其已知 policy/checkpoint ID、注册信息、旧源码、输入与状态文件哈希。新模型严格核对原图、配置、类型及神经映射、初始化、环境等契约，只接受本次明确的核心/策略源码和 backward 身份差异；通过显式迁移复制首轮全部初始化参数，不沿用旧 Adam。
- 实际新 run 内，600 步预热后的状态及两窗输出/状态各自落盘后，与首轮逐数组比较 dtype、shape 和原始字节哈希。共比较 3 个状态、2 份编码输入/DN 特征/原始三头输出；同时比较观测与神经时钟，记录不同的新旧 policy ID。任一不一致即拒绝进入更新，比较报告在拒绝前保存。
- 稳定化探针在 650–1600 每 50 步边界记录 v/g/pending 的梯度，hook 原样返回输入梯度。原始数组逐个写 `.npy`，不累积全部逐步状态；未被损失遍历的末端 g/pending 梯度明确标记缺失，不伪造零梯度。第一窗末 v 仍要求真实有限且非零，所有已观察边界梯度不得非有限。
- Adam 前确认 fresh optimizer，且每项原始梯度有限、最大绝对值不超过 `sqrt(float32.max)`（约 1.84467435e19），防止浮点平方溢出；这只是表示范围拒绝，不裁剪或缩放梯度。更新后检查 Adam 状态有限，theta 与行为/阶段头实际改变，参数头不变。失败保存原始梯度、真实前后参数和实际是否调用 step。

后续 `readout/joint/evaluate --checkpoint` 会默认继承该 checkpoint 里的显式 backward gain，显式传入不同 gain 则拒绝。此实现只为本次固定探针提供准入，不授权正式训练，也不把探针 checkpoint 标作部署模型。其他调用方（例如 M4 session）加载新 checkpoint 时同样需要传入其 backward gain；本次没有修改这些文件。

## 已执行的检查

仅执行了下面一次与修改直接相关的 CPU 子系统检查：

```sh
PYTHONPATH="$PWD/src" .venv-train/bin/python -m pytest -q tests/test_x2_behavior_training.py
```

结果：**30 passed in 1.28s**，退出码 0。新覆盖包括二值前向精确相等、默认分支返回原 tensor、surrogate 梯度为原来的 0.1、固定模式/反向身份、无重栈的 preview、配对数组一位差异/相同身份拒绝与失败记录、不同身份间参数迁移、边界 hook 不修改梯度、有限但不能安全平方的梯度拒绝。其余已有 checkpoint/失败证据/标签 mask 检查继续通过。

检查只使用 CPU 标量、读出与 archive fixture，没有构造脑图或运行 MPS。没有全仓测试、build、仿真、训练、完整脑探针、commit 或 push。首轮失败 run 保持原样；实际完整图前向是否配对一致、长信用梯度是否可用、耗时与资源均仍未知。

## 独立验收后的协调者入口

```sh
PYTHONPATH="$PWD/src" .venv-train/bin/python -u -m malecns_lab.experiments.x2_behavior_training --execute --mode probe-stabilized --output data/x2_behavior_probe_20261023_r2
```

只有协调者在新的独立验收通过后启动。沿用父进程完整脑锁、仿真关闭预检、1 GiB 系统预留、RSS/MPS 有效预算及总墙钟监督。新增 `paired-forward.json`、`boundary-gradients/` 与既有 `gradient-audits/` 共同保留本轮证据。不得据失败自动扫描倍率、混入 loss 缩放、缩图、截断信用或增加预算。
