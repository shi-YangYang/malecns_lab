# Implementation Report — 全 GPU 修复训练集成

## Summary

按 [集成规格](gpu-recovery-integration.md)完成全 GPU 独立训练分支。继承旧 r3 第 160 窗的 theta、读出权重和偏置，显式重置 Adam；新 float32 感觉流、稳定 logit 损失及 100 步递归 credit 具有新身份。全部前向神经步骤保留；50 步重算仅负责内存，不改变 100 步梯度边界。生产入口锁定 horizon=100、raw theta L2 上限 1000，不提供静默回退或自动抬高门槛。

本实施未运行完整图、未启动真实训练或设备，没有改动旧 42 份绑定源码，也没有改动已独立验收的 GPUDrive / GPUAdam。实施不替代新的独立验收。

## Files Changed

新增业务模块，沿用 learning 目录：

- `gpu_policy.py`：完整 MPS 脑、GPUDrive、原 DN 读出、显式 GPU 状态；继承参数逐位校验。
- `gpu_recompute.py`：纯 MPS phase/counts 作为 checkpoint 输入输出；切断 v/g/全部 pending 的递归梯度，DN 累加保留。
- `gpu_window.py`：GPU logit loss、原动作 MSE 报告、梯度门槛、GPUAdam、实际更新后的固定特征读出与驻留证据。
- `gpu_state.py`：完整边界下载、所有参数/矩/时钟/延迟状态恢复及设备往返精确比对。
- `gpu_contract.py`：不导入 Torch 的来源闭包、旧 160 父链严格准入、显式新协议、GPU 实际身份期望。
- `gpu_archive.py`：新 float32 phase 存档、零矩 genesis、上下文、所有数组/摘要校验。
- `gpu_numbers.py`：独立 NumPy 保存证据审查；读出导数、稳定裁剪、fused Adam 参考、纠错方向及实际参数变化。
- `gpu_evidence.py`：genesis、可丢弃 gate、窗口/段链、恢复、资源、terminal 证据验证。
- `gpu_worker.py`：先 PID/锁/claim、后重型导入和模型；initialize/gate/train 三种受监督工作。
- `train_gpu.py`：轻量父进程、共享锁、资源监督、三阶段入口与进度。

新增 `tests/test_gpu_recovery.py`、`tests/test_gpu_recovery_orchestration.py`；新增本报告。其他 Agent 的诊断/规格文件及既有未跟踪 SciPy 审计文档未修改。

## Important Decisions

1. **先 genesis，再 gate，再 train。** initialize 单独生成并由父进程实际核验继承参数、零 m/v/step、新场景 600 步 blank 后状态，genesis SHA 是首窗根。gate 用旧第 160 窗真实两帧负 vx 标签，从新 genesis 状态做一次联合更新并丢弃；不计入 160 窗，也不把 gate 更新加入正式权重。
2. **生产数学已冻结。** logit 损失为 `mean(softplus(2*z)-(1+y)*z)`，保留 gain 正则 1e-4；最终动作仍为原 tanh×轴尺度。注册生产 horizon=100，raw theta 稳定 L2≤1000，原按组 clip=1。fixture 可显式比较其他 horizon；生产准入拒绝变更。
3. **全部训练主计算在 MPS。** mask/phase/counts/increment、完整脑、读出、loss、梯度、稳定裁剪、参数及 fused Adam 的 m/v/step 均在 MPS，fallback 必须是 0。CPU 只做数据载入、固定初始化、调度、边界存档和独立证据审查；审查值不回写优化器。参数仍需每次保持有限。
4. **正式预算独立。** 新本地 cursor/Adam 从 0 开始，最多 160 窗；第一段 1 窗、第二个进程 1 窗，随后每段最多 4 窗。失败现场保留，既有失败段阻断继续，不无条件略过失败窗口。
5. **旧链保持严格。** 完整读取/验证原 CPU/MPS 160 窗父链，限定父 run `b7807f8e…`、最终档 `fd301998…`、前缀 `1964b500…`。新来源闭包包含实际计算及原严格读者依赖，不 glob 尚在改动的 gradient_probe 文件；无需归档或变更旧源码。
6. **资源约束保持。** 同一 workspace_lock，申请 6 GiB、有效上限 available−1 GiB，600 s / 0.1 s，独立 MPS allocator 上限与取样；不操作 Noria/Docker。

## Validation

- Risk Level：**S3**，涉及训练数学、状态持久化及受监督进程契约；只验证新增子系统。
- 检查 1：`PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_gpu_recovery.py -q -s`；session **59148**，exit **0**，**3 passed in 15.56s**。
  - 轻量 parent/worker 导入不加载 Torch，非法预算与未登记 worker 拒绝。
  - 5 节点显式 fixture 的 checkpoint/direct 前向 logits、DN 特征和全部末态逐位一致；输入 phase/counts/state 未被重算修改。
  - 同一前向的 100 步 credit theta norm 为 `2.0122236081713357e-05`；全递归范围为 `0.00018758603646700878`，读出梯度保持，证明实现实际改变递归 credit 而非前向。
  - 固定负标签下 vx loss `14.999531449957464 → 14.999325774200987`，实际 raw theta norm `5.9044792614505766e-05`，内部参数真实变化；动作仍处正饱和，报告明确 `actual_action_sign_reversed=false`。
  - 保存恢复后再更新，所有末态、参数和 Adam 数组逐位一致；float64 phase、自洽重哈希伪造、翻转读出梯度与过低梯度门槛均被拒绝。
- 检查 2：`PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_gpu_recovery_orchestration.py -q -s`；session **94962**，exit **0**，**1 passed in 10.85s**。
  - 经协调者明确同意，以显式 5 节点 fixture 仅替换完整图/旧父链准入，执行新生产 initialize→gate→两个单窗独立子进程的监督与证据链；不是重复旧数学测试。
  - 实际 child PID/锁/claim、资源 CSV/MPS 上限、genesis 及 gate 核验、两进程恢复、首窗继承 untouched genesis、后窗连接前档全部通过。
  - 修改 gate terminal 完成计数后，严格读取拒绝。fixture 注入只存在测试生成脚本，不是生产可选项；生产只准入完整参考图。
- 两次检查间，窗口失败证据补充在 theta 门槛前保存 raw gradient，使后续超界失败可定位；第二个集成检查覆盖该最终窗口实现。未重跑已通过的基础模块或原脑核测试，没有全量 suite/build/lint。
- 仅出现原 PyTorch Sparse CSR beta 提示，无测试失败。新检查通过后停止验证。

## Commands / 实际执行前提

必须从源码 checkout、现有 `.venv-train`、相同版本运行，并设置 `PYTORCH_ENABLE_MPS_FALLBACK=0`。父链及其 CPU/r2 祖先、密封数据集、完整参考图、全部绑定源码与注册 SHA 必须仍在原路径且不变；不能仅复制最终参数文件绕过来源。正式命令由协调者在新独立软件验收通过后运行，下面是尚未执行的实际入口：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m cnsbridge.learning.train_gpu --initialize \
  --parent-run data/neural_training_mps_20261019_r3 \
  --out data/neural_training_gpu_recovery_20261019_r1 \
  --credit-horizon 100 --theta-norm-limit 1000

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m cnsbridge.learning.train_gpu --gate --run data/neural_training_gpu_recovery_20261019_r1

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m cnsbridge.learning.train_gpu --run data/neural_training_gpu_recovery_20261019_r1 --windows 1
```

首次一窗通过后同命令再执行一次（新进程恢复一窗）；相关实际证据通过后使用 `--windows 4` 分段完成剩余预算，最后按剩余数量减小。不能用 `--windows 4` 跳过前两窗门槛。失败后保留新目录，不覆盖或自动调大资源/梯度上限。

## Known Limitations

- 本批只有软件和显式 fixture 证据，**尚无新全 GPU 完整图 genesis/gate/训练成果**；旧诊断结果不能代替这项实际门槛。
- 数值训练改变了 float32 感觉、损失、credit horizon 和 Adam 状态；保留旧参数但不称同协议无缝第 161 窗。
- 固定特征损失下降只证明当前参数更新能纠错，不证明完整闭环跟随、泛化、连接组优势或实飞。
- GPUDrive 与原脑核仍有 Python/设备标量验证同步，实际端到端吞吐需完整图测量；不能由 GPU 驻留直接声称吞吐提升。
- source/registry/terminal 失败会阻断继续，失败资产不可原地重用；若需改变数学或实现，应另立新登记。

## Remaining Questions

无未确定实现决策。下一步由新的独立验收者审查本软件，再由协调者执行完整图三阶段门槛与有界训练。
