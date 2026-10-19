# GPU训练与状态恢复

## 当前全GPU修复入口

新入口为 `cnsbridge.learning.train_gpu`。感觉掩码与事件、完整脑、读出、损失、反向、稳定裁剪及Adam参数/m/v/step均使用MPS；CPU负责资产准备、调度、I/O和独立证据校验。完整图/原始边保持，训练内部群组增益与动作读出。当前任务是合成标记目标数据的监督模仿学习，预期学习前进/后退、升降与转向，不等于已经完成闭环控制。

[根因报告](../specs/spec-008-trainable-snn/gpu-recovery-causes.md)与[集成规格](../specs/spec-008-trainable-snn/gpu-recovery-integration.md)解释本次数值变化：稳定logit损失、100步递归反向范围、全部1000步前向、原始theta L2上限1000。新协议继承旧160窗的模型参数，Adam清零，新场景使用float32 GPU感觉；旧160窗原样保留，不称同协议第161窗。

软件已独立验收，完整图实际门槛按以下顺序执行。新目录必须不存在；示例目录不要与正在运行的实验共用：

```bash
cd /Users/yang/project/FlyDrones
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m cnsbridge.learning.train_gpu --initialize \
  --parent-run data/neural_training_mps_20261019_r3 \
  --out data/neural_training_gpu_new --credit-horizon 100 --theta-norm-limit 1000

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m cnsbridge.learning.train_gpu --gate --run data/neural_training_gpu_new

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m cnsbridge.learning.train_gpu --run data/neural_training_gpu_new --windows 1
```

gate是可丢弃的一次负标签联合更新，检查真实内部梯度、参数变化和固定特征纠错，不污染正式初始参数。首次训练一窗后，用最后一条命令另起进程恢复第二窗；实际验收通过后，每次最多 `--windows 4`，总预算新增160窗。失败保留现场并阻断继续，不自动调高门槛或绕过历史来源。`progress.json`记录新epoch窗数与实际Adam步数；`initialization/`、`gate/`、`segments/`分别保存完整来源、状态、损失/梯度/驻留、更新、恢复和资源证据。设备值只在边界下载，验算不回写GPU优化器。

资源仍申请6GiB、额外预留1GiB、每段600秒，实际上限受当前可用内存约束；不操作Docker/Noria、不控制无人机。不能在同一工作区并行加载模型。即使软件、算术和恢复通过，实际学习、闭环泛化及实飞仍须单独验证。

## 历史：从CPU完整状态迁到MPS前反向

历史入口为 `cnsbridge.learning.train_mps`，以下内容用于解释和恢复既有来源，当前修复训练使用上面的新入口。历史神经网络和读出前向/反向使用 MPS float32，感觉事件仍由原 NumPy float64 相位生成，小参数使用原 CPU Adam 与 float64 范数裁剪。后端名称是 `mps_forward_backward_cpu_adam`，不能称为所有运算都在 GPU。

当前 `data/neural_training_mps_20261019_r3` 已完成既定160/160窗（CPU继承60＋GPU累计100）。本批从62续至160，新增98窗用时50分46.55秒，每窗计算均值26.97秒；见[完整轮次实测](../specs/spec-008-trainable-snn/implementation-mps-finish98.md)。新增98窗的[独立保存证据验收](../specs/spec-008-trainable-snn/acceptance-mps-finish98-actual.md)已通过。最初迁移及两次失败记录保留在[迁移实测](../specs/spec-008-trainable-snn/mps-continuation-run-20261019.md)。本轮预算已耗尽，不再对该run执行续训命令；新增epoch或实验需另行登记。

仅在另建实验时，从旧joint完整存档迁移（输出目录必须不存在）：

```bash
cd /Users/yang/project/FlyDrones
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m cnsbridge.learning.train_mps \
  --cpu-run data/neural_training_joint_20261019_r1 \
  --out data/neural_training_mps_new --windows 1
```

仅对尚未达到登记预算的GPU运行，另起进程恢复（以下为通用示例，需替换路径）：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python \
  -m cnsbridge.learning.train_mps \
  --run /path/to/incomplete_registered_mps_run --windows 4
```

`--out` 必须不存在；每次只允许1–4窗，执行后退出。启动时读取真实CPU末档和全部合法前缀，继承其已完成窗口数与总共160窗的预算。不会重放前60窗、清空Adam或重新预热当前场景。进入新场景时才按原规则重置神经/感觉状态，训练参数和Adam保留。当前已完成GPU运行绑定CPU60末档，保留旧joint分支边界以维持来源链；未完成的运行可用 `--run` 续接。

为降低完整1000步训练的中间张量内存，生产策略固定每50神经步分段重算。跨块/跨帧不截断梯度，DN仍逐步累加；感觉驱动在重算时从私有快照恢复，不推进外部时钟。直接执行仅供明确数值fixture对照，完整图不提供绕过入口。

运行使用共享数值锁、6 GiB 请求、额外 1 GiB 系统预留、600 秒单段监督。实际 RSS 上限为 `min(6 GiB, 可用内存−1 GiB)`；MPS allocator 上限不高于该数值。RSS 与 MPS current/driver 都记录，不能相加当成总物理内存。工具不会操作 Docker/Noria，也不控制无人机。

新 run 的 `run.json` 绑定旧 run、checkpoint 文件摘要、前缀、完整图、数据集、实际新源码和后端。每段目录包含：

- `resume.npz`、`resume.json`：前向之前实际恢复的全部参数、Adam、膜电位、电导、待到达队列、不应期、感觉相位/事件数及时间游标，与来源逐项核对。
- `windows/<累计窗号>/evidence.npz`：两帧输入、期望动作、DN 特征、实际动作、raw/postclip 梯度和参数/Adam 前后值。
- `event.json`：独立 NumPy 算术复核、损失、内部参数变化和同步计时。前向、反向、CPU 裁剪/Adam、梯度及参数传输分别记录。
- `checkpoint/{arrays.npz,checkpoint.json}` 与最终 `window.json`：完整边界存档和封存标记。只有最终合法 seal 才计入进度。
- `resources.csv`、`mps_memory.csv`、`mps_budget.json`、`timing.json`、`worker_result.json` 和 `result.json`：资源、构造/恢复/保存及整个段结果。

`progress.json` 分开列出 CPU 继承窗数和实际 GPU 完成窗数。进程退出成功还需要独立保存数组复核，才会返回 `registered_mps_windows_verified`。若 OOM、系统预留或时间限制触发，保留失败现场及最后合法存档；原 CPU 进度不变。若出现非有限、优化器算术或协议错误，同一 run 的自动续接被拒绝，需要定位后用已审查新来源重新登记。

完整图核级测量此前未通过严格逐元素 CPU 等价门槛，用户已在 [Decision014](../.ai/decisions/014-resume-on-mps-with-recorded-numerics.md) 接受披露的设备数值差异。这个决定没有放宽存档精确恢复、自身损失/裁剪/Adam 算术或有限性检查。核级速度不能直接代替完整 BPTT 速度，CPU 历史窗口和 GPU 新窗口不是严格配对输入。软件小图检查也不证明完整 1000 步窗口的 GPU 内存足够，更不证明网络已经学会控制无人机。
