# 完整连接组的有界连续训练

`python -m cnsbridge.learning.train_snn` 每次处理明确登记的 1–4 个两帧窗口，然后退出。生产入口只加载完整 `full_reference`，使用已封存的四个 train 场景（combined、forward、vertical、yaw，各 80 帧），一个 epoch 共 160 窗。它训练固定教师轨迹上的离线模仿；[首两个完整图跨进程窗口](../specs/spec-008-trainable-snn/p3c-run-20261019-r1.md)已独立验收，尚无完整epoch、神经闭环跟随、收益或实飞结论。

规格：[P3c2](../specs/spec-008-trainable-snn/p3c-training-runner.md)。完整状态格式：[训练存档](TRAINING_CHECKPOINT.md)。父入口、登记与证据复核不导入 Torch；脑计算全部在受监督子进程内。

## 首两个实际窗口

以下命令已在独立软件验收并提交后执行，两段实际证据已获[独立PASS](../specs/spec-008-trainable-snn/acceptance-p3c-run-r1.md)。它们是已完成的首段与续段记录，不应重复创建此目录；继续同一run从frame4开始。新实验输出目录必须不存在，不覆盖历史或失败记录。

```bash
PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.train_snn \
  --graph data/malecns_v2/full_reference.npz \
  --dataset data/neural_teacher_20261019_r1 \
  --out data/neural_training_joint_20261019_r1 \
  --mode joint --windows 1

PYTHONPATH=src .venv-train/bin/python -m cnsbridge.learning.train_snn \
  --run data/neural_training_joint_20261019_r1 --windows 1
```

首段是 combined 帧 0/1，与 r2 的初值、动作、全部 DN 特征、原始梯度和更新比较。第二段是新 OS 进程，载入首段的真实参数、Adam、神经及感觉状态，先保存并核对恢复证据，再处理帧 2/3。首次实际门槛由协调者保持为这两个单窗调用，不将 `--windows 4` 用于跳过该门槛。

这两个实际窗口另经独立验收后，才在同一 run 显式继续，每次指定 `--windows 1` 至 `4`；剩余不足 4 窗时相应减少，不能超过 160 窗。没有自动完整 epoch 循环或失败重试。

独立冻结脑对照使用一个新的目录和 `--mode frozen_brain`，其余图和数据相同。它从原始 theta=0、相同读出种子重新开始，保留完整脑动态，仅训练读出；不能沿用 joint 的最终参数或状态。恢复时只接受 `--run` 和 `--windows`，不能换图、数据、模式或源版本。

## 计算和资源

每帧执行 50 ms 神经时间，dt=0.1 ms；新场景先有 60 ms 空白无梯度预处理。同一窗口两帧连续反向；更新后 detach 保留全部神经电位、突触状态、延迟队列、感觉相位与时钟。只有场景边界重置活动状态，参数和 Adam 矩跨场景继续。

joint 的 theta lr=0.001、读出 lr=0.0001，各组独立稳定裁剪至 L2=1；frozen_brain 只裁剪读出。父复核保留 float32 的实际 Adam 下溢、一般步旧矩和 lerp 乘加顺序，不只检查参数是否变过。无效标签推进状态但不贡献损失；整窗无效跳过整个更新。当前固定数据的 320 帧全有效。第一窗若全无效，明确拒绝零步存档。

每段额外预留 1 GiB，申请 RSS 上限 6 GiB，有效上限为 `min(6 GiB, available−1 GiB)`；墙钟 600 s，0.1 s 采样。工作区 OS 文件锁避免本入口的两个生产 worker 并行。锁文件的存在不代表进程活着。不会操作 Noria 或 Docker。

窗口封存时间使用单调时钟；CSV 的内部原点稍晚于登记原点。父复核采取保守边界，要求存在足够晚且预算内的采样。若窗口紧邻最后一个采样，原点偏移可能导致证据不足而拒绝；错误明确为 `clock offset unresolved`，不会扩大预算或伪称精确对齐。采样监督不是硬内存隔离。

## 输出和恢复

```text
run.json                      固定图、数据、协议、模式、源版本、运行时与 r2 锚点
progress.json                 从真实窗口重算的进度、轴覆盖、失败/资源与最终存档
segments/000001/
  registration.json           段起点、父档、前缀、精确窗口数、当前资源
  phase.json / resources.csv / worker.log
  resume.npz / resume.json    续段才有：任何前向前保存的实际恢复值
  worker_result.json
  supervisor_result.json     原始监督结果单独保留
  result.json                父复核，不以 exit 0 代替完成
  windows/000001/
    evidence.npz             两帧原像素/标签、动作/特征、参数、梯度、前后实际 Adam
    event.json               游标、状态摘要、损失、更新/裁剪/下溢和时间
    prefix.jsonl             真实事件的固定相对路径与摘要
    checkpoint/              完整 P3c1 存档
    window.json              最后封存，完成标记
```

顺序是 evidence → event → prefix → checkpoint → window。没有最后 window manifest 的尾部保留但不计进度；只有 checkpoint 目录不算完成。父进程逐窗验证，状态数组在本窗检查后释放，不将 160 份完整状态常驻内存。

资源或进程失败仍为 failed，可以另外列出 `recoverable_prefix`；只能下一次显式调用恢复。数值/协议失败不会列为可续训练，并阻止原 run 自动续跑。坏父链、改动的数据/日志、重复窗口、错误路径或损坏已封存档均拒绝，不静默回滚到更旧档。失败和未封存文件不自动删除。

`progress.json` 的 `epoch_complete` 仅表示登记的离线训练窗口全部完成，不表示学会目标跟随。模型收益需独立冻结脑、急性恢复、视觉及拓扑比较和未见场景神经闭环证据。

## 软件验证

本批 27 项定向检查通过。joint/frozen 各自在两个真实 OS 子进程中保存/恢复首两窗，再接上下一场景，结果与连续四窗的全部参数、实际 Adam、完整状态、动作/特征精确一致；这仍是明确数值小电路，不是完整图实测。另覆盖一般步 Adam 的 160 步旧矩传递、抵消与下溢、全无效窗跳步、保存中断、资源前缀、坏链、重算摘要后的语义篡改、r2 超差、假成功和 Torch-free 父入口。
