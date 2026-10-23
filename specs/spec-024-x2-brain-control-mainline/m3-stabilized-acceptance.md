# M3 固定代理梯度增益探针独立验收

2026-10-23。独立验收者：`roadmap_m3_stabilized_review`。风险 **S3**，范围仅为 [第二次探针合同](m3-stabilized-contract.md) 与 [实施报告](m3-stabilized-implementation.md) 所列五个变更文件，以及它们直接依赖的身份和资源保护代码。未修改业务代码。

**结论：PASS，准入一次 `probe-stabilized` 实际完整脑探针。** 未发现阻断该次运行的确定问题。本结论不是探针数值通过，也不准入正式训练、部署或 M4 产品验收。

## 审查依据

- `brain/mps_trainable.py:45` 的缩放使用 `detach + gain * (spikes - detach)`；gain=1 返回原 tensor，`step` 默认分支没有新增 tensor 运算。固定 0.1 只改变 surrogate 导数；没有改阈值、曲线宽度、复位、完整图或前向输入。
- `learning/x2_behavior_policy.py:118` 将新 backward 描述、gain、loss scale 1、核心与策略源码哈希纳入新身份。`forward` 每 50 步 checkpoint，v/g/pending 与累计 DN 特征继续跨块、跨两窗传递；观察器在 checkpoint 返回后登记，未在重计算内部重复登记或截断状态。
- `learning/x2_behavior_training.py:233` 固定首轮目录及已知 policy/checkpoint，复核历史源快照并封存配对文件哈希。`:258` 的迁移仅允许已登记的核心/策略源码与 backward 身份差异，保留图、配置、映射、初始化及参数合同；全部初始化参数从首轮原始数组精确复制并核对传输后的数组身份。
- `:297` 比较每个数组的 dtype、shape、C 顺序字节 SHA256 和数组清单，保存不一致报告后拒绝。`:695–734` 在任何 backward/step 前依次比较预热状态、两帧编码输入/神经特征/三头原始输出和两个后继状态；观测、时钟与新旧身份也分别校验。首轮观测与目标固定为 `WAVE_RIGHT/START`、`HOLD/FINISH`，只用第二窗损失，没有 loss 缩放、替换样本或访问 validation/locked-test。
- `:336` 的 50 步 v/g/pending hook 将收到的原梯度流式存档并原样返回；未遍历的末端状态明确记为缺失。所有已观察边界梯度须有限，第一窗末 v 梯度另须有限且非零，不能以空记录代替跨决策信用证据。
- `:469` 在梯度准入与参数更新前保存原始梯度和更新前参数；固定探针要求 theta、行为头、阶段头有限且非零，参数头按 mask 为零。fresh Adam、float32 原始平方表示范围、一次更新上限均在 `step` 前检查；更新后核对参数与 Adam 状态有限、应变参数实际变化、屏蔽头不变。异常路径保留实际 step 尝试/完成状态及前后参数，不把部分失败记为成功。
- `:138` 的新输出目录、独占锁、仿真关闭预检和实际源码快照沿用受监督入口。600 秒预算扣除父进程准备耗时后传给子进程监督；RSS/MPS 有效预算取 min(6 GiB, 可用内存−1 GiB)，父进程持续检查系统余量，线程数固定 4/1，禁止 MPS CPU fallback。资源监测仍是现有采样保护，不声称为操作系统硬隔离。

## 实际检查与停止范围

实施者已记录 **30 passed in 1.28s** 的定向 CPU 检查；本验收逐项阅读相关测试，没有机械重跑。新增验证仅一次 stdlib 真实归档读取：调用 `prepare_paired_probe(Path.cwd())`，退出码 0，历史源文件与登记哈希、checkpoint 完整性及已知身份通过；首轮 **53 份配对文件**存在并可生成本次封存哈希。状态/帧数组的内容配对留给实际新 run 执行。返回的 policy/checkpoint/registration 身份如下；没有导入 Torch、NumPy、SciPy，没有构造模型。

- policy：`b8cb7e89613e73071c66b9c3a01007b92c0002187154b56ca49fe3414bd44a55`
- initial checkpoint：`66c4717399a9f5319252f1f0b56ea2665c8a0afda02ab4af412c134604957d57`
- registration：`82f816735f34510508222911603d26b1e64de1bf98cd05b4943f08569596971d`

本次未启动完整脑、MPS、VM、ROS、仿真或训练，没有全仓检查、commit/push。已达到本次独立准入审查所需范围，停止追加验证。

## 受审源码身份

| 文件 | SHA256 |
| --- | --- |
| `brain/mps_trainable.py` | `a2d28cfcbb727ec387663f3ee1912cfa27b784ce15bd490aff19514011e553ac` |
| `learning/x2_behavior_policy.py` | `07b06afa324f2905a8fee72bd5b79b614780dae11aad6f3da6ec6affbf704a19` |
| `learning/x2_behavior_training.py` | `3bc3d9d4e7dee3def12501d8a2fa808f00ec285b0cf9f45e25da13223860c9d5` |
| `experiments/x2_behavior_training.py` | `dc2a107baea2864ff86edd12e3e99b52ce48484dbd60ceec3ecbb349c55967f9` |
| `tests/test_x2_behavior_training.py` | `8bab5164b7d869133adf47dbdc5d186be4ec976d21dc5c80db6dad962959892d` |

前四个路径相对 `src/malecns_lab/`。实际运行另封存完整传递依赖源码；并行 M4 的加载接口修复不属于此次探针验收范围。

协调者可依合同在新目录 `data/x2_behavior_probe_20261023_r2` 串行执行一次。仍须以实测确认完整图前向字节配对、长信用梯度、一次参数更新、耗时及资源；任何一项失败均保留原始结果并停止本次探针，不扫描增益、不混入损失缩放、不自动追加预算。
