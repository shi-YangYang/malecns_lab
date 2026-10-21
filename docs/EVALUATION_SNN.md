# 完整连接组神经闭环评估

> 命名迁移：本文操作入口已更新为 `malecns_lab`。涉及旧 run／checkpoint 的目录保留为参数示例和历史证据；不能直接用新源码续跑或重新登记旧实验。须使用匹配的源码身份与新输出目录，精确恢复旧实验需原 commit。见[迁移说明](ARCHITECTURE.md#环境迁移与历史来源)。

P3d 工具让神经输出改变模拟相机的位置与下一帧。它使用已登记的红色方形标记跟随环境，不能代表任意物体识别、真实飞行、避障或拓扑优势。正式协议见 [P3d](../specs/spec-008-trainable-snn/p3d-closed-loop-evaluation.md)。当前实现检查只使用明确的微型数值电路；完整图实际评估需满足下列准入。

## 前置

- joint 与独立 frozen_brain 各自完成冻结的四场景 160 窗训练。工具从唯一封存链及完整 checkpoint 核验，并绑定每段父 result、原始 supervisor / worker、完整终态文件清单及 SHA。最终段必须有通过完成检查的 completed 父终态；缺父 result、坏证据及最终失败段均拒绝。中间已验证的资源/IO失败前缀可经显式续训最终完成，原失败记录保留。不能只修改 progress.json，不能选择中途最低损失档。
- 两次训练须使用相同的完整图、数据、初始来源和运行环境。冻结脑必须保持原始 theta，不复用 joint 训练出的读出。
- 源码与运行环境采用内容身份绑定。工具拒绝来源改变或坏链；不覆盖已有套件，不加载缩图或合成生产替代模型。
- Noria / Docker 保持。每段动态 RSS 上限为 min(6 GiB, available − 1 GiB)，墙钟 600 秒，采样 0.1 秒；与训练共享 OS 锁，不能并行加载图。

以下命令在项目根目录使用 `.venv-train`；路径为示例，须替换为确实完成训练的目录。

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.evaluate_snn register \
  --joint-run data/neural_training_joint_20261019_r1 \
  --frozen-run data/neural_training_frozen_20261019_r1 \
  --out data/neural_evaluation_r1
```

登记只核验及封存参数，不前向运行模型。五个条件依次为 joint、frozen_brain、acute_restore、direct_visual、zero。acute 保持 joint 读出，在场景初始化前恢复真实保存的原始 theta；每个条件另外绑定参数数组值，结构 policy_id 相同也不能交换参数。

## 首次实际门槛与续段

首个条件和场景固定为 joint / val-triangle-forward，先运行一帧，再以第二次命令启动新进程恢复一帧：

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.evaluate_snn run \
  --suite data/neural_evaluation_r1 --condition joint --scene val-triangle-forward --frames 1
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.evaluate_snn run \
  --suite data/neural_evaluation_r1 --condition joint --scene val-triangle-forward --frames 1
```

两段退出后须按协议完成新的独立实际验收，检查奇数边界恢复、全部状态、参数和环境。工具强制首两帧分别请求一帧；独立验收是研究工作流要求，不由程序声称已经完成。这两帧直接保留为正式 val 前缀，不重复脑计算。

实际门槛通过后每次显式请求 1–40 帧。已完成两帧时可请求 38 帧到第 40 帧，再请求 40 帧结束场景；超过剩余配额会拒绝。通常每十帧封存一个块，段末和真实物理终止允许短块。段配额完成只意味着本段完成，不代表场景成功。没有自动跑完整矩阵或失败重试。

每个条件完成 val 的 forward、yaw、vertical、combined 四场景，再进入下一个条件。所有五个条件全部 val 终局后执行：

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.evaluate_snn summary \
  --suite data/neural_evaluation_r1
```

此时工具核验并封存 validation_summary.json，才允许 test。test 顺序为 chirp-forward、chirp-yaw、chirp-vertical、chirp-combined、chirp-occluded、excursion-out-of-view，场景完整名称以 `run --help` 为准；每个条件跑完全部 test 再进入下一个条件。不得根据 test 调参或选择模型。

## 保存内容与证据范围

- run.json、conditions/：真实最终训练来源、协议、源码/runtime、有效 theta/读出及参数数值身份。
- episodes/：每场景独立初始化；完整状态保留电位、电流、不应期、全部延迟队列、感觉相位与事件计数，支持 frame 0、奇数帧及终局，无 Adam。
- segments/：原始资源监督结果、实际恢复值、worker 记录和父复核。chunks/ 保存所有图像、完整 DN 特征、raw / executed 动作、检测、介入、真值、误差、事件前缀与完整状态。
- 神经前向只接收图像和自身状态；检测失效仍推进神经，之后外层监督置零，介入单列。直接视觉及零动作基线不伪造 DN 或神经状态。
- 父进程不导入 Torch，不重放脑前缀；从初始环境重放执行动作，逐帧核对图像、检测、监督、位置和时钟。父独立复算保存 DN 到动作的转换与所有评分；这不独立证明 DN 的生物真实性。

summary.json 给出所有场景、固定场景集合等权均值及配对差值。`terminal` 仅表示环境已经终止，`evidence_complete` 还要求该终局所属段有父进程验证的 completed 终态；后者才开放下一场景、test、完整均值及配对比较。观察率/合格率分母保持计划 80 帧，物理提前失败不缩分母；缺失不当零分，局部片段 MAE 与覆盖和失败一起报告。success 须正式证据完成、合格率至少 0.8、完整 80 帧且无物理失败；未正式完成的原始轨迹评分单列 `diagnostic_metrics`，不作为正式成功。有效阴性实验也保留，不要求改善才承认证据。并列神经时间、环境时间、墙钟、资源采样和阶段耗时，不把模拟 0.1 秒当成实际推理速度。

资源或 IO 失败仍保留 failed 原值，已封存且资源采样覆盖、环境尚未终止的有效前缀可由下一次显式命令恢复；未封存尾部不计数。若失败发生在第 80 帧或物理终止封存之后，保留 terminal 和局部评分，但不升级为正式完成、不开放后续场景，也不再次前向或自动复核放行。数值/协议错误、坏 seal、来源改变、错参数、重复/跳帧均阻止自动续跑，需查明原因并登记后续方案。无有效记录时不生成虚假的行为 success。终局只能汇总，不继续前向。
