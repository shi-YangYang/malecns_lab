# 神经策略训练序列

> 命名迁移：本文操作入口已更新为 `malecns_lab`。涉及旧 run／checkpoint 的目录保留为参数示例和历史证据；不能直接用新源码续跑或重新登记旧实验。须使用匹配的源码身份与新输出目录，精确恢复旧实验需原 commit。见[迁移说明](ARCHITECTURE.md#环境迁移与历史来源)。

`learning.neural_data` 生成目标跟随的**仿真教师数据**。它运行既有速度级环境，不加载 MaleCNS，不训练神经策略，不连接 Tello。软件检查通过不代表已完成真实数据采集或学会跟随。

在项目根目录创建一个新的目录：

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.neural_data \
  --out data/neural_training_sequences_r1
```

默认相机为 160×120 BGR，环境时间步为 0.1 s、响应时间常数为 0.25 s。API 可显式提供 `TargetCamera`；CLI 可用 `--dt-s` 和 `--response-tau-s`。这些设置全部进入数据身份。命令固定收集按 ID 排序的四个 `train-sine-*` 完整场景，不接受任意场景或逐帧随机划分。默认每个场景 8 s、80 帧；这是计划数量，实际数量和失败状态以清单为准。

## 行为与监督来源

采集行为来自 `visual_teacher`，使用该场景的期望距离；环境负责限幅和目标丢失后的零指令介入。监督标签另从每帧 `truth_teacher` 得到，不把实际执行动作当作正确标签。

每场景的 `sequence.npz` 包含：

| 字段 | 数值模式 | 含义 |
|---|---|---|
| `frames` | uint8 `[N,H,W,3]` | 同时刻无损 BGR 像素 |
| `time_s` | float64 `[N]` | 连续、有序环境时间 |
| `teacher_raw` | float64 `[N,3]` | 未限幅真值教师输出 |
| `teacher_target` | float64 `[N,3]` | 经过 `limit_follow_action` 的监督目标 |
| `executed_action` | float64 `[N,3]` | 视觉行为策略实际执行的动作 |
| `valid` | bool `[N]` | 同帧像素检测有效性，即后续模仿损失掩码 |

三个动作轴为 `vx/vz/wz`，单位 `m/s、m/s、rad/s`，坐标 FLU；绝对上限分别为 `0.3、0.2、0.5`。无效帧仍保留原始标签和像素；后续训练必须继续推进神经状态，仅屏蔽该帧动作损失，不能删帧或改为零标签。

每场景还保留原始 `summary.json` 和 `trajectory.jsonl`，含真值、介入原因、下一时刻真值和覆盖/失败信息。神经策略接口只接收图像；这些教师与真值字段由训练器和评分器保管，不送入神经读出。

## 封存与读取

```python
from malecns_lab.learning.neural_data import read_neural_data

dataset = read_neural_data("data/neural_training_sequences_r1")
for sequence in dataset.sequences:
    # 这里仍未调用神经模型；图像与监督在不同字段。
    print(sequence.scene["scene_id"], len(sequence.frames), sequence.valid.sum())
```

清单 `manifest.json` 最后写入，绑定完整场景定义、train split、相机、环境时间、行为/教师来源、轴与上限、源码 SHA256、所有数据文件 SHA256、数组字段/shape/dtype、计划与实际样本数。只有全部场景达到计划长度且无环境 failure 才标为 `ready`。失败目录保留证据并标为 `collection_failed`，CLI 非零退出；中途写入失败则没有可加载清单。

loader 使用 `allow_pickle=False`，核对当前登记场景和源码身份，再校验文件、数组、时间轴、限幅、像素 valid、原始 rollout 与标签关系。它还按登记的场景、相机和时间参数重放轻量环境，逐帧精确核对像素与当前真值，用已核对的执行动作推进，再核对下一状态及完整结束边界；这不加载神经图。即使保留训练场景声明、把验证场景的实际像素/轨迹/标签整体移入并重新计算文件及清单 hash，也会被拒绝。当前源码或场景定义变化后旧数据可能不再兼容；应保留原资产并在新目录重新登记，不覆盖或篡改历史。

返回的数组由不可写字节支撑，不能原地更改。本格式不含神经特征缓存、训练结果或检查点；无效帧比例、行为策略成功与后续 SNN 表现应分别报告。
