# V2 完整连接组离线诊断

当前入口把图像变成显式工程视觉特征，驱动完整 MaleCNS 神经元参考图，记录神经活动与因果特征。它不读取动作标签，不调用旧 `MotorDecoder`，不连接 Tello，也不输出飞行指令。训练与动作效果属于后续独立阶段。

当前参考图实测为 166,700 个节点、25,582,938 条内部有向边、124,177,617 个突触，保留所有已发布内部正权重、自环和孤立节点。完整的定义和片段边界见 [V2 设计](MODEL_V2.md) 与 [P0 实施记录](../specs/spec-007-model-v2/implementation-p0.md)。

## 1. 完整参考资产

已有 `data/malecns_v2/full_reference.npz` 时直接使用。需要从已下载的完整官方发布重新构建时，必须指定新的输出目录：

```sh
.venv/bin/python -m malecns_lab.brain.reference \
  --data-dir data/malecns_v1 \
  --identity-dir data/audits/malecns_v2_identity \
  --out-dir data/malecns_v2_rebuild
```

构建器拒绝已有目录，不覆盖旧图。身份清单、11 文件来源、逐原始行的边分类索引和全部原始端点身份随参考图一起记录。这里不使用旧 `build-brain` 的默认阈值 3。

## 2. 先看标准视觉刺激

短探针包含 2 s 固定预热和 0.1 s 观察；重复两次可检查显式重置后的确定性：

```sh
.venv/bin/python -m malecns_lab.diagnostics.replay_v2 \
  --stimulus static --repeat 2 --observation-seconds 0.1 \
  --run-seed 0 --budget-seconds 120 \
  --out data/diagnostics_v2/my_static_probe
```

比较静态纹理、左/右/上/下平移、膨胀六种刺激：

```sh
.venv/bin/python -m malecns_lab.diagnostics.replay_v2 \
  --stimulus all --observation-seconds 0.5 \
  --run-seed 0 --texture-seed 7 --budget-seconds 180 \
  --out data/diagnostics_v2/my_standard_stimuli
```

每种刺激独立重置，先用同一静态纹理预热 2 s，再观察 0.5 s；默认生成尺寸为 96×72。平移 10 px/s、膨胀率 0.35 s⁻¹ 是工程诊断设置。输入标记为 `generated_engineering_stimulus`，脑仍是完整真实参考图。它不能作为真实录像、动作监督或生物行为效果证据。六种刺激来自同一个纹理族，不是六个独立采集场景。

完整图运行按仿真时间推进，不能把视频时长当作计算耗时。本机首次短探针每 50 ms 仿真窗口约需 0.3 s 计算；这不是实时飞行能力。预算在窗口之间检查，超预算保存前缀与未处理尾段，不剪图、不降低模型规模。

## 3. 重放录好的视频

已有项目地面采集目录包含 `session.json`、`video.avi`、`frames.csv` 时：

```sh
.venv/bin/python -m malecns_lab.diagnostics.replay_v2 \
  --recording logs/my_recording \
  --family-id room_a_phone_sequence \
  --stimulus-kind object_motion \
  --run-seed 0 --budget-seconds 600 \
  --out data/diagnostics_v2/my_recording_run
```

`scene_id/session_id` 默认读取采集清单；`family_id` 必须显式提供。同一素材的剪辑、增强、重复运行与不同输入种子必须使用相同素材族标识，不能改名字制造独立训练/测试样本。`labels.csv` 不会被读取，尚未标动作也可做神经诊断。

普通视频需要显式身份：

```sh
.venv/bin/python -m malecns_lab.diagnostics.replay_v2 \
  --video /absolute/path/clip.avi \
  --scene-id room_a_scene_01 --session-id capture_01 \
  --family-id room_a_phone_sequence --stimulus-kind object_motion \
  --out data/diagnostics_v2/plain_video_run
```

本期只接受名义 20 fps 的视频，不隐式变速或重采样。视频时间是 `frame_index / 20`，不是已验证的相机曝光时间。采集目录中的 host dequeue 时间仅保存为诊断，不替换播放时钟。普通视频没有这些时刻时，数组保存 NaN 缺失值。

固定无人机、移动手机应标记 `object_motion`；移动相机的录像标记 `camera_motion`，两者混合为 `mixed`，不确定为 `unknown`。这些是刺激描述，不能自动转换成期望转向或升降标签。

## 4. 显式分段与失败

默认处理整段视频。若采集协议已经规定独立连续片段，可在运行前准备 JSON：

```json
[
  {"segment_id": "first", "start_frame": 0, "end_frame": 120},
  {"segment_id": "second", "start_frame": 140, "end_frame": 300}
]
```

用 `--segments /absolute/path/segments.json` 指定。范围为左闭右开、不得重叠；每段重新初始化视觉、膜电位、突触状态、延迟、不应期、事件步号、历史及 EMA，并重新预热 2 s。片段仍属于原来的场景、会话和素材族。不得根据标签、预测分数或某次神经响应临时挑段。

`run_summary.json` 的 `source_coverage` 还按整个源视频统计片段外未登记帧，避免只显示选中片段的覆盖率。片段外帧、失败后的未处理尾段与实际失败帧是不同类别；重复运行仍不是新增独立场景。

任一半幅无有效光流、扩张无法辨识、图像损坏或时间错误，立即停止当前段。失败帧不生成输入率、不推进神经网络或视觉对照历史；已经产生的有效前缀、失败原因、侧别及未处理尾段分别保存。程序只处理预先声明的后续独立片段，不自动跳过坏帧接着模拟。

## 5. 读取记录

每次运行创建新目录：

```text
run_manifest.json          模型/源码/输入身份、种子、预登记片段及预算
run_summary.json           每段覆盖、失败、处理耗时及进程峰值 RSS
source_00_repeat_00_segment_000/
    metadata.json          场景/会话/素材族、来源、覆盖和全脑摘要
    rows.npz               逐帧视觉/输入/神经/质量/时间数组
    neuron_totals.npz       全部节点 body ID 与该段累计发放数
```

`rows.npz` 关键字段：

| 字段 | 含义 |
| --- | --- |
| `frame_index`, `input_cutoff_s`, `target_time_s` | 源帧号、当前可见输入时刻、下一 0.05 s 目标时刻 |
| `valid`, `initialization`, `prediction_valid`, `target_available` | 输入有效、首帧初始化、完成预热且目标在段内、目标时刻有记录支持 |
| `visual_features` | 左右各七通道，共 14 维；亮度、ftb、btf、up、down、loom、loom_speed |
| `input_rates_hz`, `input_event_counts` | 14 个视觉群每节点输入率与实际事件总数；对应群人数在 metadata |
| `neural_counts`, `neural_rates_hz` | 固定 33 个输出 body ID 的当前 50 ms 发放数及 Hz |
| `neural_history_hz` | 4 个窗口，按最旧到最新排列，132 维 |
| `visual_history` | 同样 4 窗的 14 特征，另加 0.05/0.2/1 s EMA，共 98 维 |
| `group_rates_hz` | 元数据指定群组的均值，仅用于诊断 |
| `L_*`, `R_*` | 每半幅网格亮度、px/s 光流、有效掩码、纹理、扩张和 EMA 诊断 |
| `network_*` | 全脑当前窗发放、活跃比例、高频比例与分位数；200 Hz 是诊断阈值，非筛除规则 |
| `simulation_start_ms`, `simulation_end_ms`, `processing_ms` | 仿真窗口与真实处理耗时，分开记录 |

无效浮点值为 NaN，未计算计数为 -1；必须使用有效性掩码，不能补零。JSON 缺失值使用 null，不写非标准 NaN 数字。首帧允许亮度输入与零运动初始化，但只用于预热；前 40 窗不作为预测样本。最后一帧缺少段内目标时刻，`prediction_valid=False`。

简单查看：

```python
import json
import numpy as np
from pathlib import Path

run = Path("data/diagnostics_v2/my_standard_stimuli")
manifest = json.loads((run / "run_manifest.json").read_text())
print(manifest["model_source_id"])
with np.load(run / "source_00_repeat_00_segment_000/rows.npz") as rows:
    valid = rows["prediction_valid"]
    print(rows["neural_rates_hz"][valid])
    print(rows["input_event_counts"][valid])
```

参数、图、选组、视觉算法、共同事件算法、时间或特征顺序变化，都会影响来源身份。运行种子单独记录；同种子下，每个输入 body ID 和仿真步的外部事件不因其他节点是否存在、遍历顺序或内部发放而改变。它是每步最多一次事件的 Bernoulli 近似，不是精确连续时间 Poisson 过程。

现阶段记录只能回答输入通道是否响应、神经活动是否变化、是否可重复及计算成本如何。神经群有变化不等于“知道该往哪里飞”；没有独立动作标签、训练与对照时，不报告动作准确率、连接组优势或避障能力。

## 6. 生成可复用汇总与图

对任意已有 V2 run 执行：

```sh
.venv/bin/python -m malecns_lab.diagnostics.report_v2 \
  data/diagnostics_v2/my_standard_stimuli \
  --out data/diagnostics_v2/my_standard_stimuli_report
```

报告只读取保存的记录，不重新运行脑。新目录包含 `diagnostic_summary.json` 和 `diagnostic_summary.png`，显示输入通道、输出群与全部 33 个输出细胞在可预测窗口的均值。普通单段录像、重复运行或部分失败同样支持，图中 `n` 是可预测窗口数，不是动作得分；零可预测窗口显示灰色，JSON 均值为 null，同时保留完整覆盖和失败记录。

绘图依赖已有可选 `analysis` 环境中的 Matplotlib；只需要 JSON 时可加 `--no-plot`。报告目录也必须是新目录。

## 7. 已完成的完整图诊断

2026-10-16 已实际运行完整参考图，运行记录在 `data/diagnostics_v2/standard_stimuli_20261016/`，可复用报告在 [diagnostic_summary.json](../data/diagnostics_v2/standard_stimuli_report_20261016/diagnostic_summary.json)，图见 [diagnostic_summary.png](../data/diagnostics_v2/standard_stimuli_report_20261016/diagnostic_summary.png)。这些是本地运行资产，不纳入 Git。输入种子为 0、纹理种子为 7，六种刺激各为 2 s 预热加 0.5 s 观察；每段 50 帧全部输入有效，除去预热和最后目标边界后各有 9 个可预测窗口。

下表是这 9 个窗口中的群组平均放电率，单位 Hz，**不是动作输出**：

| 刺激 | DNg02 左 | DNg02 右 | DNp03 左/右 | DNp01 左/右 |
|---|---:|---:|---:|---:|
| 静态 | 18.22 | 7.14 | 0 / 0 | 0 / 0 |
| 左移 | 13.33 | 6.98 | 0 / 0 | 0 / 0 |
| 右移 | 14.22 | 1.75 | 0 / 0 | 0 / 0 |
| 上移 | 29.48 | 26.35 | 0 / 0 | 0 / 0 |
| 下移 | 3.56 | 1.27 | 0 / 0 | 0 / 0 |
| 膨胀 | 18.52 | 9.05 | 15.56 / 0 | 6.67 / 0 |

这些记录表明当前编码能够改变完整图中的输出活动，也暴露出左右响应不对称、部分群在短观察窗内静默。一次纹理、一个输入种子和短窗口不能证明稳定的方向编码，不能据此指定左转/上升等动作意义。另一次静态短探针在相同输入种子下重置重复两遍，输入事件、神经输出、历史和有效性等 10 组数组完全一致，记录见 `data/diagnostics_v2/pilot_static_20261016/determinism_check.json`。

六段合计 15 s 仿真耗时 96.83 s，进程峰值 RSS 1,102,233,600 字节；每 50 ms 窗口的处理耗时中位数约 304–324 ms。本机可以离线运行完整图，目前不能按 20 Hz 实时运行。这里保留了完整图，没有为了加速剪边。

下一步使用真实录像重复诊断，再按独立任务规则标注；[学习工具](LEARNING_V2.md)会拒绝将这些生成的诊断刺激作为真实动作训练数据。
