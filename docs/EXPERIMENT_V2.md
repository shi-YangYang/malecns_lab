# 完整连接组的离线研究工具

本流程把完整真实参考图 G、重连图 R、权重置换图 W，以及通路边干预 P 放进同一份预登记协议。每个图与输入种子分别重放、分别拟合读出。这里没有设备连接或飞行入口。

当前用户尚未录制动作数据，因此已执行的只有完整图信号诊断和一次完整 W 图研究路径接通。真实录像上的动作解码、连接组贡献与跨设备效果尚未验证。标准人工视觉刺激只能用于诊断，不能作为动作训练录像。

以下命令在项目根目录执行；所有输出路径必须尚不存在。

## 1. 显式重放研究图

生产 `Brain` 与默认完整图重放器拒绝 `research_control`。研究图必须提供独立母图，经专用加载器校验后进入离线运行器：

```bash
.venv/bin/python -m malecns_lab.learning.research_replay_v2 \
  --parent data/malecns_v2/full_reference.npz \
  --control data/malecns_v2/controls/W_seed201_toolcheck.npz \
  --out logs/v2-W-static-new-run \
  --stimulus static --run-seed 0 \
  --observation-seconds 0.1 --budget-seconds 180
```

每个独立片段固定先预热 2 秒。上例随后观察 0.1 秒；由于预测目标位于输入之后 0.05 秒，最后一帧没有片段内目标，不进入有效预测。不能据此把 42 个输入帧说成 42 个动作样本。

研究运行器真正用派生图权重重建 LIF 网络，保留相同的感觉编码、输入事件、偏置、输出 body ID 和因果历史。派生图身份、母图身份和研究适配器代码进入新的 `model_source_id`；`common_computation_id` 对比除图结构外的共同计算。不同源身份的特征不得伪装成同一模型来源。

录像入口将 `--stimulus` 换为以下二选一，并提供身份：

```text
--video /绝对路径/video.avi
--recording /绝对路径/地面录制目录

--scene-id 场景名 --session-id 采集会话名 --family-id 素材族名
--stimulus-kind object_motion
```

录像须为名义 20 fps；不隐式重采样。`--segments` 可指定预声明的有序、不重叠帧区间 JSON，每段独立重置并预热。无效感觉输入终止当前段，保留有效前缀、失败行和未处理尾段；不会自动重新开始以挑选可用片段。运行器不读取动作标签。

## 2. 先写协议，再登记

生成模板不会创建数据或填入假定的容差：

```bash
.venv/bin/python -m malecns_lab.learning.experiment_v2 template \
  --out logs/experiment-protocol.template.json
```

模板含 `null` 与空列表，必须由实际任务信息补齐后才能登记：

| 字段 | 内容 |
|---|---|
| `phase` | `pilot` 或 `confirmatory` |
| `parent` | 完整参考图路径 |
| `graphs` | 图 ID、角色和资产路径；必须恰有一个 G；R/W/P 的角色与资产来源必须相符 |
| `input_seeds` | 明确、不重复的输入事件种子；所有图共用同一列表 |
| `task` | 独立教师、标注规则、动作语义、vx/wz 的 SI 容差和静止容差 |
| `practical_delta` | 预先规定的归一化误差最小实用改善量；工具不代填经验值 |
| `budget_seconds` | 整个执行的墙钟预算 |
| `replay_budget_seconds` | 单个图×输入种子重放的墙钟上限，不得大于总预算 |
| `bootstrap` | 明确的重采样种子和重复数 |
| `comparisons` | G 对哪些登记图或 `visual`/`mean`/`zero` 做描述性配对比较 |
| `ridge_candidates` | 固定为 `10^-4,…,10^4` 九个候选；每个独立拟合相同预算 |

图条目结构：

```json
{
  "id": "W201",
  "role": "W",
  "path": "/绝对路径/W_seed201_toolcheck.npz"
}
```

`visual`、`mean`、`zero` 是保留的基线名称，不能作为图 ID。图的随机种子从真实派生产物读取，不通过改文件名增加“独立随机图”。不同 graph ID 不能重复同一个图来源身份。

每个真实素材条目结构如下；其中数值示例只说明帧索引格式，实际帧范围须来自自己的录像：

```json
{
  "id": "scene-a",
  "video": "/绝对路径/video.avi",
  "scene_id": "scene-a",
  "session_id": "session-a",
  "family_id": "family-a",
  "stimulus_kind": "object_motion",
  "split": "train",
  "labels": "/绝对路径/labels-v2.json",
  "label_format": "v2",
  "segments": [
    {"segment_id": "whole", "start_frame": 0, "end_frame": 600}
  ]
}
```

地面采集目录使用 `recording` 替代 `video`，仍须显式登记场景、会话和素材族。多个裁剪、重放、增强或随机种子来自同一材料时共用族身份；同一场景、会话、族、源 ID 或视频内容哈希不得跨训练/验证/测试分区。

标签契约见 [V2 学习说明](LEARNING_V2.md)。标签必须绑定输入来源，六轴 FLU/SI 顺序固定，只监督已明确标注的 vx/wz。未标注轴或时间不是零标签；旧 RC、神经读出或实测速度不能自动充当期望动作。登记仅校验标签结构/来源和文件哈希，不产生动作分数，也不能证明使用者从未看过测试数据。

登记与只读检查分开：

```bash
.venv/bin/python -m malecns_lab.learning.experiment_v2 register \
  --protocol logs/experiment-protocol.json \
  --out logs/experiment-registration.json

.venv/bin/python -m malecns_lab.learning.experiment_v2 validate \
  --registration logs/experiment-registration.json
```

登记文件冻结实际图、视频、会话文件、动作标签和相关代码的哈希，包含图种子、输入种子、完整分区清单、参数搜索预算和任务容差。任何冻结文件或相关代码改变，执行时会拒绝；应建立新协议，不能覆盖旧登记。

## 3. 执行离线提取与独立拟合

有真实数据并完成登记后运行：

```bash
.venv/bin/python -m malecns_lab.learning.experiment_v2 execute \
  --registration logs/experiment-registration.json \
  --out logs/experiment-run-new
```

执行顺序为完整 G，再逐一处理登记对照；每图×每输入种子独立提取。感觉有效性、输入截止、目标时间、监督掩码和视觉特征必须严格一致，不以取交集的方式隐藏条件差异。

每图×每输入种子独立拟合神经读出及视觉读出，训练集拟合标准化与参数，验证集选择九个岭候选；均值只取训练集目标，零基线固定。R/W 不使用 G 已训练权重。所有 V/M/Z 与神经条件使用同一评分行清单。

`pilot` 不提取或评估测试源，协议中的测试材料仅冻结身份。`confirmatory` 必须先有冻结登记，才提取测试源并在验证选择之后评分；测试结果不参与参数选择。当前工具保存每个图的配对区间及图间差异，结果仍标为描述性：没有自动实施多重比较校正或随机图排名检验，不据一个区间自动宣称连接组的确认性优势。

任一登记重放失败、预算耗尽、场景无覆盖或条件有效行不一致，结果为 `incomplete`。已处理前缀、零有效场景、失败与未执行任务保留；不会删掉不利图继续汇报“优势”，也不会在仅部分重放成功时拟合可用子集。

总墙钟预算在重放、数据准备、普通拟合与急性 P 计算/保存、聚合评分、每次配对计算和最终报告保存的阶段边界检查。当前不可中断的计算或写入可返回；发现超时后停止后续计算，写入带 `stage`、`reason=total_budget_exhausted` 及全部任务状态的 `run_failed.json`，其 `comparisons` 为空，未开始任务标为 `not_executed`。若最终报告写入本身越过预算，该报告也改为 `incomplete` 并清空比较；已有局部文件保留，不通过缩短 bootstrap 或删除条件凑足预算。

输出包含 `execution_plan.json`、各图/种子重放产物、学习协议、独立读出、逐行预测、覆盖、每图指标、比较与总报告。顶层 `run_failed.json` 表示未闭环的执行，已有局部产物不能当作完整结果。

## 4. P 干预的两种含义

首期 P 是边干预：预声明源 body ID，删除其全部出边，节点仍在完整图中。配对控制从非目标源按符号与整数权重精确匹配删除边；匹配不足失败，不放宽。受影响节点数量、解剖、路径及自环数量不保证匹配。

同一 `pair_id` 的 P_target 与 P_matched 必须同时登记。模式由图资产的预声明参数决定：

- `retrained`：每个干预图和输入种子独立训练完整九候选读出，用于检验恢复能力。
- `acute`：使用本次登记 G 的同输入种子读出，在严格相同输出顺序、时间和评分行条件下评估 P 特征；保存原训练来源与实际评估来源，不伪造“在 P 上训练”的模型。此专用路径拒绝 R/W。

目前没有执行真实动作数据的 P 实验，模式名称本身不是实验完成证据。

## 5. 已有的完整 W 接通记录

`data/malecns_v2/research_replay_W201_static/` 是一次实际完整 W 图运行：固定 2 秒预热加 0.1 秒静态观察，42/42 帧输入有效、40 帧预热、1 帧可预测，模拟 2.1 秒。网络保留 166,700 个节点和 25,582,938 条边；本次记录 9,274,281 次放电，墙钟 27.205 秒，进程峰值 RSS 1,762,639,872 bytes。

这证明独立研究入口实际运行了 W 的 LIF 网络。它没有动作标签，没有 G/W 同任务对比，也不证明改变权重有益或有害。
