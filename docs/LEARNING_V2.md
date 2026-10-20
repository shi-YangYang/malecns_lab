# V2 离线掩码读出工具

此工具读取[完整脑 V2 重放](MODEL_V2_USAGE.md)产物，只训练活动到期望动作的线性读出。当前用户尚未采集录像或动作标签；这里交付的是数据合同、训练和评估工具，没有真实动作准确率或连接组优势结论。标准生成刺激用于神经诊断，不能直接作为真实动作训练录像导入。

入口：

```bash
.venv/bin/python -m malecns_lab.learning.train_v2 --help
```

先录制真实视频、登记素材族，完成 V2 重放，再按独立任务规则标动作。标签不参与脑重放，也不能从旧 RC、实测速度或神经输出自动生成。视觉特征在脑外计算；读出学习不代表全脑端到端训练。

## 1. 先确定分区和任务

先按场景、采集会话和素材族分 train / val / test。不同输入种子、同素材裁剪和增强必须留在同一分区。导入器同时检查 `scene_id/session_id/family_id/source_id/video_sha256`：复制改名的相同视频也不能跨分区。不同裁剪的内容哈希不同，关联性仍必须由人用同一 `family_id` 声明。共享会话/素材族的场景在置信区间中视作同一簇。

从每段重放的 `metadata.json.input_source` 复制精确的来源身份；不得凭文件名猜。每个目录必须保留父级 `run_manifest.json`、本段 `metadata.json` 和 `rows.npz`。同一条件必须使用一致的模型来源与有序特征：神经 132 维，视觉 98 维。不同图条件应建立独立学习协议，分别训练。

登记协议示例 `protocol.json`（路径相对此文件；数值仅展示格式，**不是实测推荐值**，必须先按任务允许误差替换和冻结）：

```json
{
  "format": "cnsbridge.learning_protocol.v2",
  "protocol_id": "my-task-pilot-v1",
  "phase": "pilot",
  "task": {
    "policy_id": "my-independent-action-policy-v1",
    "teacher_id": "annotator-yang",
    "teacher_kind": "independent_human",
    "description": "在标注前写明任务目标、期望动作判据和排除规则",
    "tolerances": {"vx": 0.2, "wz": 0.3},
    "stationary_tolerances": {"vx": 0.01, "wz": 0.01}
  },
  "practical_delta": 0.1,
  "budget_seconds": 300,
  "bootstrap": {"seed": 123, "replicates": 2000},
  "segments": [
    {"path": "replay-train/source_00_repeat_00_segment_000", "labels": "labels-train.json", "split": "train"},
    {"path": "replay-val/source_00_repeat_00_segment_000", "labels": "labels-val.json", "split": "val"},
    {"path": "replay-test/source_00_repeat_00_segment_000", "labels": "labels-test.json", "split": "test"}
  ]
}
```

`tolerances.vx` 单位 m/s，`tolerances.wz` 单位 rad/s，用于归一化主误差；`stationary_tolerances` 定义静止和方向辅助指标的判据。`practical_delta` 是归一化误差的最小实用改善量，不是百分比。二者必须来自任务约束与训练/验证 pilot，不能由测试表现反推。

`pilot` 只拟合训练集、用验证集选参数并报告开发指标，不计算 test 预测或指标。`confirmatory` 在相同训练/验证过程后评估 test；启动时先保存登记快照和哈希。快照提供追踪，不自动证明人从未查看过测试集。如果看过测试结果再改协议，该测试集应降为开发数据，另收独立测试数据。

## 2. 标注期望动作

`labels-train.json` 示例：

```json
{
  "format": "cnsbridge.motion_labels.v2",
  "source_id": "从对应 metadata.json 的 input_source.source_id 原样复制",
  "timebase": "20Hz_video_playback",
  "policy_id": "my-independent-action-policy-v1",
  "teacher_id": "annotator-yang",
  "teacher_kind": "independent_human",
  "output": {
    "axes": ["vx", "vy", "vz", "wx", "wy", "wz"],
    "frame": "body_flu",
    "units": ["m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s"]
  },
  "intervals": [
    {"start_s": 2.0, "end_s": 3.0, "target": [0.1, null, null, null, null, 0.0],
     "mask": [true, false, false, false, false, true], "reason": "按事先定义的任务规则，此区间期望低速前进且不转向"},
    {"start_s": 3.0, "end_s": 4.0, "target": [0.0, null, null, null, null, 0.2],
     "mask": [true, false, false, false, false, true], "reason": "按同一规则，此区间期望停止前进并左转"}
  ]
}
```

示例动作也仅展示格式，不应直接复制成实际标签。采用 FLU：前、左、上；正 `wz` 左转。这里的速度是期望目标，不能声称无人机已实现该速度。第一期仅支持 `vx/wz` 监督，其他四轴零占位不训练、不评分。尚不确定的轴用 `mask=false`，未知区间不写标签；不要用 0 代替未知。标签区间必须排序、互不重叠，采用 `[start_s,end_s)`。

导入器使用 `target_time_s=input_cutoff_s+0.05` 查标签。例如输入截止 2.05 s 的特征匹配 **2.10 s** 的动作，不能按当前输入时间错位。最初 2 s 预热、无效输入、末步无目标以及未标区间不进入拟合或评分。失败后保留有效前缀，未处理尾段单列；不填补假零。全静默但有效的神经活动保留为正常观测。

旧 `labels.csv` 仅在该 segment 条目显式设置 `"label_format":"legacy_v1_intervals"` 时可导入，列必须为 `start_s,end_s,vx_mps,wz_rad_s,reason`。独立 `policy_id/teacher_id/teacher_kind`、素材身份与 split 仍由新协议提供。这只是两轴区间标签导入，不把旧六轴零占位变成全轴监督，也不自动认可旧标签的教师来源。

## 3. 先校验，再训练

```bash
.venv/bin/python -m malecns_lab.learning.train_v2 \
  --protocol /absolute/path/protocol.json \
  --out logs/learning_v2_prepare --prepare-only

.venv/bin/python -m malecns_lab.learning.train_v2 \
  --protocol /absolute/path/protocol.json \
  --out logs/learning_v2_fit
```

每次使用新输出目录，模型和结果拒绝覆盖。`--prepare-only` 校验来源、时间、标签和分区，输出数据清单与覆盖，不拟合。每一个图和输入种子单独训练；同一图协议可以包含多个 seed，工具会分别输出 `input_seed_<seed>/`，不会把它们混为一个独立场景或共同拟合。

阶段 A 只学习标准化统计、权重和截距。标准化仅使用有监督且有效的 train 行；各轴损失仅包含该轴 mask=true 的行；零方差特征尺度为 1。λ 固定候选 `10^-4,…,10^4`，神经与视觉各得到相同九个候选，用验证场景的归一化 MAE 选择，平局取第一个最小值。测试数据不参与拟合或选择。不加入固定巡航、逃逸、动作裁剪或 RC。

主训练入口要求 train 和 val 均有可评分的 `vx/wz` 数据；验证场景有零覆盖会明确停止，保留覆盖记录，不能丢掉失败场景继续选参。底层 `MaskedRidgeV2` 可表示只监督一个轴的模型，返回 `supported_axes` 与 `prediction_valid`；未训练轴保持无权重占位。无效单步返回零占位及 `prediction_valid=false`，调用方不能把它当有效动作。批量无效预测保存 NaN。

## 4. 读取结果

输出包括：

- `registration.json`：实验登记与协议哈希。
- `dataset_manifest.json`、`coverage.json`：全部来源、标签、模型和重放哈希、分区、段覆盖与未处理尾段。
- `input_seed_*/model_neural.json`、`model_visual.json`：独立训练的 132/98 维读出、训练统计、轴支持及模式身份。
- `input_seed_*/predictions.npz`、`rows.json`：对应 row key、目标、监督 mask、有效性和神经/视觉/训练均值/零四种预测；pilot 不导出 test 预测。
- `input_seed_*/fit_report.json`：九个验证候选、选中 λ、训练身份、逐场景指标、常量基线、配对差异及簇区间。
- `run_summary.json`：实际耗时、进程峰值内存、代码哈希、输入种子和证据范围；错误或预算耗尽写 `run_failed.json`，部分产物不代表完整结果。

误差先在每场景/轴/输入种子内计算 MAE，再按种子、归一化轴和场景等权汇总。零可评分场景不会消失；报告保留它及失败、预热、缺标签和未处理尾段。至少两个独立簇且全部登记场景/轴/种子可评分时才生成 95% 簇 bootstrap 区间，否则为 `insufficient_evidence`。即使可评分，低时间覆盖也应结合任务要求解释，不能泛化为完整场景可用。

配对差异为 `control_error - neural_error`，正值表示当前标签任务上神经读出误差较低。区间条件于登记的图和输入种子，不是 p 值，不涵盖随机图总体的不确定性。图间变异须由 P3 实验单独报告；开发集上的区间不构成确认性结论。仅当独立测试协议、覆盖和完整研究对照都满足时，才可讨论真实连接结构是否具有实用贡献。

有可用区间时，解释以登记的实用改善量 `δ=practical_delta` 为界：下界严格高于 δ 为 `practical_advantage_on_scored_data`；上界严格低于 δ 为 `no_confirmed_practical_advantage`，表示当前条件下不支持达到该实用量，不表示等效；跨越或接触 δ（包括任一端点等于 δ）均为 `insufficient_evidence`。覆盖或独立簇不足时仍保持 `insufficient_evidence`，不生成区间。

P3 集成接口：`for_input_seed()` → `fit_readouts()` 为每个条件独立拟合；`assert_paired_rows()` 要求相同有序 row key、有效性、mask 和目标，拒绝静默求交集。`score_predictions()` 保存评分清单哈希，`paired_scene_comparison()` 在逐行评分身份和覆盖完全一致时比较场景差异。缺少或失败条件应保留失败，不按得分换 seed。

这些工具没有接入设备控制；线性解码结果不代表自主避障、真实飞行、果蝇认知或跨设备迁移。
