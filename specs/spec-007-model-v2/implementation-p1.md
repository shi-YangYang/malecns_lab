# Implementation Report — P1 完整图输入与神经诊断

## Summary

完成已通过独立验收的 V2 纯视觉接口与完整参考脑接入。模型只接收图像，使用显式群组广播与共同外部事件；全部 166,700 节点和 25,582,938 条连接参与传播，33 个细胞仅是输出探针。记录神经与直接视觉的因果历史、输入质量、时间及失败覆盖，不读动作标签、不使用旧 Encoder/MotorDecoder 或 RC。

真实完整图短探针与六种标准刺激已经实际运行，记录见下文。它证明这套离线链路可以运行并对输入产生不同神经响应，不证明动作正确、连接组优势或实时飞行能力。

## Files Changed

- `brain/input_events.py`：基于 run seed、body ID、片段内仿真步的无状态共同 Bernoulli 事件。
- `brain/lif.py`：`step(external_events=...)` 显式事件注入；替代该步旧随机率路径，参数省略时保持 V1 行为。事件在修改状态前检查。
- `senses/encoding_v2.py`：14 个视觉群统一每节点率；无索引到图像位置映射，HAL 不提供外部驱动。
- `diagnostics/history_v2.py`：33×4=132 维神经历史，14×4+14×3=98 维视觉历史/EMA，前 40 窗预热。
- `diagnostics/runtime_v2.py`：严格完整参考加载、固定 LIF 和 8 mV DNg02 偏置、全脑传播、失效与完整重置、模型来源身份。
- `diagnostics/sources_v2.py`：名义 20 fps 的采集目录/普通视频，显式场景族，预声明片段和六种工程刺激；不读取 labels.csv。
- `diagnostics/replay_storage_v2.py`：V2 逐帧 NPZ、缺失掩码、片段元数据、全脑节点累计计数、失败前缀与尾段/整源覆盖。
- `diagnostics/replay_v2.py`：离线命令、预算与独立新目录；默认脑路径从项目根定位。
- `diagnostics/report_v2.py`：通用已保存 run 的 JSON/热图报告，不运行脑；支持普通录像、重复段和零有效段。
- `tests/test_replay_v2.py`：事件不变性、注入边界、因果历史、群组广播、失败不推进、重置确定性、覆盖与分段的合同测试。
- `tests/test_report_v2.py`：单录像零可预测窗口仍保留覆盖/缺失语义，以及来源不一致拒绝。
- `docs/MODEL_V2_USAGE.md`：可直接执行的构建、标准刺激、录像重放、分段及产物说明。

P0 业务代码和已独立通过的 `vision_v2.py` 未在本批修改。数据目录新增诊断资产及一次性可复现汇总脚本，均按项目规则不提交 Git。

## Important Decisions

### 固定输入与动力学

每眼 brightness/ftb/btf/up/down/loom/loom_speed 共 7 通道，分别广播到 R16、T4/T5a–d、LPLC2、LC4。每节点最大率依次为 40/120/120/120/120/200/150 Hz。同群所有细胞收到同率；记录群人数、每节点率和实际事件总数。HAL 虽有完整身份和神经诊断，主实验无外部输入；不把缺失遥测称为测得零。

LIF 值按设计显式构造，不由旧配置合并覆盖：rest/reset -52 mV、threshold -45 mV、tau_m 20 ms、tau_syn 5 ms、t_ref 2.2 ms、delay 1.8 ms、w_syn 0.275、poisson_factor 250、dt 0.5 ms、noise 0。量化后的 delay/refractory 均为 2 ms；DNg02 左右分别加 8 mV 工程偏置。

这些动力学借鉴既有简化文献模型，离散步长、外部编码率和偏置是已登记建模/工程假设，不是本批优化得出的生物数值。全脑使用 float32 数值积分，P0 原始突触整数资产仍保留不变。

### 共同事件与重置

对 seed/body ID/step 使用不同域常数的 SplitMix64 混合，取高 53 位均匀数，与 `rate_hz * dt_ms / 1000` 比较。每步每节点最多一次事件，明确是 Bernoulli 步进近似，不声称精确连续时间 Poisson。

事件不依赖节点遍历顺序、其他群是否禁用、内部发放或图的传播结果。每个显式片段从仿真步 0 重新开始相同种子的流；run seed 单独记录，不混入固定模型来源身份。

重置包括视觉前图/时间/膨胀、神经 v/g/延迟缓冲/不应期/步号/统计/RNG、输入率、全脑累计计数、神经/视觉历史及 EMA。固定偏置保留相同值。视觉失败在生成率和事件前返回，不推进任何神经或历史状态；失败后锁停至显式新片段。

### 因果时间与产物

源帧 k 只有当前及过去输入，播放时刻 `k*0.05`；当前图驱动下一个 50 ms 仿真窗，目标时刻 `k*0.05+0.05`。33 个输出 body ID 数值升序，四窗最旧到最新。视觉对照使用相同有效前缀、重置、预热和四窗，另加 tau 0.05/0.2/1 s 的零初态 EMA。第一帧和前 40 窗不作为预测样本；最后一帧目标越过片段末端也不可评分。

新格式为 `cnsbridge.neural_replay_run.v2` / `cnsbridge.neural_activity.v2`。`rows.npz` 关键读取契约固定为：`valid`、`prediction_valid`、`target_available`、`input_cutoff_s`、`target_time_s`、`neural_history_hz (T,132)`、`visual_history (T,98)`、`neural_counts/rates_hz (T,33)`、`visual_features/input_rates_hz/input_event_counts (T,14)`，另有质量/时间/群组诊断。无效浮点为 NaN、未计算计数为 -1；JSON 使用 null，不写非标准数字。后续监督必须独立读取标签，按 target_time_s 对齐并使用 prediction_valid。

模型身份覆盖完整图来源、节点/群组、视觉配置、事件算法、固定动力学/偏置、实际量化值、时间/特征定义及相关源码哈希。输入视频/生成器与 scene/session/family、run seed、预声明片段、预算另存。每次输出拒绝已有目录。

## Validation

- Risk Level：**S3**；具体依据为 LIF 外部事件扩展及跨视觉/神经/持久化的因果状态契约。只运行相关检查。
- `.venv/bin/python -m pytest tests/test_replay_v2.py tests/test_lif.py -q`：**13 passed in 0.12 s**。未重复已通过的纯视觉测试或全项目测试。
- 实际运行之后增补整源覆盖汇总，防止显式片段外的帧被隐藏；只运行新增相关测试 `test_whole_source_coverage_includes_predeclared_gaps_and_zero_valid_scenes`：**1 passed in 0.02 s**。该补充和 CLI 默认路径定位不改变任何神经计算代码；没有因此重复全脑实验。
- 报告工具定向检查 `.venv/bin/python -m pytest tests/test_report_v2.py -q`：**1 passed in 0.02 s**。随后用既有真实完整图记录运行报告命令一次，成功生成六段汇总与图，不重跑脑。图中文字明确 eligible prediction windows，不称动作得分；普通录像标签调整仅文案，无重复测试。
- 读取差异与针对本批文件的 diff 格式检查。未安装 Ruff，未把未执行 lint 写成通过。
- 无设备、飞行、动作训练、B/C 参数优化或 commit。

### 真实完整图短探针

```sh
.venv/bin/python -m cnsbridge.diagnostics.replay_v2 \
  --stimulus static --repeat 2 --observation-seconds 0.1 \
  --budget-seconds 120 --run-seed 0 \
  --out data/diagnostics_v2/pilot_static_20261016
```

两段各 42 帧、2.1 s 仿真，全部输入有效；各 40 帧预热和 1 个可预测窗口。每段计算约 12.2 s，完整命令墙钟 26.713 s。显式 reset 后，输入事件、神经计数/率/历史、视觉特征/历史、群组率、有效性和仿真时间等 10 类数组逐元素一致。处理耗时不要求一致。证据：`pilot_static_20261016/determinism_check.json`。

### 六种真实完整图工程刺激

```sh
.venv/bin/python -m cnsbridge.diagnostics.replay_v2 \
  --stimulus all --observation-seconds 0.5 --run-seed 0 \
  --budget-seconds 180 \
  --out data/diagnostics_v2/standard_stimuli_20261016
```

每种独立 2 s 预热 + 0.5 s 观察。300/300 帧输入有效，0 失败、0 未处理；每段 9 个可预测窗口。共 15 s 仿真，实测 96.830 s 墙钟，峰值 RSS 1,102,233,600 bytes，约 1.03 GiB。单帧处理 p50 为 303.7–324.4 ms；当前运行明显慢于 50 ms 观察间隔，不能作为实时闭环依据。

下表是每种刺激 9 个可预测窗口的描述均值，单位 Hz：

| 刺激 | DNg02_L | DNg02_R | DNp03_L | DNp03_R | DNp01_L | DNp01_R | 相对静态均值变化的输出细胞数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 静态 | 18.222 | 7.143 | 0 | 0 | 0 | 0 | 0 |
| 左移 | 13.333 | 6.984 | 0 | 0 | 0 | 0 | 11 |
| 右移 | 14.222 | 1.746 | 0 | 0 | 0 | 0 | 12 |
| 上移 | 29.481 | 26.349 | 0 | 0 | 0 | 0 | 12 |
| 下移 | 3.556 | 1.270 | 0 | 0 | 0 | 0 | 13 |
| 膨胀 | 18.519 | 9.048 | 15.556 | 0 | 6.667 | 0 | 12 |

左右差异、静默群组和短窗口均如实保留。没有依据把上移条件的 DNg02 增加解释成“无人机应该上升”，也没有输出人工 throttle/yaw 来制造行为。

- 模型来源 ID：`b8eee9b918fae00b904e56cf920f0c55a078f9f944615c19e0cab43a49bda7c0`。
- 原始运行：`run_manifest.json`、`run_summary.json` 和每段 `rows.npz`、`metadata.json`、`neuron_totals.npz`。
- 汇总证据：`standard_stimuli_20261016/diagnostic_summary.json`，包含 14 通道、33 个输出、6 组均值、全脑静默/高频率、整源覆盖和处理延迟。
- 图：`standard_stimuli_20261016/diagnostic_summary.png`，使用 Matplotlib 绘制并已视觉检查；描述选中的有效预测时刻，无动作分数。
- 可复现汇总脚本：`data/diagnostics_v2/summarize_p1.py`。它只读取保存记录，未重新运行脑。

上述一次性分析已收敛为通用业务命令：

```sh
.venv/bin/python -m cnsbridge.diagnostics.report_v2 \
  data/diagnostics_v2/standard_stimuli_20261016 \
  --out data/diagnostics_v2/standard_stimuli_report_20261016
```

正式通用报告的 `diagnostic_summary.json/png` 位于 `standard_stimuli_report_20261016/`；后续用户可对自己的 run 使用同一入口，无需依赖忽略目录中的分析脚本。报告按每段真实清单工作，没有硬编码六种刺激；来源不同则拒绝拼接，零可预测窗口用 null/灰色而不是虚构零响应。

## Known Limitations

标准刺激均属于一个工程纹理族和单一输入种子，不是独立真实场景；观察期短，群组均值不构成统计优势或生物行为验证。首帧、预热、失效和末帧排除已单独记录，不将 300 帧全部当作可训练样本。

当前无真实录像或独立动作标签，因此没有训练新解码器、动作误差或连接组对照结论。固定相机物体运动与设备自运动仍是不同输入分布。P3 研究图不通过本入口；它只接受 P0 完整真实参考。

本批不改变视觉模块已报告的局部 LK 限制：强照明变化、遮挡、大位移不保证正确。当前完整图吞吐不足实时，后续优化必须保持完整图并单独实测，不能默认恢复阈值 3 或用旧人工动作掩盖延迟。

## Remaining Questions

本批无用户决策阻塞。交新的独立验收 Agent 审核；实施者不填写验收 PASS。后续 P2 可读取冻结的 V2 逐帧模式，实际监督数据仍等待用户采集。
