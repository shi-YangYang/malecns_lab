# 离线诊断与研究工具

本手册保留 V2 冻结脑、可训练脑可观测性、拓扑对照和标记跟随评估的操作契约。它们是历史研究工具，不是 X2 主线前置门槛；完整脑实验仍按用户当前授权执行。阶段成绩统一见 [Spec007](../history.md#spec-007) 与 [Spec008](../history.md#spec-008)，不在手册重复验收流水。

命令从项目根目录执行，源码未 editable 安装时加 `PYTHONPATH=src`。每次登记/报告使用新输出路径，不覆盖失败现场。旧 run 与 checkpoint 绑定原源码/运行时；保持其 `sources` 快照，不改哈希冒充兼容。完整图计算串行、受资源监督，不调整 Noria/Docker，不连接设备。

<a id="model-v2-usage"></a>
## V2 完整脑重放

资产构建和因果/失效语义见[数据模型](data-model.md)。`diagnostics.replay_v2` 不读动作标签或旧 MotorDecoder，只记录显式视觉输入与完整图活动。

```bash
.venv/bin/python -m malecns_lab.diagnostics.replay_v2 \
  --stimulus static --repeat 2 --observation-seconds 0.1 \
  --run-seed 0 --budget-seconds 120 --out data/diagnostics_v2/static_new

.venv/bin/python -m malecns_lab.diagnostics.replay_v2 \
  --stimulus all --observation-seconds 0.5 --run-seed 0 --texture-seed 7 \
  --budget-seconds 180 --out data/diagnostics_v2/stimuli_new
```

生成刺激为静态、四方向平移及膨胀，每种独立重置、静态预热 2 s；默认 96×72，平移 10 px/s，膨胀 0.35 s⁻¹。它们属于同纹理族的 `generated_engineering_stimulus`，不能冒充真实动作录像。仿真时长与计算耗时分开，预算在窗口间检查，超限保留前缀而不裁图。

```bash
.venv/bin/python -m malecns_lab.diagnostics.replay_v2 \
  --recording logs/my_recording --family-id room_a_phone_sequence \
  --stimulus-kind object_motion --run-seed 0 --budget-seconds 600 \
  --out data/diagnostics_v2/recording_new

.venv/bin/python -m malecns_lab.diagnostics.replay_v2 \
  --video /absolute/path/clip.avi --scene-id room_a_scene_01 \
  --session-id capture_01 --family-id room_a_phone_sequence \
  --stimulus-kind object_motion --out data/diagnostics_v2/video_new
```

采集目录需 `session.json/video.avi/frames.csv`；普通视频显式提供场景/会话/素材族。同源裁剪、增强和种子仍属同族。视频仅接受名义 20 fps，时间为 `frame_index/20`，不重采样；host dequeue 只是诊断。刺激分类为 object_motion/camera_motion/mixed/unknown，不自动生成动作标签。

`--segments` 接受事前 JSON 数组，元素为 `{segment_id,start_frame,end_frame}`，范围左闭右开、不重叠；每段完整重置并预热 2 s。无效光流、扩张、图像或时钟终止当前段，不跳坏帧。整个视频的未登记帧、失败帧和未处理尾部分别计覆盖，不能事后按得分挑片段。

产物为 `run_manifest.json/run_summary.json` 与每段 `metadata.json/rows.npz/neuron_totals.npz`。关键数组：

| 字段 | 含义 |
|---|---|
| `frame_index/input_cutoff_s/target_time_s` | 源帧、可见输入时刻、后移 0.05 s 目标 |
| `valid/initialization/prediction_valid/target_available` | 输入、首帧、预热/历史完整、段内目标有效性 |
| `visual_features/input_rates_hz/input_event_counts` | 14 通道特征、每节点驱动 Hz、实际群事件总数 |
| `neural_counts/neural_rates_hz/neural_history_hz` | 33 个固定节点的当前计数、Hz 与 4 窗 132 维历史 |
| `visual_history/group_rates_hz` | 98 维视觉历史/EMA；群均值只作诊断 |
| `L_*/R_*/network_*` | 光流/纹理/扩张有效性，全脑放电、活跃及高频分布 |
| `simulation_start_ms/simulation_end_ms/processing_ms` | 神经时间与墙钟耗时 |

缺失浮点为 NaN、未计算计数为 -1，JSON 缺失为 null；必须用掩码，不能补零。首帧零运动初始化只供预热，前 40 窗和无段内目标的最后一帧不评分。

```bash
.venv/bin/python -m malecns_lab.diagnostics.report_v2 \
  data/diagnostics_v2/stimuli_new --out data/diagnostics_v2/stimuli_report_new
```

报告只读已存记录，不重算脑，输出 summary JSON/PNG；`--no-plot` 不依赖绘图。图中 n 是可预测窗口数，零覆盖为 null/灰色，不是动作成绩。

<a id="observability"></a>
## 完整图配对可观测性

`diagnostics.observability` 对比零背景和固定 L1–L5 8 mV，使用全部 R1–R6 与全部 DN，不训练或输出动作。模板命令只写协议，不载脑：

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.diagnostics.observability \
  --write-template data/observability_candidate.json
```

固定 CPU float32/4 线程、theta=0、rho=0；12×16 掩码，映射/相位种子 1701/1702，150 Hz、68.75 mV、dt=0.1 ms。条件为空白、居中、左、右、上、下、大块；小块 4 格，大块 16 格。每模式共享同一完整预处理状态与相位，等面积不保证事件数相同。

模板为预处理/刺激/撤除 100/50/50 ms，共 16,000 神经步；不能视为已证明本机可在预算内完成。事前可调三个 `windows_ms` 为 1–1000 整数，实际历史短协议为 60/20/10 ms。不得看响应后选窗、删条件或自动重试。默认 600 s、6 GiB、系统额外 1 GiB、0.1 s 采样；只能收紧预算。

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.diagnostics.observability \
  --graph data/malecns_v2/full_reference.npz --protocol data/observability_candidate.json \
  --out data/observability_new
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.diagnostics.observability \
  --validate-run data/observability_new
```

父进程登记来源/预算后才启动 worker；图只加载一次，两个模型依次释放。validate 仅读小产物，不导入 Torch/重算脑，核对身份、状态、事件账本、资源终态与汇总。进程 completed 不等于 protocol_complete；`response_observation` 单列，零差异也可为有效完整实验。

输出保留 registration、原 supervisor/resources、模型/映射身份、selections、完整基态、预处理、各条件 NPZ、参数不变证据与差异汇总。每 1 ms 桶记录复位后 DN v/g 平均及 int64 发放数，原始有符号差保留；电位容差 `atol=1e-5 mV, rtol=1e-6`，计数精确比较。容差不是生物显著性。直接掩码绕过识别，背景活动或 DN 差异不证明正确动作、拓扑优势或实时性。

<a id="learning-v2"></a>
## V2 掩码读出

`learning.train_v2` 读取真实视频重放，阶段 A 只训练线性读出。生成诊断刺激被拒绝；标签不从 RC、旧控制器或神经响应自动生成。协议先隔离 scene/session/family/source/video_sha256，同源材料不得跨 split。

协议 `cnsbridge.learning_protocol.v2` 至少含 protocol_id、phase、task、practical_delta、budget_seconds、bootstrap 与 segments。task 保存 policy_id、teacher_id、teacher_kind、description，以及 vx/wz 的 SI 容差和 stationary_tolerances；segments 含 path、labels、split，路径相对协议文件。

`pilot` 只训练/验证，不提取 test 预测；`confirmatory` 先封存身份再评 test。实用改善量 δ 是归一化误差绝对量，不是百分比，必须由任务与开发数据事前冻结。

标签格式示例（数值仅说明格式）：

```json
{
  "format": "cnsbridge.motion_labels.v2",
  "source_id": "复制对应 metadata.input_source.source_id",
  "timebase": "20Hz_video_playback",
  "policy_id": "independent-task-v1", "teacher_id": "annotator",
  "teacher_kind": "independent_human",
  "output": {
    "axes": ["vx", "vy", "vz", "wx", "wy", "wz"], "frame": "body_flu",
    "units": ["m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s"]
  },
  "intervals": [{"start_s": 2.0, "end_s": 3.0,
    "target": [0.1, null, null, null, null, 0.0],
    "mask": [true, false, false, false, false, true], "reason": "独立任务判据"}]
}
```

区间有序、不重叠、左闭右开，仅监督 vx/wz；未知轴/时间不补零。输入截止 2.05 s 对应 2.10 s 目标。预热、失败、末步无目标和未标区间不拟合，静默但有效神经活动仍保留。旧 CSV 仅在 segment 显式 `label_format=legacy_v1_intervals` 时读入，列必须为 `start_s,end_s,vx_mps,wz_rad_s,reason`，来源与教师仍由新协议给出。

```bash
.venv/bin/python -m malecns_lab.learning.train_v2 \
  --protocol /absolute/path/protocol.json --out logs/learning_prepare_new --prepare-only
.venv/bin/python -m malecns_lab.learning.train_v2 \
  --protocol /absolute/path/protocol.json --out logs/learning_fit_new
```

prepare-only 校验身份/标签/划分，不拟合。每图/输入种子独立训练，train 有效监督行拟合统计，零方差尺度=1；神经 132/视觉 98 维分别获得相同九个岭候选 `10^-4,…,10^4`，按验证场景归一化 MAE 选择，平局取第一个最小值。train/val 须有可评分 vx/wz，验证零覆盖则停止。

`MaskedRidgeV2` 可表示单轴支持，返回 supported_axes/prediction_valid；无效单步零占位不能执行，批量无效为 NaN。输出含 registration、dataset_manifest、coverage、逐种子模型/预测/row keys/fit_report 与 run_summary；失败部分产物不算完整结果。

评分按场景、轴、种子等权；至少两个独立簇且全部登记单元可评分才给 95% 簇 bootstrap 区间。`control_error-neural_error` 为正表示改善，按 δ 判断优势/未确认优势/不足，端点等于 δ 也为不足。区间条件于已登记图/种子，不覆盖所有随机图。`assert_paired_rows` 要求有序 row key、mask、有效性和目标完全一致，不取交集隐藏失败。

<a id="experiment-v2"></a>
## G/R/W/P 离线结构比较

G 为完整真实图；R 为同符号源的非自环双边交换；W 为每源非自环出边权重置换；P 为事前声明的通路边干预。生产默认加载器拒绝 research_control，只走独立离线入口。

R 将 `u→v,w→x` 交换成 `u→x,w→v`，权重随源保留；拒绝重复边、新自环或无效交换。保留节点、入/出度、逐目标正负入度、逐源出强度与总体有符号权重分布，**不保留逐目标入强度、解剖组织、类型连接或路径**。R/W 原始自环不动；W 非自环出边不足两条的源不变。交换数不证明混合，不可因差分数/静默/不可达删掉随机图。

```bash
.venv/bin/python -m malecns_lab.learning.research_replay_v2 \
  --parent data/malecns_v2/full_reference.npz \
  --control /absolute/path/research_control.npz --out logs/research_replay_new \
  --stimulus static --run-seed 0 --observation-seconds 0.1 --budget-seconds 180
```

录像可换 `--video` 或 `--recording` 并提供 scene/session/family/stimulus-kind；20 fps、2 s 预热、分段/失效约束与 V2 一致。实际派生图重建 LIF，`model_source_id` 区分图与适配器，`common_computation_id` 比较其余共同计算。

```bash
.venv/bin/python -m malecns_lab.learning.experiment_v2 template --out logs/protocol.template.json
.venv/bin/python -m malecns_lab.learning.experiment_v2 register \
  --protocol logs/protocol.json --out logs/registration.json
.venv/bin/python -m malecns_lab.learning.experiment_v2 validate --registration logs/registration.json
.venv/bin/python -m malecns_lab.learning.experiment_v2 execute \
  --registration logs/registration.json --out logs/experiment_new
```

模板 null/空表必须补齐：phase、parent、graphs（恰一个 G，条目 id/role/path）、input_seeds、task、practical_delta、总/单重放预算、bootstrap、comparisons、固定九候选 ridge_candidates；真实素材含身份、split、labels、video/recording 与预声明 segments。visual/mean/zero 是保留基线名，同一真实图来源不得重复登记成多个图。

register 绑定图/视频/会话/标签/代码哈希和图种子。execute 先 G 后对照，每图×输入种子独立提取/拟合，同输入事件、有效行、标签和预算。R/W 不借用 G 读出；均值只取 train，零基线固定。pilot 不执行测试源，confirmatory 在验证选择后才评分。现有图比较仍为描述性，不自动做多重校正或随机化检验。

任一重放失败、预算耗尽、零覆盖或配对不一致即 incomplete，不拟合成功子集。阶段边界检查总预算，超限保存 run_failed 与未执行任务，清空 comparisons；不缩 bootstrap 或删条件凑预算。

P 删除预声明源的全部出边；匹配干预从非目标源按符号/整数权重匹配删除边，匹配不足失败。同 pair_id 的 target/matched 必须一起登记。`retrained` 各自重训，`acute` 使用同次 G 的同种子读出，保留训练/评估双来源；acute 路径拒绝 R/W。

<a id="bounded-rewiring"></a>
## 有界 R 生成与准入

`learning.generate_rewiring` 固定 seed=1901，最多 25,582,837 次成功交换或 102,331,348 次提案。首两段各 100,000 提案，第二段必须新 worker 检查真实恢复；实际门槛通过后每段最多 1,000,000，不自动循环/重试。

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.generate_rewiring \
  --graph data/malecns_v2/full_reference.npz --out data/topology_new --proposals 100000
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.generate_rewiring \
  --run data/topology_new --proposals 100000
```

续段不能换图、seed、总预算、来源或 runtime。共享训练锁，6 GiB 请求、额外预留 1 GiB、600 s/0.1 s 采样；完整图在 worker 内。checkpoint 保存真实 CSC、PCG64（含缓存）、累计计数、父档与前缀，恢复前重验逐源权重多重集、自环/符号/度数及 RNG。

只有父 assessment 认可状态能续，失败段不贡献前缀；缺 assessment 不猜终态。到总边界才产生 candidate 并重载验证。机械向量包含边重叠、类型配对、逐节点强度和输入↔DN 最短跳数，`-1` 为不可达，不等于功能传播。正式入口只接受 `accepted_candidate(run_directory)` 的唯一父批准终局，裸 NPZ/中途档不算完成。

<a id="research-policy"></a>
## 正式 R 策略工厂

`research_admission.admit_research_graph(run)` 轻量核验批准链、来源/runtime、文件/机械向量摘要及唯一终局，不导入 Torch 或矩阵。返回审计快照；`revalidate_admission` 会重读原证据，改字典/自算摘要不能绕过准入。W/P、探针、中间/失败态及变化来源均拒绝。

`research_policy.from_terminal_research_run` 只在已登记监督 worker 内构造离线策略，不前向或训练。它用真实 G 初始化非图参数，释放 G 缓存后载入正式 R，重建/核验完整正向 CSR、伴随及只读共享 SciPy 缓存；权重按先 float32 后乘 weight_mv，失败不返回半安装对象，不接受已有动态状态来换图。

固定全部 L1–L5 8 mV、CPU float32、全部 3,377 R1–R6/1,314 DN，保留所有节点/边/自环/原注释。构造仍可能并存稀疏表示，不能凭释放顺序推断峰值。G 母本写 reference_*，R 自身写 research，raw_ordered_counts_sha256 必须来自真实 R。

`common_numeric_contract` 绑定共同动力学、背景、分组、空间/相位、相机/时钟、特征/动作和初始参数摘要；只接受初参，逐字段比较，不删除未知字段制造一致。共同契约相等不代表拓扑或行为相等。旧参数/训练存档与 full_reference loader 拒绝研究角色；不能拿旧训练 CLI 直接训练该对象。NeuralController 的类型检查不是研究权限隔离，调用方仍须保持离线。

<a id="follow-task"></a>
## 三轴标记环境与教师

协议 `cnsbridge.marked_follow.pilot.v1`，依赖 NumPy/SciPy，环境本身不载脑。20 cm 红色方标，默认距离 1.5 m，相机 160×120 BGR、水平视场 72°；真实投影四角，标记朝世界 x，不随相机转面。

```bash
.venv/bin/python -m malecns_lab.learning.follow_task --list-scenes
.venv/bin/python -m malecns_lab.learning.follow_task \
  --scene train-sine-combined --policy visual --out data/follow_visual_new
```

`--policy zero` 为零动作对照。世界初始前/左/上对应 x/y/z；FLU 仅 vx/vz/wz，限幅 ±0.3/±0.2/±0.5，其他轴非零报错。图像 x_error 正在右、y_error 正在上，分别以半幅归一化，所以右目标需负 yaw。默认控制 0.1 s，速度响应 tau=0.25 s；零指令不瞬停。

这是速度级简化环境，未标定电机/气动。z≤0.15、|x/y|≥5、z≥3 或距标记中心≤0.2 m 终止，无房间障碍几何，不能检验避障。

`FollowPolicyInput` 仅 frame/observation/camera/time；`truth_teacher(FollowTruth)` 仅供监督。`FollowEnvironment.frame/step/truth` 分开，`rollout(...,record_frames=...)` 可采像素；策略不能闭包读取真值。API 不是恶意 Python 沙箱。

检测选最大红色连通域，极小、裁切、同级多候选、缺损/形状不符为 invalid，误差/距离 None。confidence 是工程分数，不是概率或通用识别。无效时外部执行零并记 lost_target 原因，原模型动作仍保存，不能把保护算学会搜索。

14 场景按整场景划分：train=sine、flat/1.0、种子101–104；val=triangle、stripes/0.8、201–204；test=chirp/excursion、checker/0.65、301–306，含遮挡与出画返回。轨迹不依赖策略，测试不能选参。

每周期动作前评分：观察有效、横纵真值误差绝对值≤0.15、距离误差≤0.25 m 为合格；完整 80 帧、无物理失败且合格比例≥80% 才成功。提前失败分母仍计划 80，MAE 同报覆盖/失败，缺失不当零。输出 summary metadata/metrics 与 trajectory.jsonl，逐步保留 raw/executed/intervention、teacher、truth/next_truth、errors/qualified。CLI 默认不存图像。

<a id="evaluation-snn"></a>
## 完整图神经闭环评估

`learning.evaluate_snn` 让神经输出改变下一观察；前置是 joint 与独立 frozen_brain 各自完成同图/数据/初始来源/runtime 的 160 窗并具有有效父终态。不能选中途最低损失档或改 progress 冒充完成；冻结脑不得复用 joint 读出。

```bash
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.evaluate_snn register \
  --joint-run /path/to/completed_joint --frozen-run /path/to/completed_frozen \
  --out data/neural_evaluation_new
PYTHONPATH=src .venv-train/bin/python -m malecns_lab.learning.evaluate_snn run \
  --suite data/neural_evaluation_new --condition joint --scene val-triangle-forward --frames 1
```

登记不前向；首两帧必须分别新进程请求一帧，实际恢复检查后每次 1–40 帧，不超过剩余额度。五条件依序 joint/frozen_brain/acute_restore/direct_visual/zero；acute 保留 joint 读出，在场景初始化前恢复原 theta，参数值另行绑定。

每条件先 val 的 forward/yaw/vertical/combined；全部 val 终局后运行 `evaluate_snn summary --suite ...` 封存 validation_summary 才可 test。test 顺序 chirp-forward/yaw/vertical/combined/occluded、excursion-out-of-view，完整名称看 CLI 帮助，不能按测试结果调参。

episodes 保留全神经/感觉状态（无 Adam），支持奇数帧恢复；chunks 保存像素、DN、raw/executed、检测、介入、真值、误差与完整状态。检测无效仍推进脑，再由外层置零；非神经基线不伪造 DN。父进程只重放轻量环境并独立复算读出/评分，不重放脑。

`terminal` 只表示环境已终止；`evidence_complete` 还需该段父 completed 复核，后者才开放下一场景和正式均值。success 沿用完整 80 帧/≥0.8/无物理失败且证据完整，部分原始轨迹仅 diagnostic_metrics。神经时间、环境时间、墙钟分列。

资源/IO 失败只可显式恢复获准且环境未终止的封存前缀；若在终局封存后失败，保留 terminal 与局部评分但不升级完成、不重新前向放行。数值/协议错误、坏 seal、来源变化或重复/跳帧阻断续跑。阴性结果照实保留，不要求改善才能承认实验有效。
