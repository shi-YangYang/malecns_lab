# Task Handoff — spec-008-trainable-snn / Implementation

## 当前任务：按用户要求预留 1 GiB 并重新测量（2026-10-19）

用户明确要求“保留到1gb”。[Decision 012](../decisions/012-one-gib-system-reserve.md)和[资源调整子规格](../../specs/spec-008-trainable-snn/resource-reserve-1g.md)授权将共享监督、P1及配对诊断新协议默认预留改为1GiB；保持6GiB进程上限、600秒、持续监控及完整连接组。实施与独立验收通过后提交，再由协调者用新目录运行一次P1探针，实际结果另报。旧4GiB登记/失败证据不覆盖，Noria限制继续有效。下面blocked及4GiB限制属于此前阶段记录，不覆盖本次用户授权。

软件调整现已完成，41项定向检查通过并获新独立验收者[PASS](../../specs/spec-008-trainable-snn/acceptance-reserve-1g.md)；两份报告均已读取处理，没有未解决返工项。下一步按新规则实际测量，不以软件通过计作全图学习通过。

## 当前资源限制：Noria 必须保持运行（2026-10-19）

用户要求检查资源并清理无用内容后，进程审计未发现遗留训练或测试进程，发现 Docker 的四个 Noria 容器仍在运行。用户明确拒绝停机：“不可以，Noria 还在使用”。按[Decision 011](../decisions/011-preserve-noria-services.md)，禁止为本项目释放内存而停止、暂停、删除 Noria 或其 Docker / 虚拟机。没有执行此类清理；完整图训练仍未完成，目标当前为 blocked。后续依据其他可用资源变化继续，不再将 Noria 作为待批准的清理对象。

## 最新进展：P1 实际加载触及 RSS 上限（2026-10-18）

用户重新要求“继续”，无遗留模型进程，内存已高于4GiB门槛，因此按原协议只运行一次新登记 `data/trainable_probe_20261018_p1_r2/`。启动前 available=5,190,090,752 bytes；扣除4GiB预留后 RSS 上限为895,123,456 bytes。约2.7075秒后在 `load_graph` 阶段因 `rss_limit` 停止，采样峰值918,536,192 bytes。没有进入模型构造、前向或更新，P1/AC2仍未完成。原始记录与静态内存分析见[P1第二次登记报告](../../specs/spec-008-trainable-snn/p1-run-20261018-r2.md)。

本轮无业务代码修改，不重试、不裁图、不降低预留。仅优化loader无法使当前未改动的模型构造适配约0.83GiB上限；更大预算下的完整训练成本仍待实测。用户回复“可以，释放后告诉你”，等待通知后重新读取资源并使用新目录继续。旧失败资产保持不变。P2d工具已在 `2f7b7db` 提交，实际配对实验仍未执行；下面此前内存测量与通知状态为历史记录，不代表本次运行结果。

本次失败记录已通过新独立验收者的[S0文档验收](../../specs/spec-008-trainable-snn/acceptance-p1-run-r2.md)，报告已读取处理；PASS仅确认记录准确，数值运行仍FAIL。内存分析者也已返回报告，本轮无未处理子任务。

## 活动目标：完整路线实施与逐步提交（2026-10-17）

用户已授权完成全部路线、删除无用内容、每步 commit。活动 goal 不因 P0/P1 或文档通过而结束，参见 [Decision 009](../decisions/009-fullgraph-training-execution.md)。此前“本轮只规划/不训练/不提交”的记录仅指历史阶段。

- 前置基线已复核并提交 `69f0711`：完整数据、V2 离线研究工具与规划。没有重复已通过的旧诊断。
- [P0 核心](../../specs/spec-008-trainable-snn/p0-core.md) 已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p0.md)并提交 `44aa5ac`，19项定向检查；尚无全图更新。独立 `.venv-train` 已安装 torch 2.14.1、NumPy 2.5.4、SciPy 1.18.1、pytest 9.1.1、psutil 7.2.2 等。旧 `.venv` 保留。
- [P2a 环境](../../specs/spec-008-trainable-snn/p2-environment.md) 已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2a.md)并提交 `365fab0`；三轴环境、像素检测、分区与日志齐全。已有训练场景视觉基线合格率75% / 零动作0%，二者任务成功均为false；不是 SNN 结果。
- [P1a 工具](../../specs/spec-008-trainable-snn/p1-tools.md)经返工后独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p1-tools-r1.md)并提交 `0d6e18a`，16项初始检查和30项返工检查；最终通过须验证完整产物，不只信任退出码。首次实际登记 `data/trainable_probe_20261017_p1/` 的 available=3,674,931,200 bytes，低于预留4GiB，明确未启动模型。用户回答会先释放内存、完成后通知；尚未收到完成通知，不自动重试。P1实际更新/AC2仍未完成。
- [V1 清理](../../specs/spec-008-trainable-snn/cleanup-v1.md) 及20项定向检查通过；两处文档问题经新实施者返工、新验收者确认 PASS，已提交 `4e4a67f`。V2 已替代该离线流程，旧设备链仍保留到新链完成。
- [旧仿真器清理](../../specs/spec-008-trainable-snn/cleanup-sim.md)已独立PASS并提交 `31b9761`；6项相关检查，感知回归改用明确的测试内针孔纹理刺激，未改Retina或新环境。
- [P2b 映射](../../specs/spec-008-trainable-snn/p2b-mapping.md)已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2b.md)并提交 `4eada75`，29项定向检查。完整R1–R6输入、全部descending_neuron输出、元数据身份校验及工程空间事件编码已实现；未加载完整图或训练动作，见 NEURAL_INTERFACE 文档。
- 新发现：3,377个R1–R6全部histamine，当前P0只驱动它们、其余细胞静息无背景时无法产生第一批非输入脉冲。此为方程与注释的静态推论，不是全图实测。P1对L1膜电位的优化仍有效，但不能据此推断多跳输出已接通。细节与文献边界见[可观测性记录](../../specs/spec-008-trainable-snn/signal-observability.md)。[Decision 010](../decisions/010-lamina-tonic-variant.md)已单列可选L1–L5背景假设，不改变原P0/P1无背景协议。
- [P2c 数值变体](../../specs/spec-008-trainable-snn/p2c-tonic.md)已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2c.md)并提交 `d9d79a5`，53项定向检查通过。仅精确L1–L5的有界平衡点偏移，fixed/learnable模式独立身份，旧checkpoint明确拒绝变体以免遗漏rho；没有完整图传播或任务收益证据。独立设计审查与实施/验收报告均已读取处理。该轮开始时可用内存约3.0GiB，未收到用户释放完成通知，没有重试P1；数值子批完成不代表整体目标完成。
- 2026-10-18的[P2d配对可观测性工具](../../specs/spec-008-trainable-snn/p2d-observability-tools.md)已独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2d.md)，47项不同定向检查有通过记录。实现预登记、完整状态/相位分支、全部DN与完整节点计数、增量身份/hash保存及独立证据校验；实际[候选实验协议](../../specs/spec-008-trainable-snn/observability-plan.md)仍为Draft，窗口/预算将在P1实测后冻结。没有加载完整图或运行诊断，没有收到内存释放完成通知；实施与新验收报告均已读取处理。
- 本轮再次只读复核：available=2.8643035888671875GiB、swap_used=7.70867919921875GiB，没有运行中的P1/配对诊断进程。仍低于4GiB系统预留，实际运行连续受相同条件阻塞；软件准备不能代替完整图证据。后续窗口/工作点/训练方案依赖P1成本和配对响应实测，不继续凭假设扩展训练实现。保留旧失败登记，等待用户内存释放完成通知后从新的输出目录继续。
- 各实施/验收报告已读取处理，root按阶段提交。尚未完成全图内部更新、三轴学习、对照研究或实飞；整体目标未完成，不以工具子批通过结束完整路线。

用户通知内存已释放后，先按现有P1协议在新的输出目录登记实际完整图更新，600s/6GiB上限与4GiB系统预留保持。遇到具体瓶颈保留原结果再定向解决；不重跑旧检查。随后完成视觉到下降神经元的可观测性诊断，必要的背景工作状态/分级电位假设单列数值契约与对照，再进入任务训练、内部干预/重连比较、部署时延及Tello实验。无实测不得写成完成；硬件动作需当前现场状态，自动化检查不触发飞行。

## 最新进展：Mac 执行路线（2026-10-17）

用户在查新后确认按完整 MaleCNS 内部训练方向推进，并询问结合当前项目和 Mac 的具体下一步。本轮只读核对代码、已有运行记录及本机环境，新增 [Mac 执行路线](../../specs/spec-008-trainable-snn/mac-execution-plan.md)。M5 / 10 核 / 16 GiB；当前 `.venv` Python 3.14.6，模块发现查询未找到 torch/mlx/jax。未安装、未运行训练、未重跑旧诊断、未操作硬件、未提交。

最小下一批：先完成 LIF 独立数值契约与稀疏反向算子检查，再用完整图完成一次内部共享增益更新并记录资源和干预证据。优先候选为按突触前类型共享有界正增益，固定基图乘完整状态；不能把旧事件索引筛选直接当作正确代理梯度。建议采用独立训练环境，先 CPU、单序列短窗；真实训练速度和 MPS 支持待测。目标标记、跟随容差与完整任务实现仍为后续阶段，整个 Spec 保持 Planning。

本轮文档风险 S0；独立审查 [PASS](../../specs/spec-008-trainable-snn/acceptance-mac-plan.md)，仅确认规划与事实一致，不覆盖前次规划验收，也不代表后端可用或训练成功。报告已读取处理，本轮停止；后续按路线细化实施子批。

## 当前任务（2026-10-17）

用户明确要求 MaleCNS 网络自身学习控制，而不只训练解码器；已确认第一项任务为指定目标跟随，学习前进、升降与转向，当前只有这台 Mac。本机只读查询为 Apple M5 / 10 个物理 CPU 核 / 16 GiB，未进行训练性能测量。

本轮整理 [Decision 008](../decisions/008-trainable-malecns-control.md)、[Spec 008](../../specs/spec-008-trainable-snn/spec.md)及 Plan，并更新使命和路线；只涉及文档，S0。新 Spec 为 Draft / Planning，未实现后端、未训练、未接设备、未付费或提交。此前冻结脑＋岭回归保留为对照，不能继续作为用户新目标的终点。

独立文档审查 [PASS](../../specs/spec-008-trainable-snn/acceptance.md)，仅确认规划一致性；AC1–AC6 未完成。本轮未运行测试或仿真。

下一步先把独立 LIF 数值语义和完整图内部参数更新的资源探针细化到可实施范围；目标指定、空间感觉映射、训练后端、资源门槛与任务容差按新 Spec 的待确认项处理。不得重跑此前已通过的诊断来代替新的问题；不得因 Mac 资源有限暗中删图。用户已经回答的目标轴与算力不再重复询问。

## 前一任务记录：spec-007-model-v2

以下保留已完成工具批次的范围、证据和停止记录，不代表新 SNN 训练已完成。

## Role / Scope

EXISTING。用户授权“开始做，自由决策”，并确认“还没录，先完成诊断和工具”。本次范围为 [执行计划](../../specs/spec-007-model-v2/execution-plan.md) 的 P0–P3 离线工具、完整图资产和标准刺激诊断；M4 真实动作确认研究待采集后继续，不制造标签或研究结果。保留前序审计/设计未提交文件、原始数据和历史 V1 资产；没有 commit/push 授权。

## Evidence

- P0：全部 11 份官方 v1.0 flat-connectome 数据为母本，按作者有出处的 neuron 定义核对 body ID。完整参考图为 166,700 节点、25,582,938 条内部有向边、124,177,617 突触计数，保留弱边、101 条原始自环、217 个内部孤立节点和缺失注释；片段边界另有逐行/逐 ID 账本。资产 `data/malecns_v2/full_reference.npz` 不覆盖旧图。见 [Decision 007](../decisions/007-full-dataset-baseline.md)、[P0 报告](../../specs/spec-007-model-v2/implementation-p0.md)及[角色校验返工 PASS](../../specs/spec-007-model-v2/acceptance-p0-r1.md)。
- P1：完整图传播、共同输入事件、132/98 维因果特征、失败/重置/覆盖及信号报告已 [PASS](../../specs/spec-007-model-v2/acceptance-p1.md)。六类标准刺激各 50 帧全部有效；合计 15 s 仿真耗时 96.83 s，峰值 RSS 1,102,233,600 bytes。活动有刺激差异、左右不对称和静默，不代表动作正确。操作和实测见 [MODEL_V2_USAGE](../../docs/MODEL_V2_USAGE.md)。
- P2：掩码训练、分区隔离、每图每输入种子独立拟合、val 选参、pilot 不算 test、同评分行和覆盖统计已通过。[原验收](../../specs/spec-007-model-v2/acceptance-p2.md)的 CI 三态问题已关闭，见 [P2-R1 PASS](../../specs/spec-007-model-v2/acceptance-p2-r1.md)。
- P3 图工具：R/W/P 机械不变量与离线资产隔离已 [PASS](../../specs/spec-007-model-v2/acceptance-graph-controls.md)。完整 W 实际重放 42 帧、2.1 s 仿真，耗时 27.205 s，峰值 RSS 1,762,639,872 bytes；无动作训练或 G/W 效果比较。

## Current Status

当前 P0–P3 工具与完整图诊断批次已完成，M1–M3 已勾选。P3 登记/执行集成原验收的总预算终态问题已由新实施/验收者关闭，见 [P3-R1 PASS](../../specs/spec-007-model-v2/acceptance-p3-r1.md)。所有子批报告已读取处理，当前无未解决返工项；M4 保持未完成。不得重复前序检查或完整脑运行。

## Entry Points / Assets

- [完整脑重放与报告](../../docs/MODEL_V2_USAGE.md)：`diagnostics.replay_v2` / `diagnostics.report_v2`。
- [掩码读出](../../docs/LEARNING_V2.md)：`learning.train_v2 --protocol ... --out ...`。
- [登记与对照](../../docs/EXPERIMENT_V2.md)：`learning.experiment_v2 template/register/validate/execute`；研究图仅通过显式母图/派生图入口。

运行证据在 `data/diagnostics_v2/standard_stimuli_20261016/`、`data/diagnostics_v2/standard_stimuli_report_20261016/` 和 `data/malecns_v2/research_replay_W201_static/`，属于本地忽略资产。当前没有真实录像/标签，不运行动作训练、不连接设备；首期只监督 vx/wz，其他轴不当作已学会。

## Validation / Stop

总实施按 S3 的具体跨模块风险做定向验证，局部返工按 S1；保留原 FAIL 与后续 PASS。已通过检查不重复；Ruff 未安装，未执行也不称通过。数据已校验，不无原因重下载、重扫 31 GB 或重建图。本批已满足停止条件。后续需真实录像、独立动作标签和任务容差，才能推进真实训练及 M4；P4 实体闭环与 B/C 参数优化另立任务。
