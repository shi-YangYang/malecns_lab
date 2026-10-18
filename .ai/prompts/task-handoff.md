# Task Handoff — spec-008-trainable-snn / Implementation

## 下一次动作更新协议已冻结（2026-10-19）

[P3b](../../specs/spec-008-trainable-snn/p3b-joint-update-pilot.md)已获新的独立[协议PASS](../../specs/spec-008-trainable-snn/acceptance-p3b-protocol.md)，完整报告已读取处理：固定背景、50ms每帧、两帧连续100ms反向、60ms无梯度预处理；仅train-sine-combined的0/1帧同帧限幅教师目标，Adam一次联合更新，内部恢复/完整恢复/重载五分支。估计303s不是实测；内部数值更新与动作依赖性分开判定，阴性不重试。当前P3a仍实施中，P3b工具尚未实现；必须待数据/软件前置验收后再新目录登记实际运行，不把协议PASS当动作训练完成。

## 最新实际证据与进行中工作（2026-10-19）

[持续刺激实测](../../specs/spec-008-trainable-snn/sustained-observability-run-20261019-r1.md)新独立[PASS](../../specs/spec-008-trainable-snn/acceptance-sustained-actual.md)已读取处理，所有72组窗口统计和296字段前缀复核通过。379.671s/RSS1.500GiB，固定背景在持续输入21–25ms已有差异，50ms窗口覆盖方向/大小信号，零背景阴性；不再重跑。本轮P2e已提交a2320b0。按[P3a](../../specs/spec-008-trainable-snn/p3a-data-checkpoint.md)实施教师序列与完整策略参数存档，单一实施Agent仅新增文件，尚待验收；完整训练续跑状态/优化器不在该参数格式中。下一次实际动作损失更新需预登记窗口、成本和干预标准；全部目标未完成。

## 最新完成：P2e软件接口（2026-10-19）

[P2e](../../specs/spec-008-trainable-snn/p2e-neural-policy.md)实施报告和新独立[PASS](../../specs/spec-008-trainable-snn/acceptance-p2e.md)已读取处理，28项定向检查通过，无返工项，随本状态更新提交。全部DN两半窗电位、内部theta梯度、跨帧状态与controller已接好，只是软件证据，没有动作学习结论。持续刺激单次运行也已正常终止：exec56241退出0、19,400步、379.671s、RSS1.500GiB；固定背景持续期间有差异、零背景阴性，296前缀字段精确相同，实际独立验收正在进行，不重跑模型。后续要完成策略持久化及有界联合训练协议，整体AC3–AC6未完成。

## 当前子批：有状态策略接口与持续输入实测（2026-10-19）

上一轮前向优化有实际进展，所有实验已终止。本轮按[持续刺激协议](../../specs/spec-008-trainable-snn/sustained-observability-plan.md)冻结60/100/30ms、19,400步，独立[协议PASS](../../specs/spec-008-trainable-snn/acceptance-sustained-protocol.md)已读取处理；提交后新目录登记并受监督运行一次，当前不预报响应。并行实施[P2e神经策略接口](../../specs/spec-008-trainable-snn/p2e-neural-policy.md)，仅新增learning模块/测试/说明，不修改诊断绑定的脑或空间驱动源码。两者真正独立：实际持续响应决定后续训练窗口，接口要求显式窗口配置，不用未验证默认值。Noria/Docker及1GiB预留不变。

## 当前断点：前向优化已完成完整图验证（2026-10-19）

本轮有实际进展：内核基准2a3979a、实际依据与生产规格ef7a315、生产实现13716c6。软件首次验收发现证据校验仍期望旧backend，已由新实施者修复、新验收者通过；46项数值＋37项证据检查通过，原FAIL报告保留。P1 r5和配对r2随后均正常终止，新的[实际独立PASS](../../specs/spec-008-trainable-snn/acceptance-forward-csr-actual.md)已读取处理，无返工项。所有本轮Agent已完成，模型进程已退出，无需等待。

最新成本与证据见[实测报告](../../specs/spec-008-trainable-snn/forward-csr-run-20261019.md)：P1 32.391s、RSS1.753GiB，全部已保存响应/gradient/theta/gain与r4精确一致；配对108.558s、RSS1.506GiB，22个NPZ/465字段与r1精确一致。两次源身份更新正常，原始图/数据未改，1GiB额外预留与Noria/Docker限制保留。不得重跑这些已通过实验或测试来增加信心；当前前向约3.9s/20ms神经窗仍非实时。

下一步推进三轴神经策略集成，而非继续停留在数值探针：先明确跨帧保留LIFState及SpatialDriveState、场景边界reset、TBPTT detach保留队列/相位/时钟；读出选全部1314 DN的电位/时间信息，不能只用已证明各条件相同的总发放率。视觉仅进入SpatialDrive，不直连动作；BodyVelocity的float转换只能在部署边界，训练须tensor输出。当前60/20/10ms结果的DN差异均在撤除后出现，因此持续输入与控制窗口的关系须用预登记新证据确定，不能默认每帧reset20ms。可依据新前向成本执行有界持续刺激协议，同时准备可配置但身份绑定的状态/时间接口；具体实际协议需先冻结，不根据输出追改窗口。

完整图任务训练还需独立策略checkpoint（theta/读出/归一化/固定背景/图与时间身份，旧checkpoint有意拒绝tonic）、教师/场景划分/预算、冻结脑与急性恢复/直接视觉/研究重连对照、部署时延及Tello现场条件。AC3–AC6和整个goal未完成，不标记complete或blocked，也不以资源设置已完成结束路线。未启动新硬件或付费资源。

## 当前子批：完整图前向内核候选计时（2026-10-19）

生产缓存A/B已实现，46项数值检查通过；首轮独立验收发现两个证据入口仍只接受旧backend，未运行全图。新实施者同步严格新身份，37项相关证据检查通过，新验收者[PASS](../../specs/spec-008-trainable-snn/acceptance-forward-csr-cache-r1.md)已读取处理，无返工项。随状态更新提交后串行运行P1 r5与配对r2，数值协议保持；尚无此版本完整图性能结论。

软件已独立通过并提交2a3979a，实际`data/forward_kernel_20261019_r1/`正常终止5.49s、采样峰值1.443GiB；三输入全部预热/五轮输出逐元素一致，SciPy中位约15.3ms对Torch约74.6ms。新的[实际独立PASS](../../specs/spec-008-trainable-snn/acceptance-forward-kernel-run-r1.md)已读取处理，无返工项。按[生产前向子规格](../../specs/spec-008-trainable-snn/forward-csr-cache.md)实施共享SciPy前向与原Torch全细胞伴随，软件与完整P1/带背景动力学对照分别验收；当前未实现生产替换，不把内核收益当整体收益。

上一轮只确认既有1GiB设置，属于状态复核，没有新的实施进展。本轮继续已授权目标，按[前向内核实测协议](../../specs/spec-008-trainable-snn/forward-kernel-benchmark.md)先实现可重复的离线基准并独立验收，再提交和运行一次。生产模型暂不修改；数值或性能结论待原始结果。启动前只读快照可用5.513GiB、swap1.434GiB，无已有模型进程；实际预算以新登记时为准。Noria/Docker保留。

## 当前断点：完整图配对响应已有实际证据（2026-10-19）

上一轮属于实际进展：完整图内核缓存软件及r4实际对照均独立通过，首轮配对诊断也已完成。新运行 `data/observability_20261019_r1/` 正常终止，5400步、468.51s、采样峰值1.500GiB；[实测报告](../../specs/spec-008-trainable-snn/observability-run-20261019-r1.md)及[新独立PASS](../../specs/spec-008-trainable-snn/acceptance-observability-r1.md)已读取处理，无返工项。没有活跃的模型进程或未处理Agent报告，所有模型与证据保持完整。

科学结论：零背景本窗DN无差异；固定8mV背景全部六刺激的DN电位有差异，数量378/246/105/243/50/699，左右/上下/小大对比249/229/682。超容差差异首次出现在刺激开始后21–25ms桶，即撤除后1–5ms，不能区分延迟传播与撤除响应。所有条件每个DN的30ms总脉冲数相同（全体总计127），唯一脉冲桶变化为body11072的87→88ms。因此不能把当前总发放率当可区分刺激的读出，也不能每帧清零后只观察20ms。图、绘图脚本与派生时序分析在该本地目录；不重复本次已完成实验来增加信心。

下一步已授权且不受旧内存等待限制：

1. 前向成本仍约16.5s/200神经步。只读 `forward_kernel_cost_review` 已返回并处理：本机非MKL Torch路径对每条边调用长度1的axpy，建议先受监督计时同一完整权重的Torch CSR与SciPy `csr_array`乘向量（固定零/稀疏/稠密向量、包含桥接成本、全部dtype/索引/内存共享语义），不要提前改生产核或宣称加速。参考官方源码：Torch `5c4886908584029761b579af026dcfb627c84070` 的 `aten/src/ATen/native/sparse/SparseCsrTensorMath.cpp`，SciPy `v1.18.1/scipy/sparse/sparsetools/csr.h`。候选未实施/未计时；核级通过不能替代后续完整动力学与梯度对照。保留1GiB预留、Noria/Docker，不并行加载多个全图模型。
2. 只读 `snn_training_integration_readiness` 已返回并处理：复用FollowEnvironment/target_mask/SpatialDrive/全部1314 DN；新策略需 `forward_tensor` 保梯度，部署边界才转BodyVelocity；跨帧保存LIFState及SpatialDriveState、场景边界reset、TBPTT只detach。固定背景、电位或时间特征是由本次信号支持的候选，不自动解冻rho。后续协议须覆盖延迟/撤除效应并验证持续输入响应，再做有内部gain更新的三轴任务训练。
3. 策略checkpoint需独立保存theta、读出、归一化和背景/时间/图身份；旧gain-only checkpoint不支持tonic或新读出，不可绕过。研究重连图目前被训练loader及policy map拒绝，后续需要隔离的受验证研究入口，不能把research_control冒充full_reference。冻结脑、急性恢复、直接视觉和拓扑比较均保留；AC3–AC6未完成。

本轮不创建新的硬件许可、不停止在用服务、不宣布整个goal完成。旧段落中“待内存释放/尚未运行”只描述当时状态，以本节与实际报告为准。

## 最新进展：伴随优化及实际对照完成（2026-10-19）

软件提交 `3f68a09` 后，新登记 `data/trainable_probe_20261019_p1_r4/` 返回完整数值通过；[r4报告](../../specs/spec-008-trainable-snn/p1-run-20261019-r4.md)及[新独立PASS](../../specs/spec-008-trainable-snn/acceptance-p1-run-r4.md)均已读取处理，无返工项，缓存子批A–E完成。实际总墙钟83.69s、反向13.79s、采样峰值1.745GiB；与r3相比全部gain/神经响应精确相同，梯度和theta有已记录的浮点末位差。只解释同协议单次性能，不代表通路或任务能力。

下一步[首轮通路协议](../../specs/spec-008-trainable-snn/observability-plan.md)已按新前向成本冻结60/20/10ms、5400步，预存 `data/observability_20261019_r1.protocol.json`；预估574.58s并非实测。使用新目录保留全部两模型七条件，无梯度；未启动前不声称结果。现有1GiB预留及Noria/Docker保留约束不变。整体AC3–AC6仍未完成。

## 当前子批：复用完整图伴随矩阵（2026-10-19）

上一轮属于实际进展：`62818a2` 已记录完整图 P1/AC2 的新独立 PASS，短窗资源阻塞解除。按[p1-adjoint-cache](../../specs/spec-008-trainable-snn/p1-adjoint-cache.md)实施静态转置 CSR 缓存，保留所有梯度与原数值协议。软件验收及提交后仅用新目录进行一次相同 P1 对照，实际性能另报。没有新的产品决策或硬件操作，不需要再次请求已授权的本机研究权限。

伴随缓存的软件 A–D 已获新的[独立 PASS](../../specs/spec-008-trainable-snn/acceptance-adjoint-cache.md)，44 项相关检查通过，实施及验收报告均已读取处理，无返工项；E 实测仍待提交后运行。后续配对可观测性只读规划已返回并读取：完整七条件、两动力学的60/20/10ms候选为5400步，最终窗口依据优化后无梯度前向成本在运行前冻结；不提前声明视觉到 DN 接通。Noria/Docker 保留，系统预留 1 GiB。各子批提交不结束完整目标。

## 最新实测：完整图内部更新已跑通（2026-10-19）

1GiB预留的软件调整已独立PASS并提交 `28e1b55`。随后一次新登记 `data/trainable_probe_20261019_p1_r3/` 正常结束，父进程完整证据校验后返回 `passed / numeric_internal_update_verified`，见[P1运行报告](../../specs/spec-008-trainable-snn/p1-run-20261019-r3.md)。完整166,700节点、25,582,938边、124,177,617突触计数保持；本次只有R1–R6内部共享增益实际变化，809个L1响应改变，恢复/重载/原始图摘要均通过。

实测采样峰值1.830GiB、总墙钟362.64s，反向291.70s；启动可用3.699GiB、有效RSS预算2.699GiB，最低采样系统可用2.573GiB。没有停止Noria或Docker。此前内存阻塞已不适用于这次短窗运行，不能继续等待释放通知作为下一步条件；长窗训练需求仍未知。新的独立验收者已返回[PASS](../../specs/spec-008-trainable-snn/acceptance-p1-run-r3.md)，报告已读取处理，无返工项；P1/AC2已关闭，AC3–AC6及整体目标跟随、实飞仍未完成。

只读性能审查报告已读取：当前伴随路径存在每步CSR转置到CSC后由Torch转回CSR的转换；可候选预构造伴随CSR，静态额外数组约0.287GiB。尚未实施或计时归因，不把291.70秒全部归因于此。后续按数值等价与相同完整协议比较，再冻结配对神经响应实验；不得绕过完整图或先声称策略已学会。

## 已完成调整：按用户要求预留 1 GiB 并重新测量（2026-10-19）

用户明确要求“保留到1gb”。[Decision 012](../decisions/012-one-gib-system-reserve.md)和[资源调整子规格](../../specs/spec-008-trainable-snn/resource-reserve-1g.md)授权将共享监督、P1及配对诊断新协议默认预留改为1GiB；保持6GiB进程上限、600秒、持续监控及完整连接组。实施与独立验收通过后提交，再由协调者用新目录运行一次P1探针，实际结果另报。旧4GiB登记/失败证据不覆盖，Noria限制继续有效。下面blocked及4GiB限制属于此前阶段记录，不覆盖本次用户授权。

软件调整现已完成，41项定向检查通过并获新独立验收者[PASS](../../specs/spec-008-trainable-snn/acceptance-reserve-1g.md)；两份报告均已读取处理，没有未解决返工项。随后实际测量及其独立验收也已通过，见本文件顶部；软件检查与全图实测为不同证据。

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
