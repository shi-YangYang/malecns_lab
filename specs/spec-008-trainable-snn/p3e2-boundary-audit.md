# P3e2 — 研究神经策略与持久化边界审计

状态：**Draft / 只读规划审计，尚未 Ready、尚未实施、不是验收结论。** 2026-10-19，依据当前源码与 Spec 008 检查。正式 R 的首两段已独立实际 PASS，但完整生成终局仍未具备；本稿不授权使用中途图或探针图训练。G 的 29 个绑定源码、R 生成的 13 个绑定源码均须保持。

## 结论与建议分批

可以沿用现有 LIF 方程、代理梯度、正向/伴随内核、感觉编码、全部 DN 读出和数值证据复核。不能直接把研究图送入现有完整图训练入口：构造、参数格式、Adam 校验、训练登记和评估登记各有独立的完整参考图限制。

建议按协调者进一步缩小后的范围，先冻结 **P3e2：正式 R 准入、真实缓存安装及模型身份**；参数/Adam/完整状态格式和训练runner留到 P3e3，G/R闭环比较再单独分批。P3e2可以独立完成，无须先复制或实现训练/持久化边界。下文这些后续边界是依赖审计，不是P3e2实施清单。

当前不需要重启或改写已有 G run；R 必须有自己的新 run、policy/model/checkpoint 身份。后续 G/R 相同计算契约、相同初始化与训练机会由明确配对凭证约束，不要求图相关的所有 ID 字符串相同。

## 可立即独立冻结的更小 P3e2 范围

最小公开工厂可以只返回 `(research_brain, verified_mapping, admission)`：精确现有 `LaminaTonicLIF`、实际R counts/全部计算缓存、model_id及完整研究来源。新生的LIF state会自然携带R model_id；暂不创建optimizer、保存checkpoint、启动训练或闭环。若实施Plan决定同时组装现有 `NeuralPolicy`，也只用原读出/感觉构造，不开放旧参数reader；这是可选的小延伸，不是完整训练前置条件。

**这一阶段必须处理的具体漏洞**：

1. `load_research_control` 并不证明“正式终局R”：工厂必须从生成run调用父终态准入，交叉验证最终candidate实际source/hash/transform/生成checkpoint；不能只相信metadata里的 `generation_terminal=true`。
2. 安装必须同时覆盖raw counts、forward CSR、adjoint CSR、SciPy共享视图以及新model_id；三缓存逐项真实核对。仅换 `.graph`、保留G伴随、过早预处理或没有重新绑定source都不能验收。
3. 图/输入/DN/类型/tonic成员的数字和语义应相符。源树中的 `NeuralPolicy._initialize` 不会替工厂证明每条边真实替换；必须工厂自己证明。固定背景明确所有L1–L5，不能因研究role放宽成子集。
4. 工厂不能接受调用者已有运行状态或任意半成品脑对象；研究R不能被包装为数值fixture/完整参考图通过。轻量测试的数组安装器可用明确数值夹具，但公共真实工厂必须拒绝它们。
5. 先释放SciPy/NumPy持有的G缓存再建R缓存；不保留整个G模型及R多份复制并行。构造资源未知需待终局后实测；不因目前G能训练就宣称R构造也能满足预算。
6. 不改29/13绑定源；新增工厂源必须进入新的模型身份。旧生产参数loader继续拒绝研究role。注意现有 `NeuralController` 只检查 `isinstance(NeuralPolicy)`，不是角色准入器；当前并无设备路径使用它，但不能宣传“所有任意内存控制器都拒绝研究policy”。本阶段不增加任何设备入口；若只返回研究brain/map更不会新增这种接线。

P3e2的软件检查可仅覆盖准入反例、缓存结构/共享内存、显式小电路真实forward/adjoint、身份和状态混用拒绝，不需要训练checkpoint或跨进程Adam测试。当前训练正在执行时仍按根任务串行资源安排，不在本次只读审计中运行这些检查。

完整图factory实际门槛必须等正式R终局，并另登记一个受监督的小作业：保存真实G/R图与缓存全量对应证据、共同元数据/初始数值、实际R的forward和adjoint乘向量与独立矩阵参考、raw counts前后摘要/父终态。可以不执行Adam、不训练首窗、不跑预处理或整段策略。G部分应复用已验证实现/已有初始数组作锚点，只验证新增安装逻辑；具体确定性向量、浮点容差与资源预算在运行前冻结。此类结果只证明安装与计算，不能称作R已训练或恢复了训练状态。

## 实际调用链及可复用范围

| 位置 | 当前事实 | 后续最小边界 |
|---|---|---|
| `generate_rewiring.accepted_candidate` | 重新验证源/runtime、全部父终态及唯一 terminal；只返回最终路径/run/checkpoint/prefix/graph SHA，不加载图 | 研究登记必须从生成 run 调用它，不能接受裸 `candidate.npz` 路径作为充分凭证 |
| `graph_controls.load_research_control` | 校验母图、R/W/P 来源、实际连接及机械不变量；本身也接受历史研究图 | 在专门准入后使用；另明确限制 transform=R、正式 generation run/checkpoint/terminal，排除 W/P/成本探针 |
| `trainable_graph._graph` | 严格构建只读整数 counts、完整类型分组与真实 ordered-counts 摘要；不会自行证明来源 | 可以在验证过的 R 上复用，不开放手工 `TrainingGraph` 作为公共研究入口 |
| `TrainableLIF.__init__`、`LaminaTonicLIF` | 前者只接受 full_reference 或数值夹具；后者依赖它；共有实际传播缓存 | 在新工厂内先用真实 G 构造精确现有类型，任何 state/forward 之前完整安装已验证 R；不把 R 临时标成 G，不新增替代动力学 |
| `policy_groups._build_policy_map` | 按原始注释生成全部 R1–R6/DN；公共 `load_policy_neuron_map` 只接受 G | 工厂对验证过的 R 调用共享构建函数并核对实际集合；研究 map 的 role/training_graph_id 保持真实 |
| `NeuralPolicy._initialize` | 精确脑类型检查；显式 role 与图/map 一致即可，并未写死只允许 G | 以真实 `research_control` 初始化精确 `NeuralPolicy`，继续使用原 `initial_state/forward_tensor`；不使用继承类逃过类型检查 |
| `policy_checkpoint._arrays/_validate_identity` | 明确拒绝 research_control，且校验精确源清单与嵌套语义 | 独立研究参数格式/验证器；不可通过伪造 identity 或删除校验调用旧参数存档 |
| `training_optimizer._policy_parameters` | 调用上述 `_arrays`；每次 snapshot/validate 均会触达 | 新的严格命名参数准入及 Adam 绑定，复用既有 Adam 配置与 `validate_moments`，不改旧模块 |
| `training_window.compute_window` | `optimizer_snapshot → validate_optimizer/snapshot_optimizer → _policy_parameters`，因此即使传入合法研究 policy 也会拒绝 | 当前不能直接原函数复用。若需在绑定源冻结期间训练 R，只新增小型窗口调度版本，继续调用原 forward、状态摘要、clipping、loss/Adam 数值复核 |
| `training_checkpoint` | 顶层 save/read/load 均依赖旧参数/Adam边界；上下文、完整状态数组函数相对独立 | 复用 `_validate_context/_state_metadata/_state_arrays/_validate_state/_build_state` 等原状态契约；另设研究 manifest/schema/目的与严格 restore |
| `training_archive.inspect_checkpoint`、`training_evidence` | 精确旧 schema/purpose/来源、P3b r2历史和 G 第一窗比较绑定 | 新增研究准入/封存边界，复用摘要、游标、资源采样、数字复算；不调用 `production=False` 冒充实际研究验证 |
| P3d evaluation | 参数 identity、完整 state 数组与环境 replay 多为通用；suite/worker/final-training/terminal 存在 G 与固定条件绑定 | 共享 rollout/评分/状态；以后单独登记 R 适配和 G/R比较，不能直接宣布旧 CLI 已支持 R |

因此，“共享数值核心”不等于“现有 runner 原封不动接受研究身份”。必须在 Spec 中如实列出新增持久化/调度，而不是先放行 role 再补证据。

## 正式 R 的准入与身份

建议轻量父进程产生不可混用的 `ResearchGraphAdmission` 记录，字段至少包含：

- 正式生成 run 的绝对路径、run ID 与文件 SHA；唯一 terminal checkpoint ID、前缀、最终段与父 assessment ID/SHA；实际 candidate 文件 SHA/source ID/logical graph SHA。
- 完整母本的文件 SHA/source ID/logical graph SHA、元数据摘要；已冻结 seed/科学预算与实际 attempts/success/rejections；所有完整机械诊断文件 SHA。
- 生成工具 13 个源摘要和 runtime；新研究入口自身源摘要及独立协议 ID。路径只负责定位，不能替代字节和内容身份。

`accepted_candidate` 当前返回的少量字段不是完整训练登记：新边界还需从已校验父证据保存上述相关身份。父进程可读 JSON、小 NPZ 元数据和流式 SHA；实际稀疏数组及全图机械检查在受监督 worker 进行。父登记后 worker 要再次核对准入、candidate 文件和实际图，封存前再核对；不信任由调用者构造的 dataclass 即自动获得权限。

研究 `TrainingGraph.identity` 明确使用 `role=research_control`。推荐保留现有 `reference_source_id/reference_graph_sha256/reference_provenance` 的**母本 G**含义，另加 `research` 节承载实际 R 的 source/hash/transform/生成准入；`raw_ordered_counts_sha256` 必须由 R 真实数组计算，`training_graph_id` 绑定全部。不能把 R hash 填入 G 的 provenance，不能因节点/边数相同沿用 G 的 raw-counts SHA。

`_build_policy_map` 会把图 ID 绑定进 map，空间映射再绑定 map，感觉 drive 再绑定空间。因此 R 的 model/map/spatial/drive/policy/state ID 自然改变。G/R 的 body IDs、输入/DN集合、像素分配、初始 phase、初始 theta/readout **数值**则应保持；严禁为了让两侧 ID 相等改写已有 state 标签。

补一个严格的“共同计算契约”而非任意递归删除字段：明确列举 LIF配置、数值后端、固定8mV背景及成员、类型分组/顺序、输入/DN身体ID/索引、空间分配/相位种子、相机/时间、读出形状/归一化/初始数组、优化参数、数据/场景/预算。只允许图内容/研究准入及其派生 ID 不同；adapter源与数值核心源分别记录。不得用过宽的 `ignore graph fields` 隐藏动力学差异。

## 不修改旧核心的缓存安装顺序

新工厂建议放在 `learning/research_policy.py`，因为准入依赖 learning 的生成/研究协议；避免让 brain 包反向依赖这些实验工具。它只返回完整、已校验的 policy，不公开任意运行中模型的换图 API。

1. 校验正式生成准入。通过现有真实 G loader/map 创建精确 `LaminaTonicLIF(mode=fixed)`，先不创建 `PolicyState`、不调用 reset/precondition/step/forward，不建立 autograd 图。
2. 在加载/构建新传播缓存前释放 G 的 `_forward_csr` 与 `base_weights/base_weights_adjoint`；尤其 SciPy 的 NumPy view 会保留 Torch 存储，仅 `del base_weights` 不能保证释放。不得保留临时 matrix/arrays 引用。整个构造只在预算内 worker 内执行。
3. 通过专门准入及 `load_research_control` 校验实际 R 和完整母本；复用 `_graph` 形成 R 只读 integer CSC。保留全部166700节点、25582938边、类型/缺失掩码/顺序/自环，不按可达性或活动筛选。删除不再需要的大图副本；新 `_graph` 会拷贝 counts，构造峰值必须实测而非假设与 G 相同。
4. 真正替换 `brain.graph` 并建立 R 的三个传播表示：Torch 正向 CSR(post,pre)、Torch 伴随 CSR(pre,post)、共享正向 Torch 存储的 SciPy `csr_array`。权重运算严格保留先转目标 float dtype 再乘 `weight_mv` 的顺序；不能用 float64先乘再降精度。
5. 对所有边分块校验 CSR指针/索引/值与实际 R CSC；伴随由 CSC 编码的转置对应；SciPy data/indices/indptr 的指针、dtype、shape确实与正向Torch共享，且NumPy视图只读。对比所有结构/值，不能只检查某个乘向量巧合相同。全量结构核对之外，数值夹具必须实际测试 forward 和 all-cell adjoint。
6. 核对 `group_index/input_mask/input_body_ids/tonic_group_index/rho` 与 R 的完整注释/选择一致；这些向量数值应等于 G。更新 model 内真实 graph identity 与新 adapter/cache来源，重新计算 model_id。精确现有类的 `step` 保持原实现，不改全局函数或方法。
7. 使用 R map 以 `role=research_control` 调用 `NeuralPolicy._initialize`，再将研究工厂源身份纳入 policy，并重新计算 policy_id。最后才允许 initial_state。若任何检查失败不返回半安装模型。

要特别覆盖：旧 `_forward_csr` 仍指向 G、新 forward 配旧 adjoint、仅修改`.graph`、符号/转置错误、换图时已有 state/延迟队列/反向上下文、类型分组重排、只改 digest 未改缓存，以及 CPU/dtype转换后破坏共享视图。研究工厂不提供 `.to(MPS)` 或任意运行中 dtype转换；当前数值契约仍为 CPU float32。

这个方案复用了真实 G 构造时的非图参数与缓冲初始化，但 R 从未借用 G 角色运行；完整 R 在第一次神经计算前已安装。若独立设计审查不接受受控安装，则需要正式版本化构造核心，不能退回临时伪装 role。

## 后续 P3e3：参数、Adam 与完整状态的最小版本化边界

以下仅为后续依赖，不纳入缩小后的P3e2。建议新增职责清楚的少量模块，而不是复制现有整个目录：

| 候选模块 | 主要职责 | 继续调用的原实现 |
|---|---|---|
| `research_policy.py` | 受准入约束的图/计算缓存安装与共同计算契约；G锚点构造通过原 G工厂 | `_graph`、map构建、LaminaTonicLIF、NeuralPolicy |
| `research_policy_archive.py` | 明确 offline研究参数用途；全嵌套身份与三参数真实值校验、原子恢复 | JSON/hash、selection/model语义的兼容纯助手、三参数shape/dtype校验 |
| `research_optimizer.py` | exact named parameters/对象/顺序、固定 Adam选项、状态绑定与恢复 | `optimizer_contract`、`ADAM_OPTIONS`、`validate_moments`；不调用会拒绝R的旧 `_policy_parameters` |
| `research_training_state.py` | 独立研究 schema/purpose 与parent/prefix/参数/Adam/完整神经和感觉状态封存 | 原上下文、状态数组提取/校验/构造、摘要规则 |
| 后续小型研究窗口/登记边界 | 显式1–4窗及资源父证据；共享数值校验，不声称旧P3c完全通用 | 原 `forward_tensor`、`training_math`、数据选择/游标、state_digest、resource_guard |

名称是候选，不是已授权文件清单；不要为了上表每行强行建立空转发层。具体切分由 Ready Plan 根据实现大小决定。

原 `policy_checkpoint._validate_model_semantics` 可以检查不变动力学的部分，但它和 `_validate_identity` 精确要求旧字段/源清单。研究验证器应明确验证自身扩展的允许字段及源码，不能把真实研究身份临时“投影成 G”后交给旧验证器来获取通过。所有 `_` 私有助手若复用，都应列入源绑定及边界检查；不能声称它们是稳定公共 API。

参数档只保存 theta/readout，研究目的须与生产 `deployment_parameters_only` 分离。完整训练档另保存真实 Adam step/一二阶矩、`v/g/refractory_until/pending`全部18槽、感觉 phase/event_counts、frame/神经/感觉时钟、scene游标、parent/prefix、run/protocol/dataset和图/策略身份。保留“前向状态在Adam前计算，更新后保留值并detach”的现行语义；场景边界只重置神经/感觉，参数/Adam连续。

必须先校验全部身份、数组、目标对象及所需分配，再一次性修改参数/Adam；不能恢复一半发现错误。G/R即使维度完全相同也必须相互拒绝；不同R种子/生成终局/图摘要也相互拒绝。新数值初始化仍为 theta=0、同readout1703、phase1702、空间1701；不能从已训练G复制 theta、读出或 Adam 作为 R 起点。

父进程要有 NumPy/JSON 研究存档检查器。不能因为原 `training_checkpoint` 有可复用助手就把它 import 到轻量父进程：该模块顶层导入 Torch。父/worker分层继续明确。

## 为何仅等待旧 run 完成也不能随便改绑定文件

`training_contract.validate_run` 和 P3d `final_training_source` 都重新核对当前29个源码；`accepted_candidate` 重新核对生成13个源码。它们并非只在训练尚运行时检查。即使G已完成160窗，改旧文件也会使旧最终证据准入失败，而非自动允许未来的R使用新版。

首选保持这些实现字节，新增必要版本化边界；旧P3c服务当前G/frozen/已封存历史，不当作可随意删除的冗余。不能通过忽略 source mismatch、monkeypatch导入、临时改role或 `production=False` 绕过。

严格同时要求“不改任何绑定源码”和“不存在任何重复调度代码”在当前接口下不可兼得：原 `compute_window` 的 optimizer验证依赖无注入点。因此应接受**受限的新版窗口调度**，数值动力学/读出/梯度复算保持共享，并对照旧G首窗证明等价；这远小于复制整个P3c/P3d。未来如需抽取公共kernel，必须另立源码版本与历史执行/来源验证策略，不把新源码冒称旧run来源。若改动会改变 G/R 数值契约而非只改变身份边界，必须新配对协议及新run；不能接着旧G游标续训。

## 后续训练批的有界软件与首次实际门槛候选

后续训练批实施风险为 S3（研究身份和持久化）；本次仅S0文档审计，下面均未执行，也不属于P3e2仅工厂验收条件。只做相关子系统的最小充分检查，旧已通过检查不无理由重跑。

软件门槛建议覆盖：

1. 合法终局准入，及中途/探针/W/P/缺父终态/源变化/文件替换拒绝；图内容改变后重新算JSON摘要也不能过实际数组检查。
2. 显式非生物数值夹具上的G/R不同连接真实影响forward与adjoint；对完整权重的独立矩阵乘法/代理梯度参考，不以“ID不同”证明换图；生产入口仍拒绝研究格式。
3. G兼容构造保持原数值，R图/map/空间映射数值与共同契约一致但所有需隔离的身份不同；交换/错用cache、state、参数档均拒绝。
4. 同进程连续两窗与新OS进程restore两窗的参数、全部神经/感觉值、Adam逐数组等价；缺状态槽、错moments、错误mode/父游标、保存中断均拒绝，且失败不半恢复。

实际门槛在软件独立验收、正式 R terminal及机械证据具备后，另登记后执行；每项单进程串行、1GiB预留/6GiB申请/600秒/.1秒监控，不能与G训练并行：

- **G新边界等价检查**：只用新边界执行已登记初始G的第一对train-combined帧（60ms预处理、2×50ms、一次Adam）。对照既有第一窗封存证据的真实输入、参数、DN特征、动作、原始/裁剪梯度、更新值、Adam及末态数组；身份层新增字段单独解释。初始数组/输入/结构应exact；相同核及操作顺序的数值也应exact，若计划容差须在运行前冻结且报告非exact项。这是新适配的诊断，不增加G正式训练预算、不重训完整G；不需要重新跑旧实现生成对照。
- **R首窗**：正式R从相同初始参数执行一个两帧窗，完整缓存对应证据在任何forward前写出，保存实际三参数、Adam、全部状态、raw-counts前后摘要与父终态。此窗应成为唯一正式R训练前缀，不另挑一次效果好的结果。
- **R跨进程次窗**：新worker先恢复并封存真实所有参数、Adam、神经/感觉值和时钟，逐数组/摘要等于首窗，再执行下一对帧。只在新的独立实际验收确认后开放已登记的最多4窗段。不要为检查而把R首窗重复训练或重置seed。

R状态恢复是否正确用首末实际数组与数值复算判定；R与G动作是否不同、是否下降loss、是否非零theta梯度是实验结果，不是挑选有效R的准入门槛。静默、爆发、饱和、有限但零梯度保留并继续按冻结失败规则处理；NaN/Inf或资源失败留下原始证据，不更换R种子。受控R等价于连续运行的强证据先由数值夹具完成，完整图首两段只声称实际恢复/数值和来源证据，不凭空宣称与未运行的完整连续R轨迹逐位相同。

## 后续评估如何复用而不冒充支持

已有 `evaluation_rollout` 的环境更新、监督、动作复算、重放和评分可直接复用；`evaluation_checkpoint` 的完整状态函数也以传入identity为准。`evaluation_worker.execute_segment`多数数值流程可复用，但上层 suite、条件枚举、final-training准入，以及 `evaluation_evidence.validate_completion` 的固定G raw SHA都不是通用接口。

因此后续评估批需要显式研究suite/final-R准入，不能将R伪装成旧P3d的full_reference，也不能对实际R调用 `production=False` 去跳过raw-counts和资源证据。可在尚无实际P3d run时另行版本化、提取必要的准入/终态依赖注入并保留严格G默认路径；若届时P3d已有源绑定run，则继续新增窄边界。这个选择应在对应评估批 Ready时决定，本稿不提前改代码。

主比较候选仍为相同160窗预算后的G/R最终档；复用既有G证据，不能让R从G已训练读出起步。单seed1901只作探索性结论，六test场景等权主指标/失败处理等需在首次R神经结果之前正式冻结。当前本稿未替代该研究登记，也未关闭AC3–AC6。

## 本次工作记录

- 阅读治理、Spec008/Plan、P3e/P3e1与相关Decision；按实际调用链静态查看上述源码。
- 仅新建本 Draft 审计文档；未修改业务代码或任何绑定源，未改其他Agent文件。
- 未运行测试、导入Torch、加载稀疏矩阵、启动模型或生成作业；未操作Noria/Docker、未commit。
- 已按协调者的最新范围建议缩小P3e2为严格factory/cache/model identity；仍待协调者冻结准入与实际门槛、交新的独立协议审查。持久化/训练留后续P3e3；本报告不自行标记Ready或PASS。
