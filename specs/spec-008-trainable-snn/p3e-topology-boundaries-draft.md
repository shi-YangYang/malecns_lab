# P3e — 受控拓扑实验的接口边界

状态：Draft，仅只读规划，未冻结科学预算、未实施、未生成或运行新R图。协调者已读取处理独立规划报告。本文件避免将后续研究入口误接到当前训练，不改变[P3c](p3c-training-runner.md)或[P3d](p3d-closed-loop-evaluation.md)；当前G训练、r2阴性和既有探索已被观察，不能追述为整个项目事前登记。

独立只读细化后，首个[完整R生成成本探针](p3e-generation-probe.md)已单列并通过协议审查。它只固定seed1901、100000提案/成功上限及既有资源保护；其图不进入训练。本主研究仍为Draft，具体交换预算以成本证据另行冻结。

## 当前阻断点

`brain.reference.load_reference/validate_reference`、`trainable_graph.load_training_graph`只接受full_reference。`TrainableLIF.__init__`仅接受该角色或显式数值夹具；`policy_groups.load_policy_neuron_map`绑定完整母图。`NeuralPolicy.from_reference/_initialize`校验图/映射/角色及精确脑类型。参数/训练存档和P3c的`training_contract/training_worker/joint_probe_identity`还绑定固定r2图、policy及历史证据。

因此不能改标签冒充full_reference，也不能只新增research loader便直接复用P3c。当前29个训练来源文件必须保持，不能为后续对照破坏已有run的源码身份。

## 已有能力与诊断缺口

`learning.graph_controls.create_research_control`已有R：同符号源双边交换，权重留在原源，固定自环，拒绝重复边/新自环；W：固定拓扑、逐源非自环权重置换。`load_research_control`验证母图与派生身份；`graph_diagnostics.validate_rw`核对全部节点/注释/群组、逐节点入出度、符号入度、逐源有符号/绝对出强度、全局权重多重集、自环。

- 算法保留逐源权重多重集，但既有独立验证只检查逐源强度与全图多重集；如新协议声称前者，必须补相应机械校验。
- `input_output_paths`采用旧reference群组，不等于当前3377个R1–R6输入到1314个DN；应按PolicyNeuronMap另登记，并报告未保持的逐目标入强度变化。存在图路径不证明功能传播。
- 既有实际验收覆盖完整图W工具，不证明完整R图已生成、交换充分混合或拓扑有用。

## 候选研究路径

独立离线研究图工厂保持全部body ID/注释/分组和原始计数，明确research_control身份；共享现有数值运算，禁止合成脑。研究策略可参考ResearchNeuralReplayV2适配方式，但任何连接安装须在初始化/前向前同时匹配实际计数、Torch正向CSR、伴随CSR、共享SciPy CSR以及图/映射/模型/策略身份，不能只改`.graph`标签。是否采用该路径仍需正式设计与独立审查。

研究存档/runner独立于P3c固定r2身份，可复用窗口数学、Adam复算、资源监督和状态契约。除明确图差异外，G/R感觉坐标、全部输入/DN、背景、时间、分组、读出容量、优化规则与数据保持一致；实际参数另有数值身份。软件需证明研究图被生产入口拒绝、真实前后向使用研究连接、G计算与已验收实现一致、G/R状态不能互换。

## 首次R结果之前须冻结

图种子数量与生成预算、交换不足/零成功处理、G/R主比较与W是否补充、配对读出种子、主指标/场景配对单位及所有失败统计。交换次数/边重叠率不自动等于充分混合。G/R各自从相同初始化独立训练内部参数与读出、同样160窗机会；不能给R套用已训练G读出。资源失败不能只给单侧追加训练预算。

静默、爆发、不可达或训练失败必须保留，不筛选“表现正常”的R图。节点/边数不得暗中减少，W或删边P不能代替保持边集合规模的主R对照。1GiB额外预留、Noria/Docker保持；具体生成/训练资源探针另行登记，不从W历史耗时推断R可行。

## 细化后的候选边界（尚未Ready）

- 主比较可先固定单个seed1901的探索性R，与既有G同一初始化（读出1703、感觉相位1702、空间1701），各160窗；复用G实际最终档和同条件闭环结果，不重复训练G。非自环边25,582,837；候选成功目标为其1倍、提案上限4倍，必须先取得成本证据再冻结，不宣称已混合或具总体统计显著性。失败及不足预算保留，不换种子筛选。
- 现有R循环的内存不随交换次数累积，主要逐提案Python查找及列内移动；若需续段，候选只提取原提案步进，保存真实CSC、PCG64状态、计数与父摘要，不能每段重设seed。诊断若实测超预算才针对性分块，不能提前删除统计或换算法。
- 研究工厂可用真实G创建现有精确脑类型，在任何initial_state/前向之前释放旧计算缓存，再安装已验R的counts、正向/伴随CSR及共享SciPy视图、重新绑定model身份；完成全量分块匹配后，以真实research_control图/映射调用已有NeuralPolicy初始化。不得临时将R标为full_reference，不得遗漏任何缓存。
- `compute_window → training_optimizer → policy_checkpoint._arrays`仍会拒绝研究角色，因此不是只加loader就能复用训练。新增必要研究身份、命名Adam校验及持久化/登记边界；复用既有step、forward、感觉编码、数据、training_math及完整状态数组逻辑，不复制整个P3c/P3d，不monkeypatch旧验证器。
- 补逐源排序权重多重集的机械比较；以完整3377输入/1314DN计算每DN从任一输入、每输入到任一DN的最短有向跳数，两份完整body ID/距离向量及不可达项保留。这是双向多源覆盖，不是所有输入-DN对的全对最短路。另报告全部节点正/负/绝对入强度变化，不声称入强度受保持。
- 首轮主指标候选为六个test场景的等权qualified fraction与G−R；逐场景成功、三轴误差、覆盖和全部失败并列。正式R协议须先固定才能看R结果；W不能替代R，资源/证据不完整不填零或形成完整均值。
