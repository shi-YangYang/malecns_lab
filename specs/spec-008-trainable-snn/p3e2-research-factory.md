# P3e2 — 完整研究图的神经策略构造

状态：软件已实施并获新的[独立软件PASS](acceptance-p3e2-factory.md)，[实施报告](implementation-p3e2-factory.md)和验收报告均已全文读取处理，无返工项。37个不同定向检查通过；G29/R13绑定源保持。只完成此软件边界，未运行完整R神经实验或训练；实际门槛仍按本规格后文另立协议。

## 背景与目标

正式R生成器已有独立软件及首两段实际PASS，后续保持唯一前缀。现有主模型构造与训练存档只接受full_reference；仅改图标签会使真实传播缓存仍引用G。本子批建立一个明确离线研究用途的构造入口：首次神经计算前，完整R的真实整数图、正向/伴随稀疏矩阵、共享SciPy缓存、图/模型/输入输出/策略身份全部一致。

这是既有Spec008拓扑对照的必要软件边界，不是新的产品方向。用户已授权完整实施及逐步提交；只用当前Mac，保留1GiB额外预留和Noria/Docker。正式研究比较仍须在首次R神经结果之前冻结，不因软件可构造就提前查看R行为。

## 输入、输出与非目标

公共输入为正式生成run目录及显式既有模型/相机/感觉配置；不接受裸candidate路径、手工TrainingGraph或中间checkpoint作为充分准入。母本路径从已验证生成登记取得。输出为精确现有NeuralPolicy类型、真实research_control角色、固定8mV背景/CPU float32、全部3377输入和1314DN；复用原forward_tensor/initial_state及现有LIF方程、代理梯度。

本批不新增参数/Adam/完整状态存档、不复制compute_window、不新增研究训练或评估CLI、不变更生产入口、不启动任何完整图神经计算。现有旧参数/训练存档继续拒绝研究策略；后续P3e3另设严格研究持久化及训练边界。没有临时角色伪装、monkeypatch、忽略source mismatch或production=False准入。

本批构造所得初始theta=0和读出均未训练。计划后续训练的参数仍仅为brain.theta、readout.weight、readout.bias；原始计数、拓扑、递质符号、背景rho与感觉映射不训练。实际更新与任务收益不属于本批验收。

## F1 轻量终局准入

1. 父进程可通过独立函数读取生成run并调用accepted_candidate；完整父链、源码/runtime与终局必须有效。保留现有函数的严格检查，不另写一个宽松的最终文件存在判断。
2. 保存规范JSON准入身份：生成run ID/文件SHA、唯一terminal checkpoint/prefix、末段父assessment身份/文件SHA、candidate文件SHA/source ID/logical graph SHA、完整母本来源/文件/图摘要、诊断文件与完整机械向量SHA、固定协议/实际计数、生成源/runtime及本工厂来源。字段必须来自已核对证据，路径仅定位。
3. 限定transform=R、generation_run_id/checkpoint_id/terminal与父证据相同，完整维度166700/25582938/124177617/101自环、实际3377/1314选择；拒绝W/P、generation_probe_only、中途或失败终局。不得将调用者传入的字典视为已验证凭证。
4. 轻量层仅JSON、小NPZ元数据/头和流式文件摘要，不导入Torch、加载全矩阵或在无预算父进程执行神经运算。规范自身身份校验不替代重读实际生成证据。

## F2 受监督worker内的构造

1. 工厂重新执行准入核对，调用真实G loader/map构造精确LaminaTonicLIF(mode=fixed)。在任何reset、initial_state、预处理、step、forward或autograd计算前完成下列安装；仅返回完整初始化且已校验的对象，不提供运行中换图公共API。
2. 在构建R传播表示前释放G的_forward_csr、base_weights和base_weights_adjoint及所有临时视图引用，避免SciPy视图保留旧Torch大块存储。G的非图初始化可以复用，不能先在G上产生神经状态再把状态标签改成R。
3. 用已验证母本与load_research_control读取真正终局R，重新核对准入/candidate实际来源及全图机械不变量。复用_graph形成只读整数TrainingGraph，不裁节点、弱边、自环、注释缺失或不可达细胞。尽早释放不再需要的大图副本；不能假设峰值等同G，实际构造成本留待另登记探针。
4. 研究graph身份保持role=research_control，reference_source_id/reference_graph_sha256/reference_provenance明确仍指G母本；另加research节绑定实际R来源、transform、生成准入和工厂源。raw_ordered_counts_sha256由R实际数组算出，training_graph_id重算。
5. 正向Torch CSR(post,pre)、伴随Torch CSR(pre,post)及共享正向存储的SciPy csr_array都从真实R构建；严格沿用先转目标浮点dtype、再乘weight_mv的运算次序。完整原始integer CSC保持只读。
6. 分块核对所有CSR指针/索引/值与实际R矩阵及其转置；不能用一个向量乘结果代替完整结构验证。SciPy三个数组与正向Torch的shape/dtype/指针共享、NumPy视图只读。检查不创建完整稠密矩阵或逐边Python对象表。
7. group_index、input_mask/input_body_ids、tonic_group_index及固定rho逐项对完整R元数据验证。复用_build_policy_map生成全部输入/DN，核对G/R body顺序、类型分组、输入/DN集合及空间编码数值一致。群组身份不能通过手工替换ID冒充验证。
8. 更新brain.graph和model真实嵌套身份/工厂源，再以role=research_control调用NeuralPolicy._initialize；将工厂来源和准入绑定到policy并重算policy_id。返回前重新验证关键来源/缓存绑定；失败不得暴露半安装对象。原模型/策略函数不被替换。

## F3 身份隔离和共同数值契约

- model、map、spatial、drive、policy和后续state身份按真实图自然变化，不强行等于G。G/R同维度不能互用state；原LIF/PolicyState校验必须拒绝错误身份。
- 提供严格allowlist的共同计算契约，用于后续配对核对：动力学配置/后端/精度及源、背景类型/成员/rho、完整类型分组、输入/DN body IDs/索引、空间分配数值及种子、相位初始化配置、相机/时钟、特征顺序/归一化、动作轴/范围、读出形状及初始数组摘要。工厂适配来源与数值核心来源分开记录。
- 只允许真实图、准入及其派生身份不同；不使用任意递归删graph字段后的宽松比较。readout1703、drive1702、空间1701与当前G协议相同的配置必须得到相同初始数值；公共配置仍需严格验证，不自动选种子。
- 旧full_reference加载器、参数/训练存档入口继续拒绝research_control。新入口名称和说明必须明确离线研究，不增加硬件接线或自动fallback。既有NeuralController只校验NeuralPolicy类型，不是研究角色准入器；本批不声称任意内存控制器都会拒绝研究policy，也不改动该绑定源码。当前设备路径未接入该新研究工厂。

## 软件验收标准

- [x] AC-F1：合法最终父链准入；中途/探针/W/P/失败父终态/源变更/文件替换/伪造准入均拒绝，父入口不导入Torch或读全矩阵。
- [x] AC-F2：数值夹具上真正安装不同连接，完整counts/正向/伴随/SciPy共享缓存与独立矩阵参考一致；实际forward及all-cell adjoint结果符合已声明数值契约，排除只改身份、旧forward或旧adjoint的反例。
- [x] AC-F3：G安装同图的数值兼容与R不同图效应分开；保持所有输入/输出/分组/背景及共同数值契约，错误body顺序、索引/符号/转置/dtype/缓存被拒绝。
- [x] AC-F4：全新policy/state真实研究身份；G/R错配state与旧存档/full_reference加载器拒绝，构造失败不返回部分可用策略；不允许带旧状态换图。不把NeuralController的类型校验误报为研究角色隔离。
- [x] AC-F5：29/13绑定来源字节保持，只有必要新模块、相关检查和说明；没有新的训练/评估/存档版本或无关清理。

软件检查风险为S3（完整计算缓存和身份准入）。只检查新增边界及直接相关拒绝行为，不重复全项目、旧37个生成器检查或旧完整图模型。小电路明确为numerical_test_only，不能冒充完整生物模型/正式终局；准入元数据夹具只证明软件拒绝边界，不冒称真实R证据。

## 实际运行门槛及证据边界

本子规格只授权软件实施与验收。最终R尚未生成，首次实际构造/前向须另立受监督协议，固定预算、相同初始化、预选输入、G既有证据锚点及比较规则，经新的独立协议审查后执行。正式R神经结果之前，还须固定单种子G/R研究主指标、160窗配对训练预算及失败处理，不能看R结果再决定。

实际时仍保留1GiB额外预留、申请6GiB、有效min(6GiB,available−1GiB)、600s/0.1s，同一完整图作业锁；所有矩阵构造在worker，Noria/Docker不动。软件PASS不代表完整图构造成本、学习收益、充分混合或AC6完成。

## 待确认事项

无需重复用户已确认的目标/算力/资源偏好。独立设计审查若发现上述受控首次构造方案与现有核心契约矛盾，返回具体问题后重新设计，不通过放松角色或来源校验解决。实际运行/研究持久化与比较登记均明确留给后续子批。
