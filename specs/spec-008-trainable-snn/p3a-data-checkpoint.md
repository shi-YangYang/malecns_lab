# P3a — 教师序列与完整神经策略参数持久化

状态：Done（软件）。首次独立FAIL的两个校验缺口已由新实施者返工、新验收者[PASS](acceptance-p3a-r1.md)，两轮报告均已读取处理，71项受影响检查通过，随本状态更新提交。P2e接口已提交a2320b0；本批提供下一次完整图动作损失更新所需的数据及参数工具，实际采集另报，不把持久化当作训练成功。

## 目标与范围

新增 `learning/neural_data.py` 和 `learning/policy_checkpoint.py`，相应定向检查、使用说明和实施报告。保留既有gain-only checkpoint及冻结脑基线，不修改其格式或放宽完整参考图校验。不修改已验收神经核心、策略、空间驱动、环境或rollout；真实数据、历史日志、许可全部保留。

这是两个相关工具的同一实施批，由一个Agent串行完成。部署参数存档与训练中途恢复是不同格式：本批仅保存完整策略参数，不宣称已保存Adam/隐藏状态/进度；恢复训练由下一批明确优化器和TBPTT协议后实现，不能悄悄从零状态续跑。

## 教师数据契约

1. 复用现有 `scene_registry` 中全部四个 `train-sine-*` 场景，顺序按scene_id排序；只允许已登记train场景进入此训练数据集。显式相机、dt_s和response_tau_s，默认可沿用现有环境0.1s/0.25s，全部进入身份。场景不能改名或随机拆帧混入验证/测试。
2. 使用现有 `rollout` + `visual_teacher` 作为采集行为策略，通过 `record_frames` 保存同一时刻的无损BGR图像。行为策略显式传入该scene的desired_distance_m。rollout原始记录中真值教师动作是监督来源；不能把行为策略动作或保护后的停住冒充教师标签。
3. 按时间顺序每场景保存数值NPZ：`frames` uint8[N,H,W,3]、`time_s` float64[N]、`teacher_raw` float64[N,3]、`teacher_target` float64[N,3]、`executed_action` float64[N,3]、`valid` bool[N]；轴顺序vx/vz/wz、SI/FLU。teacher_target为逐帧truth_teacher输出经现有limit_follow_action限幅，不直接使用未限幅标签。保留原始rollout summary/trajectory以核对真值、介入和next_truth。
4. valid来自同帧detect_target。后续训练在无效帧继续推进状态，但不使用其动作模仿损失；不得把这些帧从时间序列删除。本批只生成数据，不运行神经模型。
5. 以manifest绑定场景完整定义/split、相机/时间、教师和行为来源、动作轴与上限、来源文件SHA、每个NPZ与summary/trajectory SHA、字段shape/dtype与实际/计划样本。保留每个场景原始failure/覆盖。必须完整收集计划步数且无环境failure，才标记数据ready供训练；失败保留证据，不静默补帧/跳过场景。
6. 写入新目录，不覆盖已有资产；manifest最后写入，loader只接受完整ready的已封存目录。读取用allow_pickle=False，验证schema/canonical ID/所有文件hash/严格数组字段/shape/dtype/有限值/时间轴/有效掩码/动作边界/场景注册与split/同一相机、源码身份，拒绝遗漏或不相容数据。调用方不能通过修改metadata并重算hash把非train场景混入训练。
7. 导出明确数据对象和CLI：`python -m cnsbridge.learning.neural_data --out <new-directory>`，生成现行四train场景；读取函数返回每个完整序列及绑定身份。读出网络仅拿frame；teacher与truth只留在训练器/评分一侧。

## 策略参数存档契约

1. 新schema `cnsbridge.neural_policy_checkpoint.v1`，JSON+非pickle数值NPZ。保存且仅保存 `brain.theta`、`readout.weight`、`readout.bias`（NPZ字段可为theta/readout_weight/readout_bias，映射固定）。固定rho、完整图、输入映射、特征归一化/尺度/时钟/相机都由完整policy.identity绑定，不能遗漏后将其当同一模型。
2. `save_policy_checkpoint(policy,directory,run_identity=...)` 创建新存档，先验证策略CPU/dtype/严格参数集合及finite，再写数组、metadata和checkpoint_id；已有目录报错，不覆盖历史。manifest最后封存。没有隐藏状态、优化器或随机游标，明确 `purpose=deployment_parameters_only`。
3. `read_policy_checkpoint` 不构造完整图，先验证schema、完整身份自洽、文件hash、严格字段/shape/dtype、finite、参数尺寸与policy声明相符，返回已验证数据。不能只信文件hash而忽略metadata语义。
4. `restore_policy_checkpoint(policy,directory)` 支持已创建的完整图或明确数值fixture。必须先校验**全部**策略身份与参数，全部成功后才改变任何参数；拒绝跨图/背景/角色/相机/时间/读出顺序/精度/源版本。失败不得只恢复了theta却损坏读出。策略结构policy_id不随学习值改变。
5. `load_policy_checkpoint(directory,reference_path)` 是生产入口：必须full_reference角色，按存档配置构造NeuralPolicy.from_reference并再次核对其全部身份，然后恢复参数。数值测试角色禁止进入此入口；不能放松现有生产loader来支持测试或研究重连。
6. 不自动调用initial_state或恢复运行controller；部署新场景仍须明确reset。旧gain-only存档被新loader拒绝，新存档亦不能被旧loader当作theta模型。可选run_identity必须是可JSON化有限数据并计入存档ID。

## 验证与交付

风险S3（持久化/数据来源与训练隔离）。只运行新工具定向检查，不重复旧模型/环境/诊断套件，不加载完整图或硬件。

- 教师序列：与同一完整短fixture场景的独立记录核对frame/time/标签/限幅/执行动作、无效帧保留、时间顺序、无真值输入神经接口；注册train split、source/shape/dtype/hash/遗漏/失败/非有限值拒绝，禁止覆盖旧输出。检查可用猴子补丁把注册场景缩短用于软件测试，但测试角色不得输出声称生产已跑的证据。
- 参数存档：数值电路联合更新后的所有参数逐位往返、恢复后同输入/完整起始状态的输出一致；故意错误相机/窗口/背景/dtype/source/role与畸形参数拒绝且目标完全不变；生产加载拒绝numerical_test_only；旧格式不混淆，不自动预处理。
- 数据采集不加载图、checkpoint轻量读取不加载图；通过后不追加全套检查。新独立验收PASS后逐步commit。

实施文件：上述两模块、`tests/test_neural_data.py`、`tests/test_policy_checkpoint.py`、`docs/NEURAL_TRAINING_DATA.md`、`docs/POLICY_CHECKPOINT.md`、`implementation-p3a.md`。角色/初始化可复用现有小电路测试fixture而不建立生产假脑路径。协调者维护本Spec/索引，验收者只写acceptance-p3a.md。

- [x] A：完整教师序列、来源/split/标签/时序契约与只读加载通过。
- [x] B：完整策略参数持久化及失败不变性通过，无旧格式混淆。
- [x] C：新独立验收通过，随本状态更新提交；实际采集/动作损失更新仍分别报告。
