# P2e — 有状态完整图三轴神经策略接口

状态：Done（软件接口），[实施报告](implementation-p2e.md)与新独立[PASS](acceptance-p2e.md)已读取处理，28项定向检查通过，随本状态更新提交。生产训练窗口与实际学习协议另行登记；不运行设备或宣布控制成功。

## 目标与非目标

将既有 target_mask、完整R1–R6空间驱动、TrainableLIF/LaminaTonicLIF、全部DN与小容量三轴读出连成可微路径，供下一批任务训练使用。环境真值、检测得到的位置/距离误差不进入动作读出。当前无动作读出接口，旧冻结脑/人工映射不能代替此路径。

本批先实现接口、数值契约与小电路软件检查；不进行任务训练、规范化数据拟合、checkpoint扩展、拓扑对照或硬件替换。这些仍是完整目标的后续要求。持续输入实验与本批可独立推进：不得修改其登记的脑核心、空间驱动、诊断或元数据源码。

## 模型与感觉输入

- 新 `learning/neural_policy.py` 提供 nn.Module 策略。生产工厂由已验证完整图加载构造模型、完整policy neuron map及spatial map；固定背景选择显式为 `fixed_lamina_8mv` 或 `zero_background`，默认不隐含。固定模式rho保持0/8mV，不让本批偷偷解冻背景。保留全部theta候选可训练参数。
- 软件数值工厂/构造入口可接受明确numerical_test_only的已有测试图和映射，但身份必须保留该角色，不能进入生产工厂。生产工厂无裁图、假图或缺失回退。
- 感觉输入只允许BGR frame、已登记TargetCamera、既有12×16占用mask与SpatialDrive；不能把FollowTruth、TargetObservation中的x/y/distance、教师动作或场景ID送到读出。mask为固定工程编码，不训练像素前端。
- 绑定model身份、神经选择、空间/驱动身份、相机、dtype、config、特征顺序/尺度、源码SHA形成独立policy ID。模型参数值（theta/readout）不是结构身份；未来参数持久化由新checkpoint另行绑定。保存现有来源与工程空间映射边界。

## 显式状态与时间

- 配置必须显式提供 `frame_ms` 和 `precondition_ms`，不指定未实测默认窗口。frame_ms>0、precondition_ms>=0，均有限并整除模型dt；frame步数必须能等分两段。控制周期单独登记为 `control_dt_s`，与神经模拟时间、实际墙钟三者分开；不把人为时钟比称为生理还原。
- PolicyState包含完整LIFState、不可变SpatialDriveState、frame_index及policy ID。神经与驱动步号一致，等于预处理步数+frame_index*frame_steps。拒绝跨模型/映射/配置/相机/精度或时钟状态，不静默重置。
- `initial_state()`只在场景边界调用：重置驱动和脑，在no_grad下用空白mask完成显式预处理，保留全部v/g、不应期、pending、phase/count与clock。构造策略时不能隐式运行预处理。预处理不称稳态，也不纳入本批反向图。
- `forward_tensor(frame, state)`输入显式状态，保存/恢复驱动相位后推进恰好frame_steps；整个推进和读出可保留autograd，不调用BodyVelocity/float/item/detach切断神经特征。返回三轴SI Tensor、原始神经特征和新完整状态。外部调用者选择torch.no_grad或训练模式；不得每画面reset。
- `state.detach()`用于TBPTT：仅切断完整LIF状态梯度，保持所有值、延迟队列、驱动相位/计数、步号和frame_index，不重置或重新预处理。对输入frame/state先验证再推进；失败不得破坏传入状态。

## 神经特征与读出

- 所有已选DN，保持升序body ID，不能事后挑响应最大的细胞。对一个frame窗的前半/后半神经步分别计算post-reset膜电位均值；连接为 `[first_half_DN, second_half_DN]`，维度2*N_DN。两半时间结构明确，避免把各条件相同的全窗总脉冲率当特征。
- 固定归一化为 `(mean_v-v_rest)/(v_threshold-v_rest)`。这是已登记的物理尺度，不从验证/测试场景拟合统计量；未来若增加训练集归一化须新身份/规格。
- 一个线性层（2*N_DN→3，含bias），之后tanh并乘轴范围 `[0.3,0.2,0.5]`，轴顺序vx/vz/wz，单位m/s,m/s,rad/s，FLU正yaw左。没有人工方向/阈值映射，无视觉跳接，也不把尚未训练的符号当正确行为。
- 权重用独立CPU Generator的显式seed、正态std `0.01/sqrt(2*N_DN)` 初始化，bias为0；不可置零权重导致首步内部梯度路径必然关闭。不得改变全局随机状态。训练参数集合严格包含brain.theta和readout weight/bias，固定rho不训练。
- 本批没有训练循环；软件验收须证明动作损失可向非零内部theta梯度传播，并完成一个明确测试电路的联合更新/状态detach后续推进。小电路不是完整图学习证据。

## 环境适配与日志边界

提供独立controller/callable适配现有 `FollowPolicyInput -> BodyVelocity`，在部署边界no_grad并显式转换Python浮点；其他轴为0，使用policy登记camera，拥有每场景显式reset的运行状态。传入observation的误差/距离不能影响原始动作；已有rollout外部失去目标保护及动作记录保持原样，不在神经策略中偷偷复制教师规则。神经/控制时钟分别可读取，首次运行或重启场景须显式状态初始化，不以旧状态继续新场景。

## 范围、验证与交付

一个实施Agent负责新 `learning/neural_policy.py`、`tests/test_neural_policy.py`、`docs/NEURAL_POLICY.md` 和 implementation-p2e.md。如文件承担超过一种明确职责，可在learning下拆读出/状态，但不新增泛化框架、不修改既有脑/诊断/环境/驱动模块、不提供无用兼容层。README/治理由协调者维护。

风险S3（核心跨模块可微/状态接口）。定向检查本接口：手工神经均值参考和固定线性代数、分帧与连续推进的状态/相位一致、detach保持值且切断梯度、mask仅驱动注册输入、全部DN与升序绑定、camera/clock/角色/dtype拒绝、独立初始化随机状态、无视觉真值跳接、完整theta梯度与联合更新。以显式数值fixture运行，不能启动完整图或硬件，不重复旧核心/诊断套件。

新的独立验收后提交。实际完整图策略/动作学习、checkpoint、闭环评估和贡献对照另批按已授权目标继续。

- [x] A：完整路径、来源/角色与全部DN特征明确，无真值/视觉直连。
- [x] B：状态/时间/梯度/读出数值及环境边界定向检查通过。
- [x] C：新独立软件验收、文档完成，随本状态更新commit，未夸大为完整图策略学习。
