# P3c1 — 完整神经状态与优化器续训存档

状态：软件已完成并获新的独立验收PASS，实施报告已全文读取处理，51项定向检查通过，无返工项。验收覆盖同进程销毁/重建后的连续更新一致性，真实跨进程完整图续跑尚待下一批。本工具不改变当前模型、数据或学习率；P3b r2的[实际结果](p3b-run-20261019-r2.md)内部贡献本步为阴性，不因存档工具完成提升为学习收益。下文为实施契约。

## 背景与目标

现有 `policy_checkpoint` 只保存部署参数；P3b 的 Adam NPZ 也只是一次更新的证据，缺少完整隐藏状态和游标，不能直接称为续训档。两帧前向约19s、反向阶段约77s是当前实测；长场景可能超过单段600s监督预算，需要在明确的窗口边界保存并恢复，而不是每次重置果蝇网络或重算整个前缀。

本批提供完整、版本化的神经训练存档 API，并通过有状态数值电路证明“连续更新”与“保存、销毁、重建、恢复后继续更新”一致。生产构造仍只加载完整参考图；数值电路仅用于独立标记的测试入口。

## 范围与非目标

在 `src/cnsbridge/learning/` 新增有明确职责的训练状态存档模块，复用现有参数存档与 PolicyState/LIFState/SpatialDriveState，避免复制其模型身份和参数语义校验。允许按优化器契约与序列化职责拆为两个模块，不建立泛化训练框架。

不修改脑核心、空间驱动、NeuralPolicy、教师数据、已有部署 checkpoint 或 P3b 已登记来源；不重写旧资产，不新增 fallback。当前不实现全轨迹 runner、闭环评估续跑、拓扑对照或设备入口；这些将在此存档接口稳定后独立实施。只读数据/参数加载仍不构造完整图。

## 固定优化器与模式

支持两种明确模式，为后续同条件独立对照保留接口：

- `joint`：theta 一组 lr=0.001，weight+bias 一组 lr=0.0001；全部三参数有 Adam 状态。
- `frozen_brain`：theta.requires_grad=False 且不进入优化器；weight+bias 一组 lr=0.0001。仍保存全部脑参数与完整活动状态，不能用缓存特征代替逐帧脑计算。

两模式 Adam 均采用 betas=(0.9,0.999)、eps=1e−8、weight_decay=0、amsgrad=False、foreach=False、fused=False、CPU float32。固定背景 rho=0 不训练。保存实际配置与稳定参数名称，拒绝额外参数、不同配置、重复绑定或未知 mode；禁止依赖跨进程不稳定的 PyTorch parameter ID 作为存档身份。此处仅固定序列化契约，损失和分组裁剪仍由后续训练协议登记。

第一版只封存完成至少一次有效更新之后的边界，三个（或冻结模式的两个）参数的 step 必须相同，实际 float32 scalar、正整数值，与游标中的 optimizer_steps 相等；两个矩为同形 float32，有限，二阶矩非负。零值二阶矩合法，保留实测下溢语义。初始零步通过已登记的初始化重建，不伪造空矩续训档。

## 完整存档内容

以新目录、JSON 和数值 NPZ 保存，`allow_pickle=False`；不序列化可执行对象，不复制巨型图矩阵。

1. 复用现有完整部署参数存档：theta、读出 weight/bias、完整 policy/model/graph/感觉/时钟/源版本身份及参数 checkpoint ID。
2. 命名 Adam 状态和上述实际参数组配置，包含各参数 step、exp_avg、exp_avg_sq；冻结模式无 theta 优化器状态。
3. 完整 PolicyState：全节点 v、g、int64 refractory_until、**全部按顺序的延迟队列**；brain step/model_id；drive float64 phase、int64 event_counts、drive/spatial ID、dt 与 step；frame_index/policy_id。原 dtype、shape、值与顺序保留，不量化、截断或重新初始化。
4. 续训上下文：training_run_id、protocol_id、dataset_id 和 dataset manifest SHA；有序场景清单及各场景帧数；epoch、scene_index/scene_id、next_frame、completed_windows、optimizer_steps；parent checkpoint ID（首个边界为 null）、已封存日志前缀摘要。身份值必须为规范摘要或明确字段，禁止自由字符串代替一致性校验。
5. 当前存档 schema/purpose、必要实现源码和运行时版本、线程设置、每个文件字节 SHA、每个数组 dtype/shape、完整 manifest 的 canonical checkpoint ID。purpose 与部署参数存档分开。记录确定性顺序且不使用随机采样/噪声/dropout；感觉相位完整保存，不能将其视为无需恢复的随机数状态。后续若添加随机算子必须升级协议和恢复契约。

根 manifest 最后写入；不覆盖既有目录。中途失败留下的无最终 manifest 目录为不完整，不可恢复，也不自动删除。全部路径使用固定本地文件名，不从外部 manifest 拼接任意路径。未知字段/漏字段、重复 NPZ 字段、dtype/shape/摘要/源码/模式不匹配要拒绝。

## 状态、游标与更新边界

保存顺序是：窗口内连续前向 → 动作损失反向 → 已登记裁剪 → Adam step → 完整 state.detach() → zero_grad(set_to_none=True) → 封存。输入 state 必须没有 autograd 历史，所有参数梯度必须为 None，防止把部分反向或半更新当成已完成。序列化必须先做验证再开始写文件。

状态由更新前参数计算，随后参数已更新再携带该状态进入下一窗口，这是明确的 TBPTT 近似，不伪装成更新后参数重演的状态。`detach` 只截断梯度历史，不清空神经电位、突触电流、待到达事件、感觉相位或时钟；仅在新场景边界显式 reset/precondition。

第一版支持固定两帧完整窗口且场景长度为正偶数。`next_frame` 是当前场景下一未处理帧索引，范围为 [2, scene_frame_count] 的偶数；必须等于 state.frame_index。brain 与 drive step 均严格为 precondition_steps + next_frame * frame_steps。场景恰好结束时保存该场景末状态及 next_frame==count，后续 runner 才执行新场景重置，不把“新场景索引+旧隐藏状态”混成一个存档。

`completed_windows` 必须由 epoch、之前场景长度和当前 next_frame 独立推导一致；optimizer_steps 为正且不大于 completed_windows。之后 runner 若支持无效标签窗口，应推进状态而跳过更新，解释两者差异；本模块不得凭 optimizer_steps 推断全部帧有效。scene_id 与有序清单匹配，epoch/计数类型为非布尔整数，所有上下文有限且身份完整。日志前缀摘要只绑定已写日志，不声称单靠摘要证明训练真的执行。

## 恢复接口与失败行为

提供保存、只读解析、恢复到已有 policy/optimizer，以及从完整参考图生产加载的清晰接口；具体函数名和返回类型由实施者在报告和文档中固定。恢复返回完整 PolicyState 和游标，不隐式调用 initial_state/precondition、不再次应用上一个窗口、不推进样本。

恢复必须先验证所有文件、nested 参数存档、身份、矩、状态、时钟、上下文以及目标 optimizer 模式/配置/参数名称，准备好所有张量后才修改任一模型/优化器对象。坏存档不得留下部分恢复的 theta、读出或优化器。调用方提供期望 run/protocol/dataset 身份进行交叉核对，不能只接受存档自报身份；生产入口拒绝 numerical_test_only 且重新验证完整参考图。不得同时常驻两份生产模型以做恢复对照。

只读 API 不实例化图或模型；允许训练模块依赖 Torch，轻量监督父进程不得导入它来检查终态。下一批 runner 可从 JSON/NumPy 汇总必要终态，不在本批为假想父进程提前扩建重复校验框架。

## 验证与验收

风险 **S3**：涉及跨进程持久化、优化器和完整递归状态；范围仅新模块及其直接接口，不重跑完整项目或旧全图实验。

- 数值电路中，至少连续两次有效更新；分叉 A 连续运行，B 在首个更新边界封存、释放对象、重建并恢复后继续。比较每个参数、命名 Adam 状态、全部神经/驱动状态、动作/特征与游标。同环境要求精确相等；如出现具体数值差异先解释和修复，不擅自扩大容差。
- 两模式分别验证；冻结脑在多次更新后 theta 不变，读出可更新，脑动态仍随帧推进。
- 真实延迟队列非空、感觉相位非零、多个时钟有效，避免“全零状态恢复”虚假通过；检查保存后原对象后续变化不污染已封存值。
- 损坏/缺失文件、参数组/step/矩/相位/队列/游标/身份错误、源不匹配和未封存目录拒绝；至少一个晚期错误证明 policy 和 optimizer 未部分修改。
- 部署参数档不能作为续训档；数值档不能进入生产加载；只读接口不构造图；文档明确旧 P3b 档不能直接续训。

单一实施者完成后返回报告，新的独立验收者检查规格覆盖与定向验证；通过后逐步 commit。当前不运行生产完整图。下一阶段先登记短的连续多窗口及跨进程恢复实测，再决定完整场景预算；不因小电路一致或序列化成功勾选 AC3–AC6。
