# Acceptance — P3c1 完整训练状态存档

## Result

**PASS**。本验收者未参与实施，独立阅读 Ready [P3c1 规格](p3c-resumable-state.md)、[实施报告](implementation-p3c-state.md)、两份新增实现、定向测试及直接依赖接口。通过范围是完整训练边界的保存、校验和恢复软件；未运行生产完整图，不代表长场景训练、真实跨进程恢复或学习收益已经验证，AC3–AC6 不因此关闭。

## Spec Coverage

- **固定优化器与模式**：`training_optimizer.py` 按稳定参数名严格绑定 joint 的 theta / readout 两组和 frozen_brain 的 readout 单组，核对 CPU float32、requires_grad、实际 Adam 类型、完整组字段、选项、顺序及多余状态。实际 step 为正整数值的 float32 标量并匹配游标；两个矩保持同形 float32、有限，二阶矩非负，零二阶矩合法。冻结模式不伪造 theta 优化器状态。
- **完整活动状态**：保存全节点 v、g、int64 refractory_until，按顺序叠存全部 delay 槽；保存 float64 感觉 phase、int64 event_counts、全部模型/感觉/策略身份和时钟。读取按策略身份推导 shape、队列长度和期望时钟，恢复不调用 initial_state/precondition、不重演已完成窗口。
- **更新边界与游标**：保存前复用现有策略状态校验，拒绝带 autograd 历史的状态或非 None 参数梯度。next_frame 必须为当前场景的正偶数窗口边界、等于 frame_index；brain/drive 时钟由 precondition + next_frame × frame_steps 独立推导。completed_windows 由 epoch、前序场景长度和当前游标推导；optimizer_steps 允许小于窗口数但必须正且一致。场景末保留本场景末状态，未暗中重置到下一场景。
- **身份与存储格式**：复用 nested 部署参数存档的模型、图、映射、动作语义和源码校验；新根 manifest 绑定文件 SHA、数值数组声明、上下文、实现来源、运行时/线程及确定性契约。固定文件名、无 pickle、不复制图矩阵、拒绝已有目录、根 manifest 最后写入。重复 JSON/NPZ 字段、未知/缺失字段及不完整目录拒绝。
- **恢复失败边界**：读取先完成全部存档语义和外部 expected 身份核对；`_restore` 随后校验目标 policy/optimizer、目标梯度和活动状态，准备全部参数及 Adam 张量后才修改模型和 optimizer.state。损坏档测试包含最后一个参数的负二阶矩，拒绝后既有参数和实际 Adam 状态逐项保持原值。
- **角色与 API**：只读接口不分配图/模型；生产加载明确拒绝 numerical_test_only，经过 `NeuralPolicy.from_reference` 重新验证完整参考图，并匹配保存身份。部署参数档不能被当作续训档。保存/读取/恢复/生产加载及两个模式的使用说明完整，旧 P3b `adam_state.npz` 缺少状态和游标的限制说明准确。
- **证据边界**：实际数据集场景真实性、真实日志前缀和父存档链连续性由后续 runner 与外部登记/文件核对，文档明确此模块的摘要格式和内部关系检查不等于这些事实已经成立。没有扩张本批范围去修改既有绑定源码或新增训练 runner。

## Validation

- Risk Level：**S3**，涉及持久化、完整递归状态和优化器恢复；验收范围仅新增模块及直接依赖。
- Evidence Reviewed：实施者定向命令 `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_training_checkpoint.py -q`，**51 passed, 1 warning in 2.27s**；实施者已收取 session `45861` exit 0，协调者已读取其完整报告。警告为既有 PyTorch sparse CSR beta。独立审查测试实现及其断言后，未重复已通过检查。
- 两种模式均实际执行两次有效更新，对比连续分支与首步封存、销毁 policy/optimizer/state、weakref 确认旧 policy 释放、重建恢复再更新的分支。逐个参数、实际命名 Adam、全部活动状态、动作/特征和下一游标精确相等，第二个完整 manifest 也相等；不是仅数组 roundtrip。测试在同一 OS 进程中重建对象，未宣称已另起进程完成生产续跑。
- 测试明确要求 18 槽、非零待到达事件、非零感觉相位/事件计数和有效时钟；冻结脑多步 theta 不变而读出改变，状态继续推进。首档在后续更新后再次读取仍保持原值。
- 独立核对 39 种坏档、7 种保存前边界/参数绑定错误、角色隔离、不可覆盖、epoch/场景游标推导，以及模拟写入失败留下不可恢复的未封存目录。没有发现需要新增数值运行才能判定的具体缺口。
- 本验收者仅新增此报告；未运行完整图、硬件、服务、付费任务或全套检查，未操作 Noria/Docker，未改变 1 GiB 系统预留，未提交 Git。

## Issues

无阻断问题，无需返工。协调者对说明中旧 P3b 文件名的 S0 更正已纳入审查；未影响实现或定向检查结论。

## Regression Risks

- 当前契约刻意限制 CPU float32、固定 Adam、相同源码/运行时/线程、两帧窗口及至少一次更新；不支持任意旧档迁移、零步续训或新增随机算子。
- manifest-last 用于拒绝未封存目录，不是文件系统事务或掉电恢复保证。
- 当前软件证据不量化完整图存档/恢复内存、进程重启成本或长时稳定性；需要下一批预登记的受监督实际试验。完整图学习和独立冻结脑收益比较仍未完成。

## Required Rework

无。本批满足规格后停止验证，交协调者更新状态并按既有授权提交。
