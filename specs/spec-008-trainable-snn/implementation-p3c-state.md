# Implementation Report — P3c1 完整训练状态存档

## Summary

按 Ready [P3c1](p3c-resumable-state.md) 新增完整训练边界存档及恢复。支持 joint 与 frozen_brain 固定 Adam 模式；封存全部参数、实际命名 Adam 矩、完整 PolicyState/延迟队列/感觉相位、上下文和游标。复用已有部署参数校验，不修改任何 P3b 已登记来源。未运行完整图、长轨迹、设备或服务，未 commit。

## Files Changed

- `src/cnsbridge/learning/training_optimizer.py`：固定两模式、参数名称绑定、CPU float32 Adam 配置/实际矩校验及快照。
- `src/cnsbridge/learning/training_checkpoint.py`：完整状态序列化、上下文和游标校验、文件/源码/运行时身份、封存、只读解析、原子恢复与完整参考图生产加载。
- `tests/test_training_checkpoint.py`：两模式无损续跑、损坏/边界/角色/绑定/不完整档检查。
- `docs/TRAINING_CHECKPOINT.md`：固定 API、上下文字段、流程、边界及证据。
- 本实施报告。

## Important Decisions

- API 固定为 `make_training_optimizer(policy, mode)`；`save_training_checkpoint(policy, optimizer, state, directory, *, mode, context)`；`read_training_checkpoint(directory, *, expected)`；`restore_training_checkpoint(policy, optimizer, directory, *, expected)`；`load_training_checkpoint(directory, reference_path, *, expected)`。
- 只读返回 TrainingArchive：metadata、parameter_metadata、parameters、optimizer_arrays、完整 state；context 属性返回独立字典。恢复到已有对象返回 `(state, context)`；生产加载返回 `(policy, optimizer, state, context)`。
- 新 schema `cnsbridge.training_checkpoint.v1` 与 purpose `detached_two_frame_tbptt_continuation`。固定 nested `policy/` 复用部署参数存档；根 `training.npz` 保存状态与实际 Adam，不使用 pickle、不复制图矩阵。最后写根 manifest；不覆盖已有目录，部分写入失败不自动删除。
- 所有参数梯度必须 None，神经张量必须无 autograd 历史。仅接受正整数实际 step，且所有活动参数 step 等于 context.optimizer_steps。完整队列与感觉相位保留。状态是更新前参数计算、更新后携带的 TBPTT 边界，不伪装成重演结果。
- optimizer 使用实际对象参数名称/顺序绑定，额外或重复状态拒绝；恢复先验证所有字段并准备张量，再复制参数与替换 optimizer.state，不调用 initial_state/precondition。
- 显式 expected 四摘要核对 run/protocol/dataset 身份。声明场景用于独立推导窗口数和状态时钟；真实场景与数据集、真实日志内容、父链连续性仍需后续 runner 对照登记和文件，本模块不把摘要字符串视为执行证明。
- 源码、运行时版本和线程数必须相同；无随机训练操作，感觉相位完整恢复。未来随机算子/配置变化需新契约。

## Validation

- Risk Level: **S3**，新跨进程持久化、优化器及完整递归状态；范围仅本模块与直接接口。
- 命令：`PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_training_checkpoint.py -q`。
- 终态：session `45861` 已收取 exit 0；**51 passed, 1 warning in 2.27s**。警告为既有 PyTorch sparse CSR beta，不是本次失败。
- 两模式各自使用两次有效更新的显式数值电路；分支 B 首步封存后销毁 policy/optimizer/state，通过 weakref 确认旧 policy 已释放，再重建恢复继续。逐个参数、各命名 Adam 矩、全部神经/驱动状态、动作、特征和下一游标严格相等，后续两个完整封存 manifest 也相同。
- 非零延迟队列、18 槽顺序、非零感觉相位/事件计数和有效时钟真实参与；冻结模式 theta 不变，读出改变，脑状态推进。
- 39 个损坏档情形包含晚期二阶矩负值等；在带真实 Adam 状态的目标上拒绝，并确认模型/优化器都未变化。另覆盖保存前边界拒绝、未知绑定、模式差异、源码/运行时/线程、非完整目录、不可覆盖、生产数值档拒绝、部署档不能续训、读取和恢复不分配完整图/不预处理。
- 已通过后停止测试，没有运行全套测试、旧检查或生产图。

## Known Limitations

- 当前仅软件/小电路结果；未实际运行生产完整图续跑，不关闭 AC3–AC6。
- 仅两帧完整窗口、CPU float32 固定 Adam 和相同运行环境；不支持零步档、随机训练或跨版本迁移。
- 没有全轨迹 runner、闭环评估续跑或设备入口；生产加载依旧验证 full_reference。既有 P3b Adam NPZ 与部署参数档无法直接升级为完整续训档。
- manifest-last 防止接受半写目录；未实现跨文件文件系统事务或掉电恢复。

## Remaining Questions

无实现阻断项。交协调者安排新的独立验收；未将实施检查自称最终验收。
