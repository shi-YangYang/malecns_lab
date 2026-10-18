# Implementation Report — P3a R1

## Summary

按 [R1 返工规格](rework-p3a-r1.md) 修复首次独立 [FAIL](acceptance-p3a.md) 的两处缺口。原实施报告与失败记录保持不变；本报告不代替新的独立验收。

## Files Changed

- `src/cnsbridge/learning/neural_data.py`：加载时使用登记的 scene/camera/dt/tau 建立轻量 `FollowEnvironment`，逐帧核对像素、当前 truth 和连续时间；仅用原有检查已验证的执行动作推进，再核对 next_truth、失败与最终场景边界。保留同帧教师、行为、标签、掩码、数组、来源、split 和 hash 检查。
- `src/cnsbridge/learning/policy_checkpoint.py`：严格校验模型键集合、gain/gradient/event_order 等固定动力学；固定 tonic 校验完整键集合、公式、范围、积分、假设说明、类型、成员数量关系和 SHA256 格式。恢复全部预验证后写参数的顺序保持不变。
- `tests/test_neural_data.py`：新增实际 val 轨迹的完整像素/记录/标签移入 train 后重封存的拒绝测试，以及当前/末帧下一状态与实际环境不一致的拒绝测试。
- `tests/test_policy_checkpoint.py`：新增 19 类重算全部相关嵌套 ID 后的不兼容动力学/成员声明拒绝检查，直接调用轻量读取并验证恢复失败不改变目标参数。
- `docs/NEURAL_TRAINING_DATA.md`、`docs/POLICY_CHECKPOINT.md`：补充环境重放及固定语义检查的边界。
- 本报告。

## Important Decisions

- 文件摘要和数据内部自洽不足以证明训练场景来源；验证来源使用已有确定性环境，不加载神经模型、不修改场景定义或放宽 split。
- canonical ID 只证明内容绑定；固定动力学声明必须符合当前版本契约。成员摘要在轻量读取阶段只验证格式与数量/类型关系，实际图成员仍由生产参考加载复核。
- 未改既有脑核心、策略、环境、空间驱动或旧 checkpoint；没有启动完整图、硬件或服务动作。固定背景不增加可训练参数。

## Validation

- Risk Level: **S2**，针对 R1 两个相关校验缺口；只执行受影响的两个新测试文件。
- Checks Performed: `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_neural_data.py tests/test_policy_checkpoint.py -q`
- Result: **71 passed，1.25 s**；唯一警告为既有 PyTorch 稀疏 CSR beta 提示。原有 49 项及本次 22 项均在这次受影响范围检查通过后停止，未重复旧核心或完整套件。
- 已检查本次修改范围；原 FAIL 和原实施报告未修改。未生成正式四场景数据，未加载完整图或训练实测。

## Known Limitations

本批证明软件拒绝路径及数值 fixture 参数往返，不是完整脑训练或跟随成功。部署参数格式仍不包含优化器、隐藏状态或训练游标；来源代码变化后的数据兼容性保持严格。

## Remaining Questions

无。没有 commit，交协调者安排新的独立验收。
