# Implementation Report — P3a

## Summary

已按 Ready 的 [P3a](p3a-data-checkpoint.md) 新增完整教师序列采集/校验工具，以及完整神经策略参数存档。未修改已验收神经核心、策略、感觉映射、环境、rollout 或旧 gain-only checkpoint。

## Files Changed

- `src/cnsbridge/learning/neural_data.py`：固定四个登记 train 场景；复用视觉行为策略与 record_frames；分离原始真值教师、限幅监督与实际动作；保留无效帧；最后封存、严格读取与 CLI。
- `src/cnsbridge/learning/policy_checkpoint.py`：独立完整策略参数格式；轻量身份验证、全部参数预验证后恢复、完整参考生产加载。
- `tests/test_neural_data.py`：短场景数值 fixture，未生成生产采集证据。
- `tests/test_policy_checkpoint.py`：复用现有显式数值电路，未新增生产合成脑路径。
- `docs/NEURAL_TRAINING_DATA.md`、`docs/POLICY_CHECKPOINT.md`：数据/参数用途、命令、来源和不支持训练中途恢复的边界。
- 本报告。

## Important Decisions

- 全部当前登记 train 场景必须完整收集；场景定义、split、源码与文件字节共同校验，不能通过重算清单 hash 把非训练场景混入。
- 无效像素帧保留教师标签和时间位置，后续只屏蔽损失；当前数据工具不调用神经策略。
- 参数存档只含 theta 和线性读出；固定背景与完整神经/感觉/动作语义绑定到 policy identity。它不是 Adam/隐藏状态/采样游标恢复格式。
- 轻量读取重建输入映射/驱动身份但不构造脑图；生产恢复仍走未放宽的完整参考入口。所有恢复校验先于参数变更。
- 当前 Mac 1 GiB 额外系统预留与 Noria/Docker 保持运行约束没有变化，本批不运行完整图。

## Validation

- Risk Level: S3，依据：持久化、训练 split 隔离及监督来源；只验证本批新模块。
- Checks Performed: `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_neural_data.py tests/test_policy_checkpoint.py -q`
- Result: **49 passed，1.20 s**；仅 PyTorch 既有稀疏 CSR beta 提示。
- 覆盖：独立同场景 rollout 对照、无效帧与限幅、字段/精度/shape/非有限值/时间/文件/来源/split/遗漏/失败/覆盖保护；float32/float64 与两种背景的联合更新后逐位参数往返、同输入输出一致、身份不匹配/畸形存档的失败不变性、数值角色生产拒绝、旧格式互拒、无隐式 initial_state/图加载。
- 未运行完整测试套件、完整图、模型训练实验或硬件。通过后停止测试；最终验收由新的独立 Agent 负责。

## Known Limitations

只证明软件契约与数值 fixture 往返，不是实际四场景采集报告、完整图动作损失更新或闭环任务成功。当前存档不保存训练中途状态；后续训练协议仍需明确优化器、TBPTT、资源和评估窗口。

## Remaining Questions

本批无待决问题。没有提交，由协调者组织独立验收并提交。
