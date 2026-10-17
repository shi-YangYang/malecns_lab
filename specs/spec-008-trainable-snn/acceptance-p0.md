# Acceptance — P0 可训练核心

## Result

PASS

2026-10-17，由未参与本批实现的独立验收 Agent 审查。本结论仅覆盖 [P0 子规格](p0-core.md) A–E，不代表 Spec 008 整体完成，不包含完整图内部更新、目标跟随训练、实时性或实飞效果。

## Spec Coverage

| 标准 | 结论与证据 |
|---|---|
| A：独立数值参考及事件语义 | PASS。核心解析积分与测试中的独立 2×2 矩阵指数一致；测试覆盖不同/相等时间常数、递归发放及抑制到达。另有手工时间表检查严格阈值、输入细胞无不应期、k+delay_steps 到达、不应期中 v/g 冻结且保存到达、k+ref_steps 恢复。文档明确这是 Shiu 方程启发模型，并未宣称复现 Brian2 调度。 |
| B：代理导数及全细胞反向 | PASS。二值前向、fast-sigmoid 反向公式与登记一致。CSR 自定义反向对所有突触前分量执行转置乘积；测试以独立 dense 代数和端到端静默细胞经过延迟队列的梯度分别核对，没有用硬阈值有限差分替代代理梯度验证。复位和不应期门停止梯度，递归脉冲保留梯度。 |
| C：增益、分组、状态及错误路径 | PASS。原始 type 的确定性排序和显式缺失组、初始增益 1、严格正有界增益、reset/detach/连续窗口均有检查。SGD 实际修改内部突触前类型增益，使下游状态改变；恢复 theta 后响应恢复。非法配置、输入形状/精度、未登记输入、非有限输入/状态/参数及跨模型状态有拒绝检查。 |
| D：完整图生产入口及身份 | PASS（代码路径与拒绝路径范围）。`from_reference → load_training_graph → load_reference` 使用现有图/来源/注释/群组身份校验；适配过程复制原始有符号 int64 稀疏计数，不设置阈值、不裁边、不裁节点，保存只读原始数组。身份含来源、图计数摘要、类型映射和输入集合；返回的身份为深拷贝。过滤图、研究图和缺失路径拒绝测试明确；小电路另有 `numerical_test_only` 身份。真实完整图加载资源与学习结果由 P1 实测，不能由此项推断。 |
| E：依赖、环境和文档 | PASS。`train` 可选依赖声明 torch 2.14.1 与 psutil，`.venv-train` 已忽略；技术栈和 TRAINABLE_SNN 文档说明独立环境、CPU float32/64、模型身份及裸 state_dict 的限制。实施报告记录安装版本与本批检查结果；本批代码没有修改既有设备环境或设备控制链。 |

## Validation

- Risk Level：S3（新增神经动力学、反向传播和模型身份）；验收范围限制为 P0 相关模块。
- 阅读 P0 Ready 契约、Spec 008、Plan、Decision 009、实施报告，以及 `trainable.py`、`trainable_graph.py`、完整 `test_trainable_core.py`、TRAINABLE_SNN 文档和依赖/环境相关差异。
- 追踪现有 `load_reference` / `validate_reference` 的拒绝与身份校验，确认新增生产适配没有使用过滤或研究图入口。没有修改该已验收加载器或重新扫描原始数据。
- 独立阅读 [Shiu 官方 model.py](https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/model.py)，核对方程、默认参数、严格阈值、输入细胞无不应期及 reset 中未声明神经元变量 `w` 的文档边界。没有运行 Brian2，未声称官方逐步等价。
- 审阅实施者证据：`PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_trainable_core.py -q` 为 19 passed，附 CSR beta 提示；`git diff --check` 通过。已逐项检查测试实质覆盖，未发现需要另行执行的具体缺口，因此未机械重复测试。
- 未运行完整脑、训练任务、全项目测试、build/lint 或设备操作；并行 V1 清理和 P2a 环境变更不纳入本报告。

## Issues

无 P0 阻塞问题。

## Regression Risks

- 全图前向、反向和状态检查的实际成本尚未知；本批小电路结果不能支持 Mac 资源或实时性结论。
- `state_dict` 只保存 theta；P1/P3 的持久化必须保存并核对图与模型身份，不能单独使用裸参数文件作为可复现证据。
- float32 与 float64 具有不同数值精度，身份已区分；后续不得把不同精度、事件语义或感觉映射的结果混为同一模型。
- 固定类型增益改变内部传播，但本批仅验证数值作用，尚未证明任务学习收益或生物有效性。

## Required Rework

无。可由协调 Agent 记录 P0 完成并独立提交；整体目标仍需继续 P1–P4 及已授权的研究与设备接入阶段。
