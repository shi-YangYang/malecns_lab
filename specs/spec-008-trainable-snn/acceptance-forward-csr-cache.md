# Acceptance — 完整图前向共享 SciPy CSR，首轮软件验收

## Result

**FAIL**。验收范围为 [子规格](forward-csr-cache.md) A/B 的软件实施，未进行 C/D 完整图实验。共享缓存及完整伴随的实现与定向数值检查没有发现新的数值阻断，但新 backend 身份未同步到两个实际运行的证据校验器，当前版本不能进入有效的完整图验收闭环。

## Spec Coverage

- A，缓存与数值路径：已审阅。`TrainableLIF` 从原始只读整数计数构造原有完整 Torch 前向及伴随 CSR；私有 SciPy `csr_array` 只构造一次，核对三组缓冲的指针与 dtype，并设置只读视图。前向执行完整向量乘法，未按活动细胞筛选、裁边、增加全零特判或改动 LIF 方程。反向继续使用完整 Torch 伴随，保留静默输入的梯度。
- A，身份与持久化：缓存未进入 state_dict，模型 backend 已显式更新，源码摘要自然改变模型身份；未添加旧 checkpoint 的放行逻辑。但跨模块证据身份契约不一致，见问题 1，因此 A 尚不能作为整体通过。
- B：实施报告记录的 46 项定向检查已通过，包含 float32/float64、密集独立代数参考、静默梯度、递归动态、内部更新及 checkpoint/P2c 相关检查。新增检查实际覆盖缓存共享/只读/释放模型后的存储生命周期、重复使用缓存、非连续输入、完整梯度及输出不污染输入/权重。生产旧 Torch 前向调用已删除，无永久后端开关；独立验收因问题 1 未通过，尚不应提交为已验收的完成步骤。
- C/D：未执行、未验收。内核实测及本轮小图检查不构成完整动力学或训练性能证据。

## Validation

- Risk Level: **S3**，共享数值算子及跨模块模型身份变化。
- Evidence Reviewed / Checks Performed：读取项目规则、验收流程、Spec/Plan、相关决策、实施报告；逐项审阅 `brain/trainable.py`、`tests/test_trainable_core.py`、`docs/TRAINABLE_SNN.md` 的实际 diff，以及 P1/配对诊断证据校验代码和 backend 相关测试 fixture。
- 本轮未重复实施者已通过的 46 项检查，没有运行完整图、基准、build 或全量测试。问题 1 是模型实际返回值与两个校验器严格等式之间的直接矛盾，不需要花费完整图计算来复现。
- 数值/缓存实现没有发现另一个需阻断的具体问题。固定 Torch 缓冲仍可由恶意或错误外部原地修改，实施文档已明确该约束；NumPy 只读视图不是对任意外部写入的隔离机制。

## Issues

### 1. 新 backend 被实际证据入口拒绝

`src/cnsbridge/brain/trainable.py:168` 输出 `scipy_cpu_csr_forward_torch_csr_adjoint`，但：

- `src/cnsbridge/learning/probe_evidence.py:112` 的 `_model` 要求 `backend == "torch_cpu_csr"`，会产生 `Model backend/dtype mismatch`。
- `src/cnsbridge/diagnostics/observability_evidence.py:138` 的 `validate_mode_identity` 同样只接受旧值，会产生 `Model identity/config mismatch`；P2c 继承的模型身份也受影响。
- `tests/test_probe_evidence.py:42` 的证据 fixture 仍构造旧 backend，无法用现有测试有效发现生产模型与证据校验不一致。

因此即使数值工作进程成功结束，新 P1 r5 和配对诊断 r2 也不能被严格证据校验认定有效。这是本次身份更新直接造成的集成回归，不是 C/D 的实验不确定性。

## Regression Risks

当前失败会使有成本的完整图计算在证据验收阶段被拒绝。返工必须保持新 backend、源码身份和模型 ID 的严格绑定，不应以接受任意 backend、同时放行旧身份或删除校验来掩盖问题。原始运行记录继续保留原身份，不重写历史产物。

## Required Rework

1. 将两个证据校验器的期望 backend 与当前生产模型同步；保持无 Torch/无全图加载的轻量校验边界，以及现有源码/图/配置/dtype 校验。
2. 更新直接相关证据 fixture，并增加或调整最小的集成身份回归检查：当前模型的新 backend 可通过，错误及旧 backend 在重新计算自洽身份后仍被拒绝。覆盖 P1 与配对诊断两种模式，不能只验证摘要字符串。
3. 仅运行上述改动直接相关的证据检查，不无理由重跑已通过的核心数值检查。由新的独立验收者复核返工，再提交及进入 C/D。

本验收者未修改业务代码、未创建 commit、未连接硬件或调整 Noria/Docker。
