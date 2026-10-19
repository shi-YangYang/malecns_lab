# Acceptance — MPS COO 构造峰值局部修正

## Result

**PASS**。范围仅为 `mps-continuation.md` 追加的缓存构造修正；不代表完整图 r2 的内存或训练验收通过。

## Spec Coverage

- `checked_coo` 先拒绝非 canonical 输入，经过带不变量检查的 Torch CSR 转为已 coalesced 的 COO；未再对全边 COO 排序或聚合。
- 原 float32 数据转换与权重乘法顺序保持。canonical CSC/CSR 转换保持唯一坐标和值；转换后的边数、GPU shape/coalesced 状态及分块返拷坐标/值仍检查。固定矩阵前向、缓存伴随、全细胞输入梯度代码未变。
- 临时 SciPy CSR 引用在 Torch 复制后释放，Torch CSR 引用在 COO 转换后释放；必要共享存储随 COO 存活。没有保留额外全边排序工作区的代码路径。
- 差异限于新 MPS helper、直接相关已有测试及本轮文档；没有修改 G29、R13 或旧 MPS20 源码。Spec 要求保留 r1 失败及542f50e来源、提交后新建 r2，未将失败窗计为训练进度。

## Validation

- Risk Level: **S1**。
- 独立阅读本轮 Spec 追加段、实施报告、完整 helper、调用处、相关测试与 Git diff。
- 采信实施者已完成的唯一针对性检查：`test_mps_signed_adjoint_silent_cell_gradient_and_state_guards`，**1 passed / 1.61 s / exit0**。新增断言对正向和转置缓存逐位比对旧 COO 路径的坐标及0.275权重，检查实际 MPS 和无权重梯度，并拒绝重复边；既有有符号前反向、静默细胞梯度与状态拒绝检查保留。
- 验收未重复测试、加载完整连接组或启动训练；没有改业务代码。

## Issues / Regression Risks

无阻断问题。该修正减少构造时不必要的临时分配，但尚无 r2 完整图峰值或1000步 BPTT证据；实际运行仍须受既有 RSS/MPS 限额、额外1GiB预留和失败留存规则约束。此次 PASS 不能用于宣称第61窗已完成。

## Required Rework

无。交协调者处理报告并提交后，在新 r2 登记实测。
