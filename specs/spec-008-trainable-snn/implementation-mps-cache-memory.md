# Implementation Report — MPS COO 构造峰值局部修正

## Summary

按 `mps-continuation.md` 的“实测后局部修正”处理 r1 在加载完整策略时触发 RSS 上限的问题。仅替换新 MPS 缓存构造 helper：canonical CSR → Torch CSR → 已排序且唯一的 COO，不再从完整 COO 发起 `coalesce()` 排序/聚合。转换后及时释放临时 CSR，原 float32 权重乘法、完整坐标和值、双向缓存及逐块 GPU 返拷核对保持。

这是对瞬时分配路径的局部优化；本报告没有实测完整图的新峰值，也没有把缓存修正称为完整 BPTT 已可运行。

## Files Changed

- `src/cnsbridge/brain/mps_trainable.py`：只改 `checked_coo`，拒绝非 canonical 输入，检查 CSR 转 COO 后的 coalesced 标记和完整边数，并保留实际 MPS 坐标/值逐块核对及 shape 检查。
- `tests/test_mps_continuation.py`：在现有矩阵/梯度测试中加入双向坐标/值与旧 COO 路径的逐位比较，采用非整数权重0.275，以及非 canonical 重复边拒绝断言。
- 本实施报告。协调者对 Spec 的现有修改未覆盖；原未跟踪 Draft 保留。

## Important Decisions

没有改动力学、梯度、精度、学习率、窗口、输入输出、完整图选择或资源限制；没有 CPU fallback。G29/R13/旧 MPS20 的业务源码均未编辑。旧 r1 失败证据和已保存542f50e来源不动；协调者在新软件独立验收、提交后另登记 r2。

## Validation

- Risk Level: **S1**，单一缓存构造 helper 局部修改。
- 执行唯一的最相关定向检查：
  `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_mps_continuation.py::test_mps_signed_adjoint_silent_cell_gradient_and_state_guards`
- 结果：**1 passed，1.61 s，exit0**，session50655终态已收取。该测试使用共享数值独占锁，实际 MPS 运行，验证双向完整坐标/权重、非对称有符号矩阵前向、全细胞输入梯度（含静默细胞）、原动态公式及状态/输入拒绝。仅有 PyTorch 稀疏类型/参考构造检查提示，无测试失败。
- 通过后未追加或重复已有子系统、全项目、build 或 lint 检查；本次未加载完整连接组、未启动训练、未发硬件命令、未操作 Docker/Noria。
- 已阅读本次差异，改动限于指定 helper 和直接相关现有测试。

## Known Limitations

消除完整 COO 重新排序的瞬时工作区不等于已经测得全图内存下降，更不保证1000步训练张量能满足当前 Mac 内存。实际 r2 必须由协调者在既有监督和1GiB额外预留下单独测量；未通过时仍保留 CPU 第60窗父边界与失败现场。

## Remaining Questions

无待用户决定事项。未创建 commit；交新独立验收，再由协调者提交和运行。
