# Implementation Report — 完整图前向共享 SciPy CSR

## Summary

按 Ready [子规格](forward-csr-cache.md)完成 A/B 的软件实施。生产前向改为构造一次的共享、只读 SciPy `csr_array`，仍对完整向量进行乘法；显式反向保留既有完整 Torch CSR 伴随。删除生产中的旧 Torch 前向乘法，没有后端开关、活动细胞筛选、全零输入特判或兼容转发层。尚待新的独立软件验收；完整图 P1 r5 和配对诊断 r2 由协调者在软件提交后按原协议运行。

## Files Changed

- `src/cnsbridge/brain/trainable.py`：私有 `_forward_csr` 共享固定 Torch CSR 的三组缓冲，核对精度与存储指针、只读标记；前向 NumPy 视图及 Torch 结果包装；backend 身份更新。
- `tests/test_trainable_core.py`：既有独立密集代数检查接入新前向；新增 float32/float64 共享存储、只读、模型释放后的视图生命周期、重复使用缓存、非连续输入的数值/梯度以及输入/输出/权重不互相覆盖检查。
- `docs/TRAINABLE_SNN.md`：说明新前向、缓存生命周期、完整伴随、身份及 checkpoint 边界；r4 耗时明确为此前版本的实测，不推断当前整模型收益。
- 本实施报告。

## Important Decisions

1. 保留现有 Torch 前向和伴随 CSR，SciPy 仅持有前向缓冲的只读视图。`csr_array` 构造一次并核对 data、indices、indptr 指针与精度，没有新增全图数据副本；视图保有底层 Torch 存储生命周期。原始只读整数连接图独立不变。
2. `fixed_sparse_matvec` 的首参直接改为缓存的 SciPy CSR，更新其唯一业务调用及数值测试，不保留旧签名适配层。每步 SciPy 完整向量乘法产生独立输出；完整 Torch 伴随对全部输入分量反传，静默分量也可有非零梯度。
3. backend 改为 `scipy_cpu_csr_forward_torch_csr_adjoint`；已有源码摘要同时改变模型身份。缓存为私有非参数对象，两个 Torch 矩阵仍为非持久化 buffer；gain-only 模型的 state_dict 仍仅 theta。未改 checkpoint 校验或旧身份拒绝规则。
4. 无背景核心、数值工厂和 P2c 继承共用新前向。方程、增益参数化、延迟、不应期、完整状态验证和外部输入限制均未更改。仍只支持声明的 CPU float32/float64；不增加二阶梯度承诺。

## Validation

- Risk Level: **S3**，核心共享前向数值算子改变。
- Checks Performed：先阅读限定业务/文档 diff，`git diff --check` 无输出；执行一次下列核心及相关 checkpoint/P2c 检查。

```sh
PYTHONPATH=src .venv-train/bin/python -m pytest \
  tests/test_trainable_core.py \
  tests/test_training_probe.py::test_checkpoint_exact_roundtrip_and_explicit_test_role \
  tests/test_training_probe.py::test_invalid_checkpoint_never_mutates_parameters \
  tests/test_training_probe.py::test_tonic_checkpoint_save_and_restore_rejected_before_side_effects \
  tests/test_tonic_dynamics.py::test_analytic_constant_flow_strict_first_spike_reset_and_refractory \
  tests/test_tonic_dynamics.py::test_fixed_and_learnable_share_forward_but_parameter_and_state_identities_differ \
  tests/test_tonic_dynamics.py::test_rho_has_analytic_voltage_gradient_updates_and_survives_reset \
  tests/test_tonic_dynamics.py::test_all_groups_constructor_retains_graph_counts_members_and_independent_identity -q
```

- Result: **46 passed, 1 warning in 0.97s**。唯一 warning 为既有 Torch CSR beta 提示。包含原有独立密集前向/伴随参考、有符号非对称连接、自连接、孤立节点、全零/部分静默输入、递归动态、延迟梯度、内部参数更新/恢复、checkpoint 严格身份与变体拒绝，以及 P2c 背景动态检查。
- 通过后停止数值验证，没有运行无关完整测试套件、build、lint 或重复上述检查。

## Known Limitations

- 本批是软件实现与数值检查，未运行完整图、基准或生产性能测量；不据小图或此前内核结果宣称整模型提速、相同脉冲轨迹或训练成功。
- 完整图 P1 与带/不带背景配对动力学对照仍待子规格 C/D，旧结果及原始资产不覆盖。
- 固定 Torch 权重及私有视图是模型常量；调用者不得修改固定权重。只读 NumPy 视图防止经该视图意外写入，不是对任意外部 Torch 原地修改的安全隔离。
- 未停止或调整 Noria/Docker，未连接硬件，未创建 commit；独立验收和提交由协调者处理。

## Remaining Questions

无实施阻塞；待新的独立 Agent 验收软件 A/B 后推进完整图 C/D 对照。
