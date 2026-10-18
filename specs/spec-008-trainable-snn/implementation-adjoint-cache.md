# Implementation Report — 完整图伴随 CSR 缓存

## Summary

按 Ready [子规格](p1-adjoint-cache.md)完成软件实现。模型在构造时保存固定前向 CSR 与固定伴随 CSR，稀疏自定义反向直接计算缓存的 `Wᵀ @ gradient`，保留每个输入分量的梯度；删除旧的每步 `transpose` 调用。未运行完整图对照，实际性能、资源和 r3 数值比较由协调者在独立软件验收及提交后另行登记。

## Files Changed

- `src/cnsbridge/brain/trainable.py`：新增非持久化 `base_weights_adjoint`；原 CSC 的列指针、行索引和有符号数据直接构成转置 CSR；前向临时 SciPy CSR 在伴随分配前释放。`fixed_sparse_matvec` 显式接收固定双方向矩阵，反向只保留并使用伴随缓存。
- `tests/test_trainable_core.py`：扩展原独立密集代数检查，覆盖 float32/float64、0.275/1.0/1.7 权重尺度、有符号非对称边、自环、孤立行列、部分及全部静默输入；核对缓存方向、前向、完整输入梯度、非持久化和原始只读整数计数。
- `docs/TRAINABLE_SNN.md`：说明缓存生命周期、P2c 继承、持久化边界及静态额外数组成本；未预报提速或节省内存。
- 本实施报告。

没有修改协调者已有的 Spec、Plan 和交接文件，也未扩大业务代码范围。

## Important Decisions

1. 全仓调用搜索确认 `fixed_sparse_matvec` 仅用于 `TrainableLIF.step` 及其数值测试，直接更新签名，不保留每步转置兼容分支。
2. 两个方向均复制到 Torch，不与原始只读整数图共享可写数组；相同浮点转换及 `weight_mv` 缩放保留原计算约定。伴随矩阵来自 CSC 原始缓冲排列，不发生节点、边或符号筛选。
3. 原 `TrainableLIF` 与数值工厂、`LaminaTonicLIF` 使用同一父类构造及 step，未增加第二套模型或配置分支。参数、方程、延迟、输入限制和代理梯度均不变。
4. 缓存登记为 `persistent=False`，原模型 state_dict 仍仅 theta。实现文件摘要自然更新模型身份，既有 checkpoint 严格匹配逻辑保持；不迁移、覆盖历史身份或运行产物。

## Validation

- Risk Level: **S3**，核心共享伴随算子改变；只执行与此风险直接相关的核心、checkpoint 和 P2c 定向检查。
- Checks Performed:

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

- Result: **44 passed, 1 warning in 1.37s**；唯一 warning 是 Torch 的既有 sparse CSR beta 提示。独立密集参考覆盖双方向数值与静默细胞非零梯度；原递归事件、不应期、延迟、内部参数更新及归零恢复检查通过，checkpoint 身份/精度/损坏保护和 P2c 继承相关检查通过。
- 已阅读业务及文档 diff；完成后进行限定路径 whitespace diff 检查。未运行无关资源监督测试、全套测试、build 或 lint；未重复已通过的数值检查。

## Known Limitations

- 软件检查不是完整图性能测量。完整图 float32 伴随缓冲静态增加约 0.287 GiB，构造峰值和总峰值以新登记实测为准；不声称省内存或稳定加速倍数。
- 本批不解决零背景视觉到 DN 的可观测性，也不构成神经策略学习或实飞证据。
- 没有运行完整图、停止 Noria/Docker、操作硬件或提交；提交及后续实际对照由协调者负责。

## Remaining Questions

无实施阻塞。待新的独立 Agent 做软件验收，再由协调者执行子规格 E 的同协议完整图实际对照。
