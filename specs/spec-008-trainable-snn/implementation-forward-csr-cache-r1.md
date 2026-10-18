# Implementation Report — 前向缓存证据身份返工

## Summary

修复[首次独立验收](acceptance-forward-csr-cache.md)发现的新 backend 被 P1 和配对诊断拒绝的问题。两个轻量证据校验器现在严格要求当前生产模型的 `scipy_cpu_csr_forward_torch_csr_adjoint`，不再接受旧值；保留原有模型哈希、来源、图、配置、dtype、登记和工件校验。本返工没有修改数值核心或实验协议，尚待新的独立验收。

## Files Changed

- `src/cnsbridge/learning/probe_evidence.py`：P1 模型 backend 严格等式同步。
- `src/cnsbridge/diagnostics/observability_evidence.py`：零背景和固定背景模型的 backend 严格等式同步。
- `tests/test_probe_evidence.py`：更新人工证据 fixture，增加当前 `TrainableLIF` 构造器生成的模型身份在完整证据校验路径通过，以及旧/未知 backend 重算 model ID、checkpoint ID 和工件摘要后仍被拒绝的检查。
- `tests/test_observability.py`：对真实构造器生成的零背景和固定背景模型身份分别检查旧/未知 backend；重算有效 model ID 后仍须因 backend 契约被拒绝。
- 本实施报告。

## Important Decisions

1. 只修改两个期望值，不增加旧后端白名单、任意字符串放行或兼容回退；不通过导入 Torch/模型核心获取常量，轻量校验边界保持。
2. P1 测试沿用明确标记的人工证据，使用小矩阵和完整参考形状的伪造图身份来检查模型元数据的跨模块契约。它不经过生产图加载，不进行脑仿真，也不是完整图证据；新模型身份由实际构造器产生，其他数值工件仍为手工软件 fixture。
3. 配对诊断复用现有小型数值记录，正向接受检查来自既有 `test_numerical_role_never_becomes_production`，新增拒绝检查覆盖两种实际模型。数值测试角色继续不能进入生产选择校验。
4. 历史实验记录及之前的 46 项核心数值检查不重写、不重复；返工不改变缓存、伴随、动力学、资源预留或 Noria/Docker。

## Validation

- Risk Level: **S2**，本次返工为两个相关证据入口的身份契约同步；上层前向算子子批仍为 S3。
- Checks Performed：阅读限定四文件 diff；执行一次以下定向证据检查，包含轻量导入子进程检查。

```sh
PYTHONPATH=src .venv-train/bin/python -m pytest \
  tests/test_probe_evidence.py \
  tests/test_observability.py::test_numerical_role_never_becomes_production \
  tests/test_observability.py::test_mode_identity_rejects_self_consistent_wrong_backend \
  tests/test_observability.py::test_parent_and_validator_import_without_torch -q
```

- Result: **37 passed, 1 warning in 0.99s**。唯一 warning 为既有 Torch CSR beta 提示。正向验证与重算身份后的拒绝检查均通过；证据模块独立导入不加载 Torch/脑加载器。
- 通过后停止测试；未重复核心 46 项检查，未执行完整图、内核基准、build、lint 或全量测试。限定 diff 的空白检查无问题。

## Known Limitations

这里只验证证据身份契约的接入与拒绝行为，不提供新的完整图运行、性能、动力学等价或任务学习证据。P1 r5 与配对诊断 r2 仍由协调者在新独立软件验收和提交后执行。旧实验保留其原后端身份和原始产物，不由新校验器冒充新实现结果。

## Remaining Questions

无实施阻塞；未创建 commit，提交及独立验收由协调者处理。
