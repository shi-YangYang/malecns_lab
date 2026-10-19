# Implementation Report — MPS 有界续训

## Summary

实现 Ready `mps-continuation.md` 的生产软件：新 MPS 完整固定 COO 前向/全细胞伴随、原感觉编码与读出规则、CPU 小参数主副本 Adam、完整状态迁移/恢复、独立保存数组算术复核、资源监督和每次 1–4 窗入口。保留 CPU 来源已完成预算，采用独立后端与来源身份。实际完整图第61/62窗由协调者在独立软件验收、提交后执行；本报告没有宣称它们通过。

## Files Changed

- `brain/mps_identity.py`、`brain/mps_trainable.py`：纯身份推导及实际 MPS LIF，CPU/GPU 完整双向缓存坐标/值分块精确检查。
- `learning/mps_policy.py`：独立 MPS 策略；保留 CPU SpatialDrive phase/events 规则；参数未恢复前拒绝前向。
- `learning/mps_optimizer.py`、`mps_window.py`：CPU 主参数 Adam、同步计时、每窗完整证据及原 `validate_window_numbers` 检查。
- `learning/mps_state.py`、`mps_archive.py`：所有递归状态迁移返拷、严格 NPZ/JSON 完整存档、实际前向前恢复证据。
- `learning/mps_contract.py`、`mps_evidence.py`：CPU 前缀继承、新源码/图/数据绑定、独立 NumPy seal/链/资源复核。
- `learning/train_mps.py`、`mps_worker.py`、`mps_memory.py`：轻量父进程、单次 claimed 受监督子进程、共享锁、RSS/系统预留/MPS allocator 和周期记录。
- `tests/test_mps_continuation.py`、`docs/MPS_TRAINING.md`：定向检查与可直接执行的迁移/恢复命令。

没有修改 G29、R13、既有 MPS20 源文件。未改依赖、硬件入口或 Docker/Noria。协调者维护的 Spec/Decision/handoff 未被实施 Agent 修改。原未跟踪 `scipy-backward-boundary-audit.md` 保留。

## Important Decisions

- 延续真实 CPU 边界而非重新训练；全部已完成 CPU 窗只作为继承前缀，不算 GPU 证据。
- 新 `cnsbridge.mps_*` 身份/模式；复用原配置、状态类型、输入生成、裁剪、Adam 契约、独立 NumPy 数值复核和若干纯前缀 helper，不改旧身份或 monkeypatch 生产对象。
- 非有限、自己的裁剪/Adam/读出不一致仍拒绝。原 CPU/MPS kernel 数值 FAIL 关联到新 run，未重写为等价通过。
- 恢复前向前逐项比较全部参数、Adam 和递归状态；只有新合法 seal 才计入窗口。数值/协议失败阻止同 run 自动续接；资源/中断允许从最后合法边界重新开启一段。
- 每段请求 6 GiB，实际不超过 available−额外1 GiB；600 s/.1 s 监督。MPS allocator 单独限额/记录，不与 RSS 相加。

## Validation

- Risk Level: **S3**，核心设备数学、优化器桥和持久化链。
- `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/test_mps_continuation.py`
  - 初轮 4 passed / 1 failed。失败来自测试夹具 delay=0.1 ms 与预期公式来自0.2 ms不一致，修正夹具后该项单独通过；未修改模型数学。
  - 完成存档语义拒绝/失败续接检查后，最终相关子系统 **5 passed，3.27 s**。覆盖真实 MPS 有符号非对称前向/全细胞伴随、静默输入梯度、设备状态/输入约束；真实小图 CPU→MPS 迁移、两次 MPS 内部更新及原 CPU Adam 独立算术；完整存档准确恢复；场景重置不清 Adam/参数；非有限/shape/step/时钟/外来run/额外字段/SHA拒绝；未封存不计进度、数值失败不续接、资源失败保留前缀；轻量父进程、源/模式拒绝、锁、预算、拒绝覆盖与失败登记保留。
- 随后为符合 Spec 把 CPU 裁剪/Adam 和参数回传计时拆开，仅重跑 `test_actual_mps_window_cpu_adam_exact_archive_restore_and_scene_reset`，**1 passed，2.89 s**。
- Ruff 命令无法执行：`.venv/bin/python` 中没有 `ruff`；没有安装依赖。未运行全项目测试、build 或硬件命令。

## Known Limitations

- 小图是明确的 numerical fixture，不是生产脑；生产入口只接受完整 reference、joint、fixed_lamina_8mv。
- 未在本实施阶段执行完整图 GPU 1000步 BPTT。该保存张量的内存远高于单核 benchmark；OOM/超时必须由受监督实测记录，不能静默缩图/缩窗。
- 当前直接保留每步设备状态/有限性检查，可能增加 GPU 同步成本。实际端到端速度尚未测出；不声称6倍训练加速。
- 独立进程的实际完整图连续两窗和实际验收由协调者完成；本报告不是最终独立验收。

## Remaining Questions

无待用户决定事项。CLI、输出布局和恢复语义见 `docs/MPS_TRAINING.md`。未提交，由协调者在独立验收通过后提交。
