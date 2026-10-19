# Implementation Report — MPS 完整窗口分段重算

## Summary

按 `mps-continuation.md` 的“完整窗口分段重算”子批实施固定 50 神经步的 non-reentrant activation checkpoint。生产 MPS 策略始终使用分段路径，保留原每帧步数、两帧窗口、全部神经元与连接；未改神经积分、感觉事件、优化器或训练预算。此次解决 r2 前向保存整窗反向张量的具体内存瓶颈，实际完整图是否适配当前预算仍待新 r3 实测。

## Files Changed

- `src/cnsbridge/learning/mps_recompute.py`：独立分块函数工厂，每块绑定感觉不可变快照、mask、神经时钟及帧内步号。完整 v/g/不应期/pending 和两个 DN 累加器作为显式张量输入输出，不跨块或跨帧 detach。保留逐步加法顺序，包含尾块和跨半帧块。
- `src/cnsbridge/learning/mps_policy.py`：生产前向接入分段路径；直接 MPS 路径只允许显式 `numerical_test_only` 图，作为数值对照。
- `src/cnsbridge/brain/mps_identity.py`、`learning/mps_contract.py`：策略身份、协议和源闭包登记 50 步、non-reentrant、`preserve_rng_state=False`、默认 determinism check、early stop 及新 helper 文件。
- `tests/test_mps_continuation.py`：新增实际 MPS 多块两帧对照，复用现有迁移 fixture 构造逻辑；其余无关检查保持。

## Important Decisions

每个 checkpoint 闭包通过工厂参数绑定独立块边界，不引用循环中随后变化的脑状态或外部感觉驱动器。重算调用创建私有 `SpatialDrive` 并立即恢复已绑定 phase/event/clock；其构造使用原固定种子初始化，恢复后不使用初始化相位，不消耗全局 RNG，也不引入随机输入。重算不会推进原感觉对象、训练游标或预热流程。

DN 累加器直接携带上块的运行和，每步仍执行原加法；没有使用“各块求和再合并”。原完整 W/WT、代理梯度和 CPU Adam 算术源码保持。本批不改旧 G29/R13/MPS20 源文件，不删除或覆盖 r1/r2 失败产物，不提交 Git。

## Validation

- Risk Level: **S3**（神经前反向的保存/重算及梯度连续性）。范围仅为该内存修正和受影响的更新/恢复，不运行全项目检查。
- Read local installed PyTorch 2.14.1 `torch/utils/checkpoint.py` to verify non-reentrant options, nested non-Tensor return semantics and deterministic-recompute constraints.
- 已阅读业务 diff 和新增 helper。
- 实际命令（共享 `data/.continuous-training.lock`，fallback=0）：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python -m pytest \
  tests/test_mps_continuation.py::test_checkpoint_matches_direct_multiple_blocks_two_frames_and_raw_gradients \
  tests/test_mps_continuation.py::test_actual_mps_window_cpu_adam_exact_archive_restore_and_scene_reset \
  -q -s
```

结果：**2 passed，8.89 s**；一项原有 Sparse CSR beta 警告。无失败重试，未重复 CLI/来源拒绝/资源检查。

实际 MPS 对照 fixture 为每帧 120 步（50+50+20），连续两帧 240 步；半帧分界 60 落在第二块内。初态有非零 v/g、全部 18 个 pending 向量、活动不应期、非零 phase/events。第二帧反向产生非零第一帧末态梯度，初始 pending 梯度非零；原输入状态及最终感觉状态在反向前后精确保持，脑/感觉时钟只前进 240 步。

- 动作、DN 特征、完整末态：直接路径与分块路径**逐位相同**。
- raw gradient：theta、readout weight/bias、初始 v/g 和全部 18 项 pending 最大绝对差全部为 **0.0**，且有限；theta 非零。
- 受影响已有检查：MPS 完整两帧更新接 CPU clipping/Adam、参数内部更新、命名矩恢复、NPZ 读写、精确状态恢复、再次更新和 scene reset 通过。

## Known Limitations

小型 fixture 只证明本次重算的数值语义，不代表完整脑训练通过或完整窗口内存足够。没有运行完整图、降低系统预留、改变 50 步块长、窗口长度或图规模；没有操作 Noria/Docker、无人机或其他服务。r3 仍需协调者在软件独立验收与提交后新登记运行；若预算仍不足，必须保留真实失败证据，不能称续训成功。

## Remaining Questions

无需求决策待确认。等待独立软件验收和完整图新登记实测。
