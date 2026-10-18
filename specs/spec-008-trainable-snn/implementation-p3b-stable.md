# Implementation Report — P3b r2 稳定分组裁剪

## Summary

按 Ready [候选协议](p3b-stable-group-clip.md)实现 r2。当前入口保留完整图、两帧连续 100 ms 信用窗、固定背景和一次真实 Adam，将优化改为 theta / readout 两组各自稳定裁剪，并保存实际命名 Adam 状态及 r1 优化前一致性对照。没有运行生产完整图；软件最终验收由新的独立验收者负责。

## Files Changed

- `src/cnsbridge/learning/joint_probe_contract.py`、`joint_probe.py`：registration/protocol/worker/final schema 升到 v2，新登记绑定 r1 历史与稳定优化规则；CLI 使用新输出目录，旧 r1 不修改。
- `joint_probe_worker.py`：先拒绝全部非有限梯度，float64 归约两个组范数，记录 float64 系数和真正乘入梯度的 float32 系数，保存全部前后梯度、分组更新统计以及三个参数的实际 step/exp_avg/exp_avg_sq；r1 比较失败先保存报告，在 Adam 前停止。
- 新 `joint_probe_history.py`：Torch-free 历史锚点、原登记/终态身份与六个原终态文件摘要核验，冻结被比较数组的当前已验收 SHA；核对同输入、全部初值、baseline 动作/全部 DN 特征和原始梯度。
- 新 `joint_probe_optimizer.py`：Torch-free 分组裁剪与 float32 Adam 独立复算，包含二阶矩下溢，严格命名状态字段、形状、dtype、有限性与 step=1。
- `joint_probe_evidence.py`：新增文件清单、独立历史与优化证据门槛，保留恢复/重载以及完整协议、内部数值更新、内部动作作用三层结论。
- `tests/test_joint_probe.py`：扩充受影响的数值、历史、篡改拒绝、轻量父进程和阴性结果检查。
- `docs/JOINT_PROBE.md`：当前 r2 命令、优化规则、版本兼容及旧失败锚点来源说明。

未修改脑核心、神经策略、教师数据、策略 checkpoint 或资源监督。协调者同时更新的 `.ai/prompts/task-handoff.md` 不属于本实施改动。

## Important Decisions

- 数值计算本体和参数仍为 CPU float32；仅梯度范数/统计采用 float64。两个组分别最大 L2=1，不能声称与 r1 统一裁剪等价。
- `adam_state.npz` 字段以 `brain.theta`、`readout.weight`、`readout.bias` 命名，每个对象保存 float32 scalar step 和两个同形 float32 矩。独立 NumPy 复算逐步保留 float32 舍入/下溢，不使用理想化首步公式替代真实矩。矩容差固定 rtol=1e−6、atol=float32 最小正次正规数；参数更新沿用原 rtol=5e−6、atol=1e−9。
- 从封存 dataset 同级目录定位固定 `joint_update_20261019_r1`；该目录缺失或锚点改变即不启动 worker。r1 registration/protocol ID 保持原值，旧优化源码不要求与 r2 相同，其他绑定模块必须保持相同。
- r1 失败时未产生数值文件总清单。新增模块中的九个字节锚点（旧 registration/result 与七个比较文件）是在 r1 已独立验收后为 r2 冻结，不冒充旧终态六文件清单。比较文件只读；新登记记录锚点和来源。
- 初始参数精确相等；baseline 动作/特征、原始梯度比较沿用 rtol=1e−6、atol=1e−7，保存 exact/allclose/maxdiff。相同输入、DN 顺序和策略身份也检查，任何超差使协议无效，不能解释后放行。

## Validation

- Risk Level: **S3**，仅本次训练优化及证据链。
- `PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_joint_probe.py -q`：**62 passed，1 warning，1.64 s**。初次命令返回 session `40519`，随后收取同一 session 的终态 exit 0，没有重启或重复运行。
- 检查覆盖真实数值小电路两帧连续状态/一次 Adam/五分支、全零梯度及 loss 增加阴性、1e31 量级有限 theta 梯度与读出分组隔离、实际 float32 系数、二阶矩下溢、NaN/±Inf 在任何梯度改动前拒绝、命名矩/步数/shape/dtype/额外字段篡改拒绝、r1 锚点/旧终态/保留源码绑定、优化前超差 invalidation、明确 v2 schema、父进程不导入 Torch、资源监督边界。
- 仅现有 Torch sparse CSR beta 提示，不是新的失败。
- `git diff --check` 与定向 diff 阅读通过；新增文件经实施阅读。没有完整测试套件、build、旧实验重跑、全图、硬件、服务或 Docker/Noria 操作，没有 commit。

## Known Limitations

数值小电路和算术 fixture 不证明生产完整图 r2 已通过。仍须新的独立软件验收、提交后由协调者在新目录实际运行一次，再按原始产物独立验收。巨大内部梯度来源和长轨迹稳定性尚未解释；保持 1 GiB 预留、6 GiB 申请、600 s/0.1 s 监督。参数存档与本次 Adam 证据不是可恢复训练存档；AC3–AC6 未完成。

## Remaining Questions

无实施阻断项。未放宽协议或试运行寻找阳性结果。
