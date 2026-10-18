# Implementation Report — P3c2 有界连续训练

## Summary

按 Ready P3c2 新增有界连续场景训练工具。每个 run 固定完整参考图、r2 策略、四个 train 场景、模式、来源和一个 epoch；每次只执行 1–4 窗，逐窗保存完整状态，下一段显式恢复。所有生产脑计算在受监督子进程内，父复核不导入 Torch。未运行完整图、设备或付费任务，未操作 Noria/Docker，未提交 Git。AC3–AC6 保持未完成。

## Files Changed

- `src/cnsbridge/learning/training_contract.py`：固定训练/优化协议、运行时/来源绑定、r2 字节锚点和来源校验、数据选择、游标和规范前缀行。
- `training_math.py`：NumPy 损失、稳定分组裁剪、从实际旧矩开始的一般步 Adam 复算、读出检查和 r2 首窗对照。
- `training_archive.py`：Torch-free P3c1 完整存档检查，交叉核对 nested 参数、完整活动状态、命名矩、上下文和外部 run。
- `training_evidence.py`：逐窗/逐段链验证、真实数据/事件/前缀绑定、资源采样边界、实际恢复值验证、数值异常拒绝续跑。
- `training_window.py`：两帧连续计算、真实梯度/矩/参数快照、保留状态的更新与无效窗跳步；异常保留到达的数值证据。
- `training_worker.py`：生产 full_reference 构造/恢复、场景边界、依序封存、恢复前证据和原始图计数摘要。
- `train_snn.py`：轻量父入口、新 run/显式续段参数、OS 工作区锁、资源监督、终态与 epoch 进度。
- `tests/test_training_runner.py`：27 项本批定向数值、跨进程、持久化、失败边界与入口检查。
- `docs/TRAINING_RUNNER.md`：生产命令、证据格式、恢复与研究边界；`docs/TRAINING_CHECKPOINT.md` 窄幅链接更新。
- 本实施报告。

既有脑、策略、感觉、教师数据、参数/训练存档、优化器、P3b 和 resource_guard 源码未修改。协调者新增的 P3d Draft 不属于本实施。

## Important Decisions

- 复用现有 P3c1 完整存档和实际 optimizer；没有改旧格式、源身份或训练模型。新轻量检查器不导入 Torch，逐窗只返回小参数/矩/摘要，完整状态数组随本窗释放。
- 封存顺序固定为 evidence → event → prefix → checkpoint → window；无循环哈希。恢复只沿全局窗口连续链，损坏封存档不会被更早档掩盖。
- 保存实际更新前 Adam 状态；首步的真实状态缺失用显式 bool 标记，复算零矩不伪称原对象已持有零矩。一般步按照协议处理 lerp 抵消，先核对实际新矩，再由实际新矩复算参数。
- 同场景每窗 detach 不 reset；下一场景恰好一次 initial_state，参数/矩保留。恢复前保存真实参数、Adam、完整状态摘要和原游标并立即核对，再允许前向。
- 资源/进程失败不改成成功，只有合法前缀可另报恢复边界；数值/协议失败明确无 recoverable_prefix，原 run 不自动继续。未封存尾部和失败证据保留。
- resource_guard 的 CSV 原点稍晚于登记时钟，边界检查保守要求后续预算内样本；紧邻末采样时可能 `clock offset unresolved` 拒绝，未放宽资源预算。此界限已单独检查并文档说明。
- 首两个实际 joint 窗口由协调者按协议使用两次 `--windows 1`，本工具没有自动长跑或重试。所有完整图效果仍待实际验收。

## Validation

- Risk Level：**S3**，新增跨进程训练持久化、源链、资源与数值证据；范围只限本批和直接接口。
- 第一次命令：`PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_training_runner.py -q`；session `54620` 已收取 **exit 0，22 passed, 1 warning in 6.11s**。
- 在协调者指出的两个具体边界（原点偏移和数值失败恢复标记）修正，以及补充重算 SHA 后的晚期语义损坏检查后，执行同一个本批定向命令：session `15953` 已收取 **exit 0，27 passed, 1 warning in 6.31s**。警告是既有 sparse CSR beta。
- 两模式一般步 Adam 各执行 160 次实际更新，包含非零旧矩、正负/零/1e−30、抵消和下溢，独立复算满足冻结容差；协议给出的抵消例实际 lerp 与新公式逐位相同，普通分步公式被拒绝。大有限 theta 梯度的稳定裁剪单独检验。
- joint/frozen 各先独立连续四窗；另一分支的首两窗分别运行在两个真实 OS 子进程（终态 returncode 0），再恢复到下一场景。两分支各窗 evidence 全部数组精确相同，最终全部参数/实际 Adam/完整状态精确相同；连续分支统计每场景恰好一次 reset。包含非零 18 槽延迟状态及感觉相位。子进程是明确数值 fixture，不是生产训练或资源实测。
- 覆盖全无效窗保留状态推进而不改参数/矩/步数；首窗无效拒绝并保留诊断；第二窗存档中断留下可验证首窗及不计进度尾部；资源失败保持 failed，fake exit 0 不补齐证据。
- 覆盖坏父/游标、真实数据变动、重复/漏窗、外部/重复前缀、缺事件/存档、恢复前值变化；额外对状态/Adam/nested run 身份篡改后重算外层 SHA/ID，仍由语义交叉核对拒绝。
- 覆盖 r2 超差、源模块导入不含 Torch/训练存档、CLI 互斥/窗口范围、工作区锁，以及原点偏差保守边界和数值失败不可续跑。
- 已通过后停止自动化检查；未重跑既有 51/62 项检查或整个项目。

## Known Limitations

- 未执行生产完整图首两窗、完整 epoch 或独立冻结脑训练，软件通过不代表长时稳定性或学习收益。当前仍为离线教师轨迹模仿，不是神经闭环。
- CPU float32、固定协议、同源码/运行时/线程恢复；不支持源版本迁移、全无效首窗、任意新数据集或多 epoch。
- manifest-last 不是文件系统事务或掉电恢复保证。资源监督为采样，窗口采样原点边界保持保守。
- 存档/扫描和重启的完整图成本尚未实测，4 窗约 400 s 仅是协议里的估计；段仍受 600 s 监督。

## Remaining Questions

无实现阻断项；交协调者安排新的独立软件验收。验收并提交后，使用文档中两次单窗命令做新 run 的首两个实际窗口；独立实际验收后再推进同一冻结 run。未自写最终验收 PASS。
