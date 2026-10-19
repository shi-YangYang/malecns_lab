# Implementation Report — 全 GPU 感觉与优化器基础模块

## Summary

按 [Ready 基础模块规格](gpu-residency-foundations.md)新增两个可复用模块，未启动完整图模型、诊断或训练，未改变损失、反向时间范围及旧 Adam 的迁移决策。此为实施报告，最终验收由新的独立 Agent 完成。

`GPUDrive` 将像素阈值、占有率、感觉相位、事件数及完整 N 维增量全部放在 MPS；每一步返回新 state，不原位修改传入 state。`GPUAdam` 直接绑定已有 MPS policy 参数，GPU 稳定裁剪后调用实际可用的内置 fused Adam，参数、梯度、m/v/step 均在 MPS，没有 CPU master 或 fallback。

## Files Changed

- `src/cnsbridge/senses/gpu_drive.py`：`GPUDriveState`、`GPUDrive(mapping, config=None, camera=None)`、`initial_state()`、`mask(frame)`、`step(mask,state)`、`snapshot(state)` / `restore(saved)`。
- `src/cnsbridge/learning/gpu_optimizer.py`：`GPUAdam(named_parameters, moments=None, steps=0)`、`step()`、`zero_grad()`、`snapshot()`、严格设备/参数绑定校验。
- `tests/test_gpu_foundations.py`：两个实际 MPS 定向测试，内部覆盖像素、感觉状态与独立优化器算术/恢复边界。
- 本实施报告。

没有修改旧业务源。其他 Agent 的新诊断文件及协调者的文档修改不属于本实施范围；没有操作已有未跟踪 SciPy 审计文档。

## Important Decisions

1. GPU 感觉相位为新 float32 契约。仅初始 PCG64 种子生成、注册几何、图片上传与快照 I/O 在 CPU；神经步无 NumPy 运算。新 drive ID 绑定配置、camera、映射 ID、dtype、初始化/事件规则、库版本及实现哈希。原 float64 档不能静默装入。
2. 像素 uint8 阈值采用整数 `100*red > 165*green/blue`，GPU 按原 floor-linspace 不重叠单元计算占有率；避免浮点阈值舍入并保持已有离散颜色规则。测试覆盖全部红/绿及红/蓝 256×256 组合，以及非整除相机尺寸。
3. 稳定裁剪不直接平方大 raw gradient：`scale=max(maxabs,1)`，`u=g/scale`，`q=sqrt(sum(u²))`，`post=u*min(scale,1/(q+eps/scale))`。返回 scale/q 等设备证据，避免将可能溢出的原始范数乘积作为更新输入。零梯度无需除零；非有限 raw gradient 在改变梯度/参数前拒绝。
4. 实际采用本地 PyTorch `Adam(fused=True, capturable=False)`，保留既定两组学习率/beta/eps，无 CPU 回退。零初始化与完整 named moments 恢复是两个明确入口；基础模块不替集成者决定继承还是重置旧 Adam。
5. 新 GPU 优化器计算顺序和旧 CPU Adam 不宣称逐位等价。设备更新与独立 CPU float64 参考按提前登记的 `rtol=2e-5, atol=1e-8` 核对，小梯度额外检查非零数/符号与新 m/v，不靠绝对容差隐藏归零。

### 供集成者使用的返回值

- `GPUDrive.step()` 返回 `(increment, GPUDriveState)`，increment 为完整 N 维 MPS float32，state 为 MPS phase/event_counts + Python step 和身份；`snapshot()` 返回带两个 NumPy 数组的明确边界字典。
- `GPUAdam` 输入命名键恰为 `brain.theta`、`readout.weight`、`readout.bias`，参数是已有的 trainable MPS float32 `nn.Parameter`。
- `GPUAdam.step()` 返回 `raw` / `clipped` 的命名设备张量副本、`groups` 的 scale/q/multiplier/裁剪后范数/非零计数、Python steps。GPU 更新已经完成后可由集成者下载保存证据。
- `GPUAdam.snapshot()` 返回 steps、optimizer_id、parameters 和 moments；moments 键为 `<参数名>.step/exp_avg/exp_avg_sq`。恢复时先以保存参数建立 MPS policy，再显式传入完整 moments 与 steps。

## Validation

- Risk Level: **S3**（新的 GPU 数值、优化器及状态边界）；验证只限这两个新模块。
- 与 `gradient_probe_impl` 消息协调：该 Agent 的数值检查结束后才运行本批；测试 fixture 内使用现有 `workspace_lock()`，没有并行数值工作。
- 唯一数值检查命令：`PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src .venv-train/bin/python -m pytest tests/test_gpu_foundations.py -q -s`。
- session **87141**，已收取 exit **0**，**2 passed in 2.12s**，没有活动数值进程。

实际结果：

| 检查 | 保存输出/结果 |
|---|---|
| GPU mask vs 原像素规则 | 全红绿/红蓝阈值组合与不整除几何，`rtol=0, atol=1e-7` 通过 |
| 两个固定掩码、每幅 100 步 | 新 GPU state 重放、每步增量与计数逐位相同，跨帧与保存恢复通过 |
| 同刺激旧 float64 参考 | 200 步最大 phase 差 `3.4630918994693616e-6`；本固定序列逐步事件差 0 |
| 显式临界相位 `.985-1e-9` | 192 个输入均出现 GPU/CPU 当步事件差，证明不能宣称感觉逐位等价；新 drive ID 不同 |
| GPU fused Adam | 实际连续 4 步，其中第 3 步后从参数/矩快照恢复再更新；全部设备驻留检查通过 |
| 独立 float64 Adam 参考 | 最大参数绝对差 `1.4171075035651626e-8`；m 差 `1.1736136895645899e-8`；v 差 `4.970439229905893e-9`，均在登记组合容差内 |
| `1e30`、零、`1e-8`、普通梯度 | 稳定裁剪、连续更新通过；小梯度 clipping 非零/符号保持，单独初始小梯度 m/v 正且参数纠正方向为负 |
| 非有限 raw gradient | 拒绝，完整参数/矩及 step 保持；非法感觉 dtype/身份/非有限 mask 同样拒绝 |
| 旧来源 | 只读 stdlib SHA 检查：r3 绑定 **42** 份源码全部保持原字节，changed 列表为空 |

相关检查通过后未追加全项目测试、build、lint 或旧训练重放。模块文件及测试已读审；没有提交。

## Known Limitations

- 未进行完整 MaleCNS GPU 感觉/优化器集成或新训练；小 fixture 不是完整图学习/内存证据。
- 感觉 float32 在临界处的事件时刻差异是真实且已展示的数值变更；旧 float64 state 不能声称无损转移。
- 独立 CPU 参考只审查已完成 GPU 结果；float32 的极小二阶矩仍可能下溢，本模块不通过虚构非零值改变 Adam 数学。
- dataclass 冻结字段，模块自身不原位修改传入 state；Torch 张量不提供全局只读标志，调用方仍须遵守状态所有权，不自行原位修改。
- 更换损失、递归反向范围、旧 Adam 继承/重置、完整旧终档导入与后续训练都由 A 诊断及 B 集成 Spec 决定，不属于本批。

## Remaining Questions

本基础模块无待用户决策事项。等待新的独立验收；后续集成者按诊断证据确定训练数学与迁移。
