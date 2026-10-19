# Acceptance — 全 GPU 感觉与优化器基础模块

## Result

**FAIL**。GPU 驻留、优化器独立算术及默认感觉序列符合基础规格，但合法感觉配置可以产生模块自身不能再次接受的相位状态。需局部返工后由新的独立验收者复核；此结论不涉及另一个 Agent 的梯度诊断模块，也不否定已通过的原检查。

## Spec Coverage

- **F1：部分满足。** `GPUDrive.mask/step` 的阈值、池化、相位、事件数与完整增量均为 MPS 运算；每步不原位修改输入 state，快照和恢复严格检查 dtype、shape、drive/map ID、时钟及范围。默认固定序列的跨帧恢复和重算已有实际证据，float64→float32 临界事件差异及新身份明确。下述合法配置边界违反 `phase∈[0,1)` 的连续状态契约，故 F1 未完成。
- **F2：满足已登记基础模块范围。** 唯一主参数、原始梯度、稳定裁剪、fused Adam 及全部 m/v/step 均在 MPS；`require_mps()` 明确要求 fallback=0。`step()` 未将 NumPy/CPU 算术结果回写模型。分组缩放裁剪在约 `1e30` 输入下数学正确，极小梯度在不需裁剪时保留其数值；fused Adam 数值和恢复采用独立 float64 参考及小梯度非零/符号检查。设备标量条件会同步到 Python 作校验，不是 CPU 优化器数值主路径。
- **F3：旧来源与范围符合，最终验收未完成。** 本验收只写该报告，没有修改业务代码、旧资产或其它 Agent 文件。只读 SHA 验证 r3 的 42 份绑定源均保持原字节；无模型、完整图或正式训练运行。

## Validation

- Risk Level：**S3**，仅新感觉/优化器的数值与状态边界。
- Evidence Reviewed：Ready `gpu-residency-foundations.md`、实施报告、两个新增模块、`tests/test_gpu_foundations.py`、现有感觉配置与像素规则、MPS 准入、项目规范及 Decision 015。
- 采信已收取的唯一原检查：session 87141，exit 0，`2 passed in 2.12s`。审查了测试中的四次连续 fused Adam 更新、第三步后恢复、独立 float64 算术、零/极大/小梯度、非有限拒绝、感觉跨帧重放和旧 float64 差异；没有重复运行该检查。
- 由于发现具体合法输入边界，经协调者确认当前无其它数值作业后，仅执行一次 `workspace_lock()` 内的小型 MPS 复现。命令进程 exit 0，耗时约 1.5 秒；未加载完整图、没有优化更新或训练。
- 同次 stdlib SHA 复核：`bound_sources=42, changed=[]`。

## Issues

### R1 — 合法单步满事件配置会返回非法 phase（P2）

位置：`src/cnsbridge/senses/gpu_drive.py:124–127`。

`SpatialDriveConfig` 明确允许 `rate*dt/1000 == 1`。取 `dt_ms=1`、`max_rate_hz=1000`，所有输入占有率为 1，合法输入 phase 为 float32 `nextbelow(1)`（`0.9999999403953552`）。实际 MPS 相加先舍入到 2，随后只减去一个事件，返回 phase **1.0**。该结果立即被 `GPUDrive.validate()` 拒绝：`Invalid device phase state`。因此一次合法调用生成的 state 无法保存或作为下一步输入。

实际最小复现结果：

```json
{"config":{"dt_ms":1.0,"max_rate_hz":1000.0},"before":0.9999999403953552,"after":[1.0],"event_counts":[1],"step":1,"increment":[68.75],"next_state_validation":"Invalid device phase state"}
```

默认 `dt_ms=0.1, max_rate_hz=150` 未在本次审查发现该边界；问题属于公开接受配置的状态闭包错误，不是原默认序列检查失败。

## Regression Risks

- 感觉 float32 事件与旧 float64 并非逐位等价，已有显式身份和反例，必须由后续集成维持这种披露。
- GPU 阈值/状态校验每神经步存在设备同步，完整图吞吐尚未由本基础模块检查证明。
- 独立算术检查不能代替完整模型的新损失、反向范围、资源、迁移及学习门槛；这些属于后续集成范围。

## Required Rework

1. 对 R1 明确并实现 float32 状态边界规则，使全部接受的配置和合法输入 state 产生可保存、可继续的合法 state；若收窄配置范围，必须显式拒绝并更新规格/数值身份，不能静默返回非法值。
2. 增加或调整一个直接覆盖 R1 的定向检查，至少验证该输入的一次推进及后续继续/恢复，并说明事件数与相位的预期。只运行与本次局部修复直接相关的最小检查，不重复无关优化器或完整图测试。
3. 由新的独立验收者审查修复、实际证据和身份影响，保留本次 FAIL 记录。
