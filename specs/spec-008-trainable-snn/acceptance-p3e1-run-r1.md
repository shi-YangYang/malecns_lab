# Acceptance — P3e1 正式 R 首两段完整规模续接

## Result

**PASS**。由未参与生成器实施、软件验收或实际运行的新独立验收 Agent，检查 [Ready 协议](p3e1-bounded-rewiring-draft.md)、[实施报告](implementation-p3e1-generator.md)、[软件 PASS](acceptance-p3e1-generator.md)、实际执行源码及 [首两段报告](p3e1-run-20261019-r1.md)。`data/topology_generation_main_20261019_r1/` 的两个 100000 提案段满足实际准入门槛，可保留该唯一前缀，继续显式调用每段最多 1000000 提案。

本 PASS 不批准任何中间文件作为最终研究图，不表示已达到总预算、完成最终机械诊断、充分混合、取得神经或学习收益，也不关闭 AC6。

## Spec Coverage

| 要求 | 实际证据及独立判断 |
| --- | --- |
| 冻结登记与完整母本 | run ID `b4eba3601105fe9ec09ed7ae272b8012cbf679303830ab7e30be4bd75e8e4d71` 的规范 JSON 身份有效。当前 13 个绑定源码与登记 SHA 一致，运行环境相同；母本文件 SHA、完整来源对象及其规范来源 ID 相符。固定 seed 1901、成功上限 25582837、提案上限 102331348、完整 166700 节点/25582938 边/124177617 突触计数/101 自环/3377 输入/1314 DN 均在登记与执行校验路径中，未使用旧成本探针作为初态。 |
| 两段真实完成与父批准 | 两份原监督 `result.json` 与各自 `supervisor_result.json` 字节相同，均为 `completed / worker_exited / returncode 0 / worker_status completed`；worker 日志中的 JSON 与 worker_result 相同。父 assessment 均为 `completed / registered_proposals_and_terminal_evidence_verified`，规范身份、每段完整 10 文件清单及实际文件 SHA 全部相符。不是仅根据退出零或文件存在判定。 |
| 首段独立锚点 | 段 1 达到 100000 提案、51847 成功；实际 checkpoint 逻辑图 SHA `18a947ba127f094693c8994acbc999590559c8d0a0ad06c707cb3265810a6292` 与旧 P3e0 creation.json 精确相同，全部六类拒绝计数也相同。不同用途的来源/文件身份不要求相同；未重跑旧探针。 |
| 实际跨进程恢复 | 段 2 registration 的 start_state、parent_checkpoint 和 prefix，与段 1 父批准状态一致；段 2 resume.json 的 actual_state 逐字段相同，包括完整 CSC 身份、母本/节点元数据、全部计数和 PCG64 的 state/inc/has_uint32/uinteger。源码先恢复实际数组并执行完整结构/不变量核对，再比较实际身份、写恢复证明，最后才执行新提案。段 2 从 100000 推进到 200000，新增 51983 成功，未重新设 seed。 |
| 真正中间数组 | 两份非压缩 arrays.npz 各 307662812 bytes。独立流式检查 6 个 NPY 头和原始数组字节 SHA，与封存描述相同：每段 data 为 25582938 个 int64、indices 为同数 int32、indptr 为 166701 个 int32，矩阵 shape 为 166700×166700。检查点 schema/purpose 明确为中间生成状态，未伪装成脑资产。 |
| 机械校验的证据范围 | 已核对实际绑定 worker 在初始加载时验证完整母本计数/输入输出选择，每段提案后调用 validate_rw，第二段恢复时再次检查真实 CSC；该验证包括节点/nullable 注释/群组、规范矩阵、符号与度数、原自环、逐源强度和逐源权重多重集。实际 worker 完成、源码与数组绑定均成立。此次验收未重新加载完整图或独立重算这些全量机械指标；流式摘要检查不冒称为第二次机械复算。 |
| 预算与资源 | 两段配额均为 100000，计数非布尔整数、提案守恒、成功及各拒绝计数单调、未触达总边界。1 GiB 额外预留、申请 6 GiB、有效 RSS 为 min(6 GiB, available−1 GiB)、600 秒/0.1 秒采样均与登记相符；61/82 条实际记录有限、时间单调、未越界，峰值与监督结果相同。 |
| 非终局准入 | 两段 terminal 均 false，没有 candidate.npz 或最终 diagnostics.json。实际调用 accepted_candidate 后被明确拒绝：`No parent-approved terminal research candidate`。中间状态不能因此进入后续神经研究工厂。 |

## Validation

- Risk Level：**S3 对应的实际持久化/续接门槛审查**；本次文件变更仅为 S0 验收记录，不修改软件。
- 阅读 AGENTS、使命/技术栈、项目规则、独立验收流程、Spec 008/Plan、任务交接及 Decision 009/011/012；阅读 P3e1 协议、实施、软件验收、实际报告及生成/恢复/父证据/资源监督的执行路径。
- 一次轻量 Python 检查（`PYTHONPATH=src .venv-train/bin/python - <<'PY' ...`，工具执行结果 chunk `d23852`）直接收取 **exit 0 / PASS**。以独立 JSON 规范摘要、CSV 记录、流式文件 SHA/NPY 头与原始 payload 摘要核对上述登记、20 个段证据文件、6 个数组和完整父链；NumPy 只解码母本的小 meta 字段、构造/恢复 PCG64，不加载图矩阵。检查同时实际调用终局准入函数并核对它拒绝中间状态；末尾确认 `torch` 未导入。
- 资源与实际结果复核如下：

| 段 | 累计提案/成功 | 监督/worker 秒 | 采样峰值 RSS bytes | 最低 available bytes | 采样数 |
| --- | --- | --- | --- | --- | --- |
| 1 | 100000 / 51847 | 6.405234708 / 6.285667584 | 1836498944 | 4993892352 | 61 |
| 2 | 200000 / 103830 | 8.639986250 / 8.482966917 | 1800830976 | 5008752640 | 82 |

- 最后已批准 checkpoint 为 `ab4783de23b8b98aeb27d67057f3096c3ab15f33ae547992aad82cf90da4714e`，逻辑图 SHA `f12a6c4c069e209a7a468a482283af51201af44cef35602642caf8c593218585`，前缀 `d4a9c03c5042f49e04d2fa4e0df49469a100d3f52b93555c258792afd3a05212`。
- 未重跑已通过的 37 个软件检查、旧探针、提案或模型；未操作 Noria/Docker、硬件或系统资源配置。协调者在本验收期间启动的 G 训练与本次只读小文件/流式核对并行，未进行并行全矩阵加载或模型计算。
- 未修改业务源码、原始数据、既有报告或创建 commit；其他工作区改动不归本验收产出。训练第 13 段不是本 R 实际门槛的验收对象，本报告不追加训练准入结论。

## Issues

未发现阻止同一唯一前缀扩大单段配额的问题。实际报告数据与核对结果一致。协议及交接中的“实际门槛待执行/待验收”阶段文字由协调者交付时更新，不改变冻结科学预算。

## Regression Risks

采样峰值不是瞬时硬上限；100000 提案段的成本不保证 1000000 提案段、增长的父链校验或最终诊断均能在预算内完成。最后仍需全节点入强度、边重叠、类型对、实际输入/DN 路径、research_control 封存及实际重载与父批准。保留所有失败/中间证据，不因后续预算失败重置种子、丢弃前缀、删诊断或提升科学预算。

## Required Rework

无。协调者读取本报告并提交阶段结果后，可在同 run 显式续段；后续正式 R 完成和神经 G/R 研究各须对应证据。本次实际门槛验收到此停止。
