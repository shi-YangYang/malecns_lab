# Acceptance — P3d 终态准入返工 R1

## Result

PASS

本验收者未参与原实施、原验收或本轮返工。已独立审查当前实际文件；[原 FAIL](acceptance-p3d.md) 的 R1、R2 已关闭，原报告保留。本结论只覆盖评估工具的软件契约，不表示两种最终训练均完成、完整图闭环已执行或跟随成功；AC3–AC6 和整体目标保持未完成。

## Spec Coverage

- **R1：最终训练来源已要求父终态。** `evaluation_contract.training_terminals` 在既有唯一链/数值扫描之后逐段读取必需的父 result 与原始 supervisor，核对 run/segment/schema/scope、实际结束窗口、完整可恢复前缀和拒绝继续原因。它比较实际存在的终态文件清单与父 SHA 清单，逐文件复算，并核对 worker 身份、失败类型及 phase/resume 文件绑定。completed 段须具备固定的完整文件集合、无未封存尾部，并通过既有 `validate_segment_completion`；最终段不能是 failed。
- **R1：合法历史失败未被误删或一律排除。** 中间段只允许父已验证的资源/IO失败，数值/协议失败、父证据无效、缺父 result、错身份/前缀均拒绝。后续段仍由旧 scanner 核对连续游标、父存档、参数、Adam、状态和采样边界；不能仅凭后续成功越过坏前缀。每段 result SHA 及完整终态文件 SHA 加入 `finalized_segments`，经最终来源、套件 ID、condition source 绑定；`validate_suite` 会重新准入并比较全部来源。
- **R2：环境终止与正式证据完成已分开。** `scan_episode` 保留重放得到的 terminal，但只有终局所在段的父结果为经过验证的 completed 才设置 `evidence_complete`。当前尚未封存父结果的段不会由扫描器提前完成；`finalize_segment` 仅在当前完成检查通过后更新该字段。失败终局保留原始段状态、环境终止和诊断评分，正式 metrics.success 强制为 false。
- **R2：所有放行和比较使用同一正式完成判据。** episode report、冻结顺序、val_complete/test_complete、验证汇总封存、固定场景均值及逐场景配对均使用 `evidence_complete`。`check_order` 还明确拒绝对未完成证据的 terminal 继续前向，register_segment 也保留环境终止限制；没有新增自动复核放行、重新前向或静默重试。非终局资源/IO前缀可显式续段，旧失败状态保持；正常封存的物理失败是有效研究终局，行为 success 仍为 false、分母仍为80。
- **原协议其余边界保持。** 复核六个新增评估模块及直接调用接口：实际参数数值身份独立于结构 ID、acute 在初始化前恢复、独立无 Adam 完整状态支持奇数帧、神经接口仅接 BGR/自身状态、环境从执行动作重放、父独立复算读出/评分、基线不虚构神经证据、共享 OS 锁及 1 GiB/6 GiB/600 秒预算均保持。未改训练、模型、图、数据或评分核心。使用文档已说明失败终局与正式完成的区别。

## Validation

- Risk Level: **S3**，范围为新评估子系统的准入、持久证据和完成状态，不扩大到全项目或旧训练数值检查。
- Evidence Reviewed / Checks Performed：阅读项目 AGENTS、使命/技术栈、项目规则、验收工作流、Spec 索引、Spec 008/Plan、当前交接、Decision 009/011/012、Ready P3d、独立协议报告、原实施/FAIL、Ready返工及完整返工实施报告；逐文件审查六个新增业务模块、定向测试和使用文档，包括尚未跟踪的文件，核对 `train_snn`、`training_evidence` 的真实父终态/恢复契约。
- 实施者已收取 session **45460** 的 exit 0：**25 passed、20 deselected，4.58秒**。其中20项是本轮新增准入/终局交叉反例，5项是直接受影响的既有检查；剩余20项有前轮通过记录。本验收审查了这些用例的实际断言和隔离范围，没有无理由重跑。
- 最终训练准入 fixture 只隔离先前验证的链扫描和最终数组读取，磁盘上的40段父/worker/supervisor及清单由新增准入层和既有完成检查实际处理。它覆盖合规链、资源/IO历史后完成、缺父、坏证据、缺 hash、错父/worker身份、数值失败、监督不符、最终失败和错前缀；不把 fixture 当作160窗生产训练。
- 终局用例覆盖 duration/物理失败与 completed/资源/IO结果的交叉，失败不会开放后续场景/test、均值或配对；另有非终局IO前缀续段并验证原 failed 字节保持。已有两个真实数值子进程的奇数边界恢复证据保留，不因本轮完成状态修正重复计算。
- 返工报告记录标准库核对当前 run 的29个训练绑定源码，`mismatches=[]`。本验收检查工作区范围并阅读受影响调用，未发现训练业务源码修改。协调者已有的训练进展/Handoff修改不归入本批实现。
- 本验收未运行测试、导入 Torch、加载图、启动硬件或操作服务；只新增本报告，未修改业务实现、原 FAIL 或实验资产，未 commit。

## Issues

未发现本轮范围内尚未关闭的阻塞问题。

## Regression Risks

完整训练来源准入和完整图评估的实际耗时仍需后续生产证据；小数值检查不能代替完整图首次1+1帧恢复验收。失败发生在终局之后时，当前实现会保留终局并停止矩阵推进，这是已冻结的保守行为；若将来需要专门恢复机制，须另立协议。不能把本次软件 PASS 解释为内部学习收益、拓扑贡献、实时性或实飞成功。

## Required Rework

无。协调者可读取本报告、更新本批状态并按已有授权提交软件；正式评估仍须等待 joint 与独立 frozen_brain 的完整最终训练链，并遵守首次实际门槛。
