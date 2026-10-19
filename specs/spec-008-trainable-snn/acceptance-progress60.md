# Acceptance — joint 60 窗与 frozen 6 窗进展记录

## Result

PASS

2026-10-19。本报告由未参与七份状态文档编辑的独立验收者完成，仅验收本批记录与现有运行证据一致，不新增训练数值或行为验收结论。

## Spec Coverage

- 核对 `.ai/prompts/task-handoff.md`、`README.md`、`constitution/roadmap.md`、`specs/README.md`、本 Spec 的 `plan.md`、`p3c-continuation-20261019.md` 与 `p3c-frozen-run-20261019-r1.md` 的本批 diff。状态一致更新为 joint 60/160、frozen 6/160；尚未完成完整训练、闭环、拓扑对照及实飞，AC3–AC6 保持开放。
- joint 第 17 段父结果为 `completed / registered_windows_verified`，监督返回码为 0，新增窗口 57–60。游标 forward 帧 32→40、更新 56→60，脑与感觉步数 16600→20600；累计 120 帧、剩余 100 窗。末档 `edc31649f0bf21a956233aa376d56f0b04864efdb5596f41099e4ee17517430c`、前缀 `b0efaa92c7613b94074df6298582d16bf37b40c94f74c3a97d2754b896fa701e` 及段 ID 均与 JSON 声明一致。
- frozen 第 3 段父结果同为 `completed / registered_windows_verified`，监督返回码为 0，新增窗口 3–6。游标 combined 帧 4→12、更新 2→6，脑与感觉步数 2600→6600；累计 12 帧、剩余 154 窗。末档 `d4b68da4308e2bb03438bb05787a330067a58b68d461611a23875ce086ec3f66`、前缀 `497abacf552cac6638f07b735acc13c4378b5a7677a9cdbbeaf4cb658e2812f2` 及段 ID 均与 JSON 声明一致。
- 两段的启动预算均保留 1 GiB，申请 6 GiB、有效上限按可用内存扣预留计算，600 秒上限及 0.1 秒采样不变；均无资源停止。joint 的 4050 个 CSV 样本给出峰值 RSS 1,784,725,504 bytes、最低 available 2,644,459,520 bytes，监督/worker 耗时 437.415638/436.027995 秒；frozen 的 823 个样本为 1,566,949,376 / 2,685,353,984 bytes，87.800171/85.952157 秒。均与文档的精度和舍入一致。
- 八窗的更新前动作 MSE、前向和反向/裁剪耗时、参数改变计数，以及 joint 原始 theta 梯度范数均与 `event.json` 一致。frozen theta 改变数逐窗为 0，读出权重/偏置分别改变 7884/3；joint 末窗 vx 为 +0.30000001192092896，而标签区间为 −0.019197015912620574 至 −0.012236834786055395。文档保留该不匹配，未把内部更新或数值通过解释为学会控制。
- 运行前后原始计数摘要在两份 worker 结果中相同。文档将当前 4 线程 CPU 路径、现场 CPU 采样和待执行的反向核基准与训练效果分开；没有把约 85–87 秒的反向/裁剪合并阶段称为已测得的纯 SpMV 成本，也没有提前宣称端到端加速。

## Validation

- Risk Level: S0。
- Evidence Reviewed / Checks Performed: 阅读本批七文件 diff、项目约束、Spec/Plan/训练协议；仅用 Python 标准库读取 joint `segments/000017` 和 frozen `segments/000003` 下的登记、恢复、父/worker/监督结果、各窗 event/window JSON，并汇总资源 CSV。
- 哈希与身份核对限于本批文档和已封存 JSON 声明的对应关系；未重新哈希完整资产、重算训练数组、调用父验证器或加载完整图。父 `registered_windows_verified` 已执行的数值与来源核验作为既有证据引用。
- 未重跑首窗、跨进程或首次续段门槛；本批没有独特的新数值实现或未覆盖的续段边界需要另设重复门槛。未运行测试、build、lint 或 typecheck；未导入 Torch/NumPy，未读取 NPZ 数组。
- 约 266.4% CPU 的三秒采样属于协调者本轮现场观察，资源 CSV 仅包含墙钟、进程树 RSS 与系统 available；本验收不将该 CPU 数值或 Noria/Docker 状态冒充 CSV 保存的遥测证据。

## Issues

无阻断问题。较低采样 RSS 不能证明模型所需内存变小；文档已保留这一限制。CPU 利用率快照与既存 swap 不能单独确定瓶颈，后续隔离基准仍需实际证据。

## Regression Risks

本批只更新进展与报告，没有修改训练来源或协议。上述 PASS 不覆盖并行实施的反向基准工具，不代表完整 epoch、冻结脑收益、拓扑贡献或无人机控制已通过。

## Required Rework

无。S0 范围已足够，停止验证。
