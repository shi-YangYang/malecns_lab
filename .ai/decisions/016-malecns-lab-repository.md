# Decision 016 — 统一为 MaleCNS Lab

## Context

2026-10-20 用户要求先整理仓库，并明确选择“项目和 Python 包名一起改成 malecns_lab”。近期方向为完整 MaleCNS 神经模型、三维活动展示及 G2/X2 机器人接入；SDK 文档可获取不等于设备已经接通。

## Decision

- 项目展示名 MaleCNS Lab；分发包、Python 包与 CLI 均为 `malecns_lab`。
- 授权 `src/cnsbridge` → `src/malecns_lab` 的单一包名迁移，现有内部职责与顶层 Agent 结构保持。
- 当前入口、文档、测试和示例同步迁移，不保留旧 import/CLI 转发层。
- 原有 `cnsbridge.*` / `flydrones.*` 持久化 schema、历史报告、数据/模型/日志、源代码 hash 校验和上游版权保留。
- 统一 README 与文档索引，将目标跟随列为现有实验；神经活动 WebGL 与智元适配列为后续工作。本批不实现它们。
- 当前 checkout 根目录和 Git 历史保持；GitHub 仓库尚未创建，不推送。本批不启动训练或硬件。

## Reason

新的统一名称容纳多种身体与神经可观测性研究；保持数据身份和实验来源使品牌迁移不改写已有证据。

## Alternatives

只改展示名而保留 cnsbridge 包名：用户明确选择一并改名。
批量替换 schema 并重写历史：不采用，会破坏已有格式与来源可追溯性。

## Consequences

旧 CLI/import 停止使用，两个本地 editable 安装需更新。源码内容/路径变化会改变实验身份，旧训练 run 不能在新源码下伪装原位续接；须使用原 commit 恢复原实验环境，或以后另立有明确父来源的迁移协议。

本决策覆盖此前治理文件中要求维持当前 cnsbridge 名称/业务路径的工程约定，但不改写那些决策的历史事实。执行范围见 [Spec 009](../../specs/spec-009-malecns-lab/spec.md)。
