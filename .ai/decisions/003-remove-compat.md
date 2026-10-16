# Decision 003 — 仅保留新的业务入口

日期：2026-10-16。基线：`5bf6642`。

## Context

用户在业务重构后明确要求删除旧兼容层，统一新目录、入口和文档命令，完成后提交。

## Decision

移除 motor 转发包、根 safety/calibrate 转发模块和 tools 七个启动脚本。使用真实 control/learning/experiments 实现，命令使用 `python -m`；原 cnsbridge CLI 不变。本决策替代 Decision 002 的“保留旧导入及脚本入口”要求。

## Reason / Alternatives

用户选择直接切换新入口；继续保留弃用包装与本次要求冲突。新模块已有可运行入口，无需再新增包装或修改算法。

## Consequences

旧导入与脚本路径停止支持，文档和入口测试同步更新。序列化的 flydrones.* 标识、人工读出基线和实验资产保留。过去的 Spec/验收记录保持历史事实，当前规则按本决策更新。

详见 [Spec 003](../../specs/spec-003-remove-compat/spec.md)和[Plan](../../specs/spec-003-remove-compat/plan.md)。
