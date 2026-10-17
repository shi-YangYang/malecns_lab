# 清理返工 R1

状态：Ready；范围 S0。独立[原验收 FAIL](acceptance-cleanup-v1.md)发现两处文档遗漏，业务删除与20项检查无其他发现。

1. `docs/MALECNS_AUDIT.md` 第231行附近的 collector 活动链接，改为原 `747a8aa` 提交的历史引用，保留原审计结论并指向现行 MODEL_V2_USAGE。
2. `docs/MODEL_V2.md` 第260–270行旧流程可执行性及未来旧模块列表，明确历史设计或更新为现行 V2 模块，不能推荐已删 prepare/decoder。

由新的实施者只改这两文档并写 implementation-cleanup-v1-r1.md；不改数值核心/环境/清理业务、不重复pytest。新的独立验收者检查diff和相关引用，记录acceptance-cleanup-v1-r1.md。
