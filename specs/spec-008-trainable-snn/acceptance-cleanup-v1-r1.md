# Acceptance — V1 清理文档返工 R1

## Result

PASS

独立验收日期：2026-10-17。本轮仅验收 [R1 返工](rework-cleanup-v1.md) 的两处 S0 文档修正，未参与实施；原 [FAIL](acceptance-cleanup-v1.md) 保留。结合原验收已确认的业务删除、保留边界与测试证据，本清理子批的两项待返工问题均已关闭。本结论不覆盖并行的可训练核心、全图探针或跟随环境，也不代表 Spec 008 总目标完成。

## Spec Coverage

- `docs/MALECNS_AUDIT.md` 的 collector 引用已明确为 `747a8aa:src/cnsbridge/diagnostics/collector.py:72` 历史源码身份；该提交中的文件和 `ActivityBrain` 定义实际存在。文案保留审计时行为，明确 V1 入口已移除，并指向实际存在的 `MODEL_V2_USAGE.md`。
- `docs/MODEL_V2.md` 明确旧资产保留不代表已删除的 V1 离线流程仍可运行；阶段表改列现行 V2 模块，表中模块均存在。P0–P3 工具验收与未完成的动作有效性、结构优势和 P4 实体研究得到区分，没有继续推荐已删除的 prepare/decoder 作为未来实现入口。
- 两文档中剩余旧 collector/prepare/decoder 源码引用均有历史身份与移除说明；现行诊断、学习、研究入口分别指向已有 V2 文档。
- R1 只涉及指定文档和实施报告，没有改动业务逻辑，也没有删除或覆盖原 FAIL 记录。

## Validation

- Risk Level: S0。
- 阅读项目规则、独立验收流程、当前交接、Decision 009、清理子规格、原验收、返工要求和 R1 实施报告。
- 独立审阅两份文档的实际 diff，定向搜索旧模块引用，并核对新模块/文档路径与历史提交中的 collector 定义。
- 审阅 R1 实施者的限定路径 `git diff --check` 通过记录，未无理由重复执行。
- 原清理实施者已执行的 20 项定向测试证据沿用；本轮不改可执行逻辑，未重复 pytest，未执行训练、全图诊断、构建或硬件操作。

## Issues

无本轮阻塞问题。

## Regression Risks

V1 离线入口被移除后旧调用者需迁移至 V2，这是已授权行为；本次修正使文档反映该边界，不提供兼容包装。历史资产和旧实飞证据不因此转换成新模型的训练或实体有效性证据。

## Required Rework

无。交协调者更新清理子批状态并单独提交；本验收者未修改业务代码、未 commit。
